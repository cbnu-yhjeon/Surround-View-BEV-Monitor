"""3D 박스를 BEV · 카메라 영상 위에 투영하고 TTC 색으로 표시 (README G3).

IPM은 지면만 정확하다. 차량·보행자처럼 높이가 있는 물체는 카메라에서 멀어지는
방향으로 길게 늘어진다. 그래서 박스의 **바닥 사각형**을 BEV 좌표로 바로 그려서
"실제로 어디 있는지"를 덧그린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.bev.ground_homography import BEVGrid
from src.calib.camera import CameraModel, Pose
from src.constants import CAMERAS, EGO_X_MAX, EGO_X_MIN, EGO_Y_HALF
from src.data.nuscenes_lite import Box, Frame
from src.overlay.ttc import compute_ttc, risk_level

RISK_CATEGORIES = ("vehicle.", "human.", "animal")
EGO_CENTER_X = (EGO_X_MIN + EGO_X_MAX) / 2
_RANK = {"critical": 0, "warning": 1, "caution": 2, "passing": 3, "safe": 4}


@dataclass
class RiskObject:
    box: Box                  # 기준 시각 자차 좌표계
    ttc: float
    dca: float
    dist: float
    level: str
    color: tuple[int, int, int]
    camera: str | None        # 가장 잘 보이는 채널

    @property
    def rank(self) -> int:
        return _RANK[self.level]


def is_risk_category(category: str) -> bool:
    return category.startswith(RISK_CATEGORIES)


def best_camera(frame: Frame, p_ego: np.ndarray) -> str | None:
    """점이 영상 안에 들어오는 채널 중 광축과 각도가 가장 작은 것."""
    best, best_cos = None, -2.0
    for c in CAMERAS:
        cam = frame.cameras[c]
        uv, z = cam.project(p_ego[None])
        if not cam.in_image(uv, z)[0]:
            continue
        d = p_ego - cam.center_ego
        cos = float(d @ cam.axis_ego / (np.linalg.norm(d) + 1e-9))
        if cos > best_cos:
            best, best_cos = c, cos
    return best


def assess(frame: Frame, boxes_global: list[Box], max_dist: float = 40.0) -> list[RiskObject]:
    """GT 박스 → 자차 좌표계 + TTC/DCA + 색. 위험 순(rank, TTC, 거리)으로 정렬."""
    T = frame.ego_from_global
    ego_center_g = frame.ref_pose.apply(np.array([EGO_CENTER_X, 0.0, 0.0]))
    out = []
    for b in boxes_global:
        if not is_risk_category(b.category):
            continue
        be = b.transformed(T)
        dist = float(np.hypot(be.center[0] - EGO_CENTER_X, be.center[1]))
        if dist > max_dist:
            continue
        v = b.velocity if np.all(np.isfinite(b.velocity)) else np.zeros(2)
        ttc, dca = compute_ttc(b.center, ego_center_g, v, frame.ego_velocity)
        # 충돌로 볼 최접근 거리: 자차 반폭 + 객체 반폭 + 여유
        dca_hit = EGO_Y_HALF + 0.5 * float(b.size[0]) + 0.5
        level, color = risk_level(ttc, dca, dca_hit)
        out.append(RiskObject(be, ttc, dca, dist, level, color, best_camera(frame, be.center)))
    out.sort(key=lambda r: (r.rank, r.ttc, r.dist))
    return out


def attention_camera(risks: list[RiskObject], default: str = "CAM_FRONT",
                     tau: float = 3.0) -> str:
    """채널 점수 s(c) = Σ exp(−TTC/τ) (형제 연구의 위험도 항). 위험 없으면 기본 채널."""
    score = dict.fromkeys(CAMERAS, 0.0)
    for r in risks:
        if r.camera and r.level in ("critical", "warning", "caution"):
            score[r.camera] += float(np.exp(-r.ttc / tau))
    best = max(score, key=score.get)
    return best if score[best] > 0 else default


# ---------------------------------------------------------------- 그리기
def draw_ego(img: np.ndarray, grid: BEVGrid) -> None:
    corners = np.array([[EGO_X_MAX, EGO_Y_HALF], [EGO_X_MAX, -EGO_Y_HALF],
                        [EGO_X_MIN, -EGO_Y_HALF], [EGO_X_MIN, EGO_Y_HALF]])
    px = np.round(grid.to_pixel(corners)).astype(np.int32)
    cv2.fillConvexPoly(img, px, (60, 60, 60), cv2.LINE_AA)
    cv2.polylines(img, [px], True, (230, 230, 230), 2, cv2.LINE_AA)
    nose = grid.to_pixel(np.array([[EGO_X_MAX - 0.3, 0.0], [EGO_X_MAX - 1.3, 0.6],
                                   [EGO_X_MAX - 1.3, -0.6]]))
    cv2.fillConvexPoly(img, np.round(nose).astype(np.int32), (230, 230, 230), cv2.LINE_AA)


def draw_range_rings(img: np.ndarray, grid: BEVGrid, radii=(5, 10, 15, 20)) -> None:
    c = grid.to_pixel(np.array([[EGO_CENTER_X, 0.0]]))[0]
    for r in radii:
        cv2.circle(img, tuple(np.round(c).astype(int)), int(round(r / grid.res)),
                   (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r}m", (int(c[0]) + 4, int(c[1] - r / grid.res) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def draw_risks_bev(img: np.ndarray, grid: BEVGrid, risks: list[RiskObject],
                   label: bool = True) -> None:
    for r in sorted(risks, key=lambda r: -r.rank):       # 위험한 것이 위에 그려지게
        bottom = r.box.corners()[:4, :2]
        px = np.round(grid.to_pixel(bottom)).astype(np.int32)
        thick = 3 if r.rank <= 1 else 2
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, px, r.color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        cv2.polylines(img, [px], True, r.color, thick, cv2.LINE_AA)
        front = grid.to_pixel(((bottom[0] + bottom[1]) / 2)[None])[0]
        ctr = grid.to_pixel(r.box.center[None, :2])[0]
        cv2.line(img, tuple(np.round(ctr).astype(int)), tuple(np.round(front).astype(int)),
                 r.color, thick, cv2.LINE_AA)
        if label and r.rank <= 2:
            txt = f"{r.ttc:.1f}s"
            p = tuple(np.round(px.min(0) - [0, 4]).astype(int))
            cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, r.color, 1, cv2.LINE_AA)


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_risks_camera(img: np.ndarray, cam: CameraModel, risks: list[RiskObject],
                      cam_from_ego: Pose | None = None, scale: float = 1.0) -> None:
    for r in sorted(risks, key=lambda r: -r.rank):
        uv, z = cam.project(r.box.corners(), cam_from_ego)
        if np.any(z < 0.5):
            continue
        if not cam.in_image(uv, z).any():
            continue
        p = np.round(uv * scale).astype(np.int32)
        thick = 3 if r.rank <= 1 else 2
        for a, b in _EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), r.color, thick, cv2.LINE_AA)
        if r.rank <= 2:
            q = tuple(p[4:].min(0) - [0, 6])
            cv2.putText(img, f"TTC {r.ttc:.1f}s", q, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, f"TTC {r.ttc:.1f}s", q, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        r.color, 2, cv2.LINE_AA)
