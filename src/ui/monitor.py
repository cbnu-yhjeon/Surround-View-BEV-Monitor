"""관제 UI 데모 화면 (README 마일스톤 5).

    ┌──────────────────────────── 360° 파노라마 스트립 ─────────────────────────────┐
    ├──────────────────────┬─────────────────────────────────────────────────────────┤
    │                      │  attention 뷰 (위험 점수 최대 채널, 3D 박스 + TTC)       │
    │   BEV 합성 + 위험     ├─────────────────────────────────────────────────────────┤
    │   객체 오버레이       │  6채널 썸네일 (attention 채널 강조)                     │
    │                      │  위험 객체 목록 · 상태                                  │
    └──────────────────────┴─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.bev.ground_homography import BEVGrid
from src.bev.pipeline import SurroundBEV, load_images
from src.constants import CAMERA_SHORT, CAMERAS
from src.data.nuscenes_lite import Frame, NuScenesLite
from src.overlay.box_projection import (
    EGO_CENTER_X,
    assess,
    attention_camera,
    draw_ego,
    draw_range_rings,
    draw_risks_bev,
    draw_risks_camera,
)
from src.panorama.cylindrical import CylindricalPanorama

W, H = 1920, 1080
BG = (24, 24, 28)
FG = (230, 230, 230)
DIM = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.55, color=FG, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class MonitorRenderer:
    def __init__(self, ns: NuScenesLite, variant: str = "V4", time_comp: bool = True,
                 grid: BEVGrid | None = None, gain_alpha: float = 0.3):
        self.ns = ns
        self.bev = SurroundBEV(grid, variant=variant, time_comp=time_comp, gain_alpha=gain_alpha)
        self.pano = CylindricalPanorama(width=2400)
        self.pano.gain.alpha = gain_alpha
        self.variant, self.time_comp = variant, time_comp

    def render(self, frame: Frame) -> np.ndarray:
        t0 = time.perf_counter()
        images = load_images(frame)
        res = self.bev.compose(frame, images)
        t_bev = time.perf_counter()
        pano = self.pano.compose(frame, images)
        t_pano = time.perf_counter()
        risks = assess(frame, self.ns.boxes(frame))
        att = attention_camera(risks)

        canvas = np.full((H, W, 3), BG, np.uint8)
        # ---------- 헤더
        _text(canvas, "Surround View BEV Monitor", (16, 28), 0.8, FG, 2)
        sync = frame.sync_error_ms()
        worst = max(sync.values(), key=abs)
        t_rel = (frame.timestamp - self.ns.scene_samples(frame.scene)[0]["timestamp"]) / 1e6
        hdr = (f"{frame.scene}  t={t_rel:5.2f}s  "
               f"speed {frame.speed() * 3.6:5.1f} km/h  |  {self.variant}"
               f"{' + time-comp' if self.time_comp else ''}  |  cam sync max {worst:+.0f} ms")
        _text(canvas, hdr, (420, 28), 0.6, DIM)

        # ---------- 파노라마
        ph = int(round(W * pano.shape[0] / pano.shape[1]))
        py0 = 42
        canvas[py0:py0 + ph] = _fit(pano, W, ph)
        for c in CAMERAS:
            u = self.pano.azimuth_to_u(frame.cameras[c].yaw) * W / self.pano.width
            u = min(max(u, 12), W - 24)
            _text(canvas, CAMERA_SHORT[c], (int(u) - 8, py0 + ph + 16), 0.5,
                  (80, 220, 255) if c == att else DIM, 2 if c == att else 1)
        for r in risks:
            if r.rank > 2:
                continue
            th = np.arctan2(r.box.center[1], r.box.center[0] - EGO_CENTER_X)
            u = int(self.pano.azimuth_to_u(th) * W / self.pano.width)
            tri = np.array([[u, py0 + ph - 2], [u - 7, py0 + ph - 14], [u + 7, py0 + ph - 14]])
            cv2.fillConvexPoly(canvas, tri, r.color, cv2.LINE_AA)
        top = py0 + ph + 24

        # ---------- BEV
        bev = res.bev.copy()
        g = self.bev.grid
        draw_range_rings(bev, g)
        draw_ego(bev, g)
        draw_risks_bev(bev, g, risks)
        side = H - top - 8
        canvas[top:top + side, 8:8 + side] = _fit(bev, side, side)
        _text(canvas, "BEV", (16, top + 22), 0.7, FG, 2)

        # ---------- attention 뷰
        x0 = side + 24
        aw = W - x0 - 12
        ah = int(round(aw * 9 / 16))
        ah = min(ah, int((H - top) * 0.62))
        aw2 = int(round(ah * 16 / 9))
        cam_img = images[att].copy()
        T = frame.cam_from_ref_ego(att) if self.time_comp else None
        draw_risks_camera(cam_img, frame.cameras[att], risks, T)
        canvas[top:top + ah, x0:x0 + aw2] = _fit(cam_img, aw2, ah)
        cv2.rectangle(canvas, (x0, top), (x0 + aw2 - 1, top + ah - 1), (80, 220, 255), 2)
        _text(canvas, f"ATTENTION  {att}", (x0 + 10, top + 26), 0.7, (80, 220, 255), 2)

        # ---------- 썸네일
        ty = top + ah + 10
        tw = (aw2 - 5 * 6) // 6
        th_ = int(round(tw * 9 / 16))
        for i, c in enumerate(CAMERAS):
            tx = x0 + i * (tw + 6)
            canvas[ty:ty + th_, tx:tx + tw] = _fit(images[c], tw, th_)
            col = (80, 220, 255) if c == att else (90, 90, 90)
            cv2.rectangle(canvas, (tx, ty), (tx + tw - 1, ty + th_ - 1), col, 2 if c == att else 1)
            _text(canvas, CAMERA_SHORT[c], (tx + 6, ty + 18), 0.5, col, 1)

        # ---------- 위험 목록 / 상태
        ly = ty + th_ + 26
        _text(canvas, "RISK OBJECTS (TTC, cpa)", (x0, ly), 0.6, FG, 2)
        shown = [r for r in risks if r.rank <= 3][:5]
        for k, r in enumerate(shown):
            yy = ly + 24 * (k + 1)
            cv2.rectangle(canvas, (x0, yy - 12), (x0 + 14, yy + 2), r.color, -1)
            ttc = f"{r.ttc:4.1f}s" if np.isfinite(r.ttc) else "  inf"
            _text(canvas, f"{r.level:8s} {ttc}  dist {r.dist:4.1f}m  DCA {r.dca:4.1f}m  "
                          f"{r.box.category.split('.')[-1]:12s} {CAMERA_SHORT.get(r.camera, '-')}",
                  (x0 + 22, yy), 0.5, FG)
        if not shown:
            _text(canvas, "no approaching objects", (x0 + 22, ly + 24), 0.5, DIM)
        sx = x0 + aw2 // 2 + 120
        stats = [f"BEV  {res.timings['total']:5.1f} ms (warp {res.timings['warp']:.1f}, "
                 f"gain {res.timings['gain']:.1f}, blend {res.timings['blend']:.1f})",
                 f"pano {1e3 * (t_pano - t_bev):5.1f} ms",
                 f"frame {1e3 * (time.perf_counter() - t0):5.1f} ms (incl. JPEG decode)"]
        for k, s in enumerate(stats):
            _text(canvas, s, (sx, ly + 24 * (k + 1)), 0.45, DIM)
        return canvas
