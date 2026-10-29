"""채널별 지면 호모그래피 (Lec 9 호모그래피 · Lec 10 외부 파라미터).

자차 지면(z = z₀) 위의 점 (x, y)는

    p ≃ K · [R | t] · [x, y, z₀, 1]ᵀ = K · [r₁  r₂  (z₀·r₃ + t)] · [x, y, 1]ᵀ
    → H = K · [r₁  r₂  z₀·r₃ + t]                       (지면 → 영상)

BEV 픽셀 (u, v) → 지면 (x, y) 를 S 로 두면 BEV(u, v) = I( H · S · [u, v, 1]ᵀ ).
여기서 R, t 는 **자차 → 카메라** (calibrated_sensor 의 역변환).

BEV 캔버스 규약: 위쪽이 전방(+x), 오른쪽이 차량 우측(−y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calib.camera import CameraModel, Pose


@dataclass(frozen=True)
class BEVGrid:
    """BEV 캔버스. 기본: 40 m × 40 m, 5 cm/px → 800 × 800, 차체 중심(x≈1.3 m)이 가운데."""

    x_min: float = -18.7
    x_max: float = 21.3
    y_min: float = -20.0
    y_max: float = 20.0
    res: float = 0.05

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def S(self) -> np.ndarray:
        """BEV 픽셀 [u, v, 1] → 지면 [x, y, 1] (픽셀 중심 기준)."""
        r = self.res
        return np.array([[0.0, -r, self.x_max - 0.5 * r],
                         [-r, 0.0, self.y_max - 0.5 * r],
                         [0.0, 0.0, 1.0]])

    def S_inv(self) -> np.ndarray:
        return np.linalg.inv(self.S())

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """지면 (N,2) → BEV 픽셀 (N,2) 실수 좌표 (u, v)."""
        xy = np.atleast_2d(xy)
        u = (self.y_max - xy[:, 1]) / self.res - 0.5
        v = (self.x_max - xy[:, 0]) / self.res - 0.5
        return np.stack([u, v], 1)

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        x = self.x_max - (uv[:, 1] + 0.5) * self.res
        y = self.y_max - (uv[:, 0] + 0.5) * self.res
        return np.stack([x, y], 1)

    def ground_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """캔버스 전체의 지면 좌표 (H,W) x, y."""
        v, u = np.mgrid[0:self.height, 0:self.width].astype(np.float64)
        return self.x_max - (v + 0.5) * self.res, self.y_max - (u + 0.5) * self.res

    def scaled(self, factor: float) -> "BEVGrid":
        return BEVGrid(self.x_min, self.x_max, self.y_min, self.y_max, self.res * factor)


def ground_homography(cam: CameraModel, cam_from_ego: Pose | None = None,
                      z0: float = 0.0) -> np.ndarray:
    """지면(z=z0) → 영상 호모그래피 H (3×3)."""
    T = cam_from_ego or cam.cam_from_ego
    R, t = T.R, T.t
    M = np.column_stack([R[:, 0], R[:, 1], z0 * R[:, 2] + t])
    return cam.K @ M


def bev_homography(cam: CameraModel, grid: BEVGrid, cam_from_ego: Pose | None = None,
                   z0: float = 0.0) -> np.ndarray:
    """BEV 픽셀 → 영상 픽셀 호모그래피 H · S."""
    return ground_homography(cam, cam_from_ego, z0) @ grid.S()


def image_to_ground(cam: CameraModel, uv: np.ndarray, cam_from_ego: Pose | None = None,
                    z0: float = 0.0) -> np.ndarray:
    """영상 픽셀 (N,2) → 지면 (N,2), H⁻¹ 사용 (IPM 역투영)."""
    Hinv = np.linalg.inv(ground_homography(cam, cam_from_ego, z0))
    p = np.column_stack([np.atleast_2d(uv), np.ones(len(np.atleast_2d(uv)))]) @ Hinv.T
    return p[:, :2] / p[:, 2:3]
