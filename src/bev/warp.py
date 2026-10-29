"""Inverse Warping → BEV 캔버스 (Lec 9 Forward / Inverse Warping).

Forward warping(영상 픽셀을 BEV로 밀어 넣기)은 구멍이 생긴다. BEV의 모든 픽셀에서
H·S로 원본 위치를 찾아 보간하는 inverse warping을 쓴다.

속도 (README 지표 "룩업 테이블 사전 계산 전후"):
  - `warp_perspective`: 매 프레임 cv2.warpPerspective — 좌표를 매번 계산.
  - `Lut` + `remap`: 좌표(map_x, map_y)를 한 번만 계산하고 cv2.remap 만 반복.
  - `Lut.fixed()`: convertMaps 로 고정소수점(CV_16SC2) 맵 — 가장 빠름.
캘리브레이션이 고정이면 LUT는 한 번 만들면 끝이다. 시간 보정(채널별 촬영 시각의
자차 이동 반영)을 켜면 프레임마다 LUT를 다시 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from src.bev.ground_homography import BEVGrid, bev_homography
from src.calib.camera import CameraModel, Pose
from src.constants import BODY_CROP_ROWS, EGO_X_MAX, EGO_X_MIN, EGO_Y_HALF


@dataclass
class Lut:
    map_x: np.ndarray        # (H,W) float32, 영상 x
    map_y: np.ndarray        # (H,W) float32, 영상 y
    valid: np.ndarray        # (H,W) bool
    _fixed: tuple | None = None

    def fixed(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fixed is None:
            self._fixed = cv2.convertMaps(self.map_x, self.map_y, cv2.CV_16SC2)
        return self._fixed


def ego_footprint_mask(grid: BEVGrid, margin: float = 0.0) -> np.ndarray:
    x, y = grid.ground_coords()
    return ((x >= EGO_X_MIN - margin) & (x <= EGO_X_MAX + margin)
            & (np.abs(y) <= EGO_Y_HALF + margin))


@lru_cache(maxsize=8)
def _static_masks(grid: BEVGrid, max_range: float | None) -> np.ndarray:
    """카메라와 무관한 제외 영역(자차 차체, 최대 거리 밖)."""
    keep = ~ego_footprint_mask(grid, margin=0.2)
    if max_range is not None:
        x, y = grid.ground_coords()
        keep &= np.hypot(x, y) <= max_range
    return keep


def build_lut(cam: CameraModel, grid: BEVGrid, cam_from_ego: Pose | None = None,
              min_depth: float = 1.0, max_range: float | None = None,
              border: int = 2, z0: float = 0.0) -> Lut:
    """BEV 픽셀마다 원본 영상 좌표와 유효 마스크.

    p = HS·[u, v, 1]ᵀ 를 행·열 1차원 배열의 바깥 합으로 계산한다(전체 격자 행렬곱보다
    수십 배 빠름) — 시간 보정을 켜면 매 프레임 호출되므로 중요하다.
    """
    HS = bev_homography(cam, grid, cam_from_ego, z0).astype(np.float32)
    u = np.arange(grid.width, dtype=np.float32)
    v = np.arange(grid.height, dtype=np.float32)[:, None]
    depth = HS[2, 0] * u + HS[2, 1] * v + HS[2, 2]      # K의 3행이 [0,0,1]이라 깊이와 같다
    front = depth > min_depth
    inv = np.where(front, 1.0 / np.where(front, depth, 1.0), 0.0).astype(np.float32)
    mx = (HS[0, 0] * u + HS[0, 1] * v + HS[0, 2]) * inv
    my = (HS[1, 0] * u + HS[1, 1] * v + HS[1, 2]) * inv
    bottom = cam.height - 1 - border - BODY_CROP_ROWS.get(cam.name, 0)
    valid = (front & (mx >= border) & (mx <= cam.width - 1 - border)
             & (my >= border) & (my <= bottom) & _static_masks(grid, max_range))
    mx[~valid] = -1
    my[~valid] = -1
    return Lut(mx, my, valid)


def remap(img: np.ndarray, lut: Lut, fixed: bool = True,
          interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    if fixed:
        m1, m2 = lut.fixed()
        return cv2.remap(img, m1, m2, interp, borderMode=cv2.BORDER_CONSTANT)
    return cv2.remap(img, lut.map_x, lut.map_y, interp, borderMode=cv2.BORDER_CONSTANT)


def warp_perspective(img: np.ndarray, cam: CameraModel, grid: BEVGrid,
                     cam_from_ego: Pose | None = None) -> np.ndarray:
    """LUT 없이 매 프레임 호모그래피로 바로 워핑(비교 기준선).

    주의: 카메라 뒤쪽(깊이<0) 점도 투영되므로 마스크는 LUT 것을 따로 써야 한다.
    """
    HS = bev_homography(cam, grid, cam_from_ego)
    return cv2.warpPerspective(img, HS, (grid.width, grid.height),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT)
