"""원통 투영 파노라마 스트립 (README G2, Lec 9 원통 투영).

자차 위 가상 원통(중심 c₀ = 카메라 중심들의 평균, 반지름 R)에 6채널을 붙인다.
파노라마 픽셀 (u, v) → 방위각 θ, 높이 h → 원통 위 3D 점

    P(θ, h) = c₀ + (R cos θ, R sin θ, h)

을 각 카메라로 투영해 inverse warping 한다. 카메라 중심이 c₀와 달라서 생기는 시차는
거리가 R인 물체에서만 0이다(R = 원통이 가정하는 장면 깊이). 가로축은 왼쪽 끝이
후방, 가운데가 전방, 오른쪽으로 갈수록 차량 우측 → 후방 순서.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.bev.blend import blend, feather_weights, winner_labels
from src.bev.gain import GainCompensator, apply_gains
from src.bev.warp import Lut
from src.calib.camera import CameraModel
from src.constants import BODY_CROP_ROWS, CAMERAS
from src.data.nuscenes_lite import Frame


class CylindricalPanorama:
    def __init__(self, width: int = 2400, radius: float = 10.0,
                 elev_up_deg: float = 15.0, elev_down_deg: float = 18.0,
                 method: str = "multiband", gain: bool = True, levels: int = 4):
        self.width = width
        self.radius = radius
        self.tan_up = np.tan(np.radians(elev_up_deg))
        self.tan_down = np.tan(np.radians(elev_down_deg))
        # 원통 전개면에서 가로·세로 축척이 같도록 높이를 정한다 (R·dθ = dh)
        self.height = int(round(width * (self.tan_up + self.tan_down) / (2 * np.pi)))
        self.method, self.levels = method, levels
        self.gain = GainCompensator() if gain else None
        self._cache: dict[tuple, tuple] = {}

    def _points(self, c0: np.ndarray) -> np.ndarray:
        u = (np.arange(self.width) + 0.5) / self.width
        theta = np.pi - u * 2 * np.pi
        v = (np.arange(self.height) + 0.5) / self.height
        h = self.radius * (self.tan_up - v * (self.tan_up + self.tan_down))
        T, Hh = np.meshgrid(theta, h)
        return np.stack([c0[0] + self.radius * np.cos(T), c0[1] + self.radius * np.sin(T),
                         c0[2] + Hh], -1)

    def _lut(self, cam: CameraModel, P: np.ndarray) -> Lut:
        uv, z = cam.project(P.reshape(-1, 3))
        crop = BODY_CROP_ROWS.get(cam.name, 0)
        valid = (z > 0.5) & (uv[:, 0] >= 1) & (uv[:, 0] <= cam.width - 2) \
            & (uv[:, 1] >= 1) & (uv[:, 1] <= cam.height - 2 - crop)
        shape = P.shape[:2]
        mx = np.where(valid, uv[:, 0], -1).reshape(shape).astype(np.float32)
        my = np.where(valid, uv[:, 1], -1).reshape(shape).astype(np.float32)
        return Lut(mx, my, valid.reshape(shape))

    def luts(self, frame: Frame):
        key = tuple((c, frame.cameras[c].K.tobytes(), frame.cameras[c].ego_from_cam.t.tobytes())
                    for c in CAMERAS)
        if key not in self._cache:
            c0 = np.mean([frame.cameras[c].center_ego for c in CAMERAS], 0)
            P = self._points(c0)
            luts = {c: self._lut(frame.cameras[c], P) for c in CAMERAS}
            w = feather_weights([luts[c].valid for c in CAMERAS])
            self._cache[key] = (luts, w, winner_labels(w))
        return self._cache[key]

    def compose(self, frame: Frame, images: dict[str, np.ndarray]) -> np.ndarray:
        luts, weights, labels = self.luts(frame)
        warped, masks = [], []
        for c in CAMERAS:
            m1, m2 = luts[c].fixed()
            warped.append(cv2.remap(images[c], m1, m2, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT))
            masks.append(luts[c].valid)
        if self.gain is not None:
            warped = apply_gains(warped, self.gain(warped, masks))
        return blend(warped, masks, weights, self.method, self.levels, labels)

    def azimuth_to_u(self, theta: float) -> float:
        """자차 기준 방위각(rad, 좌측 +) → 파노라마 가로 좌표."""
        return ((np.pi - theta) % (2 * np.pi)) / (2 * np.pi) * self.width
