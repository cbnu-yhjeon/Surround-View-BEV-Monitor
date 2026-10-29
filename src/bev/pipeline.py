"""6채널 BEV 합성 파이프라인: LUT 워핑 → (노출 보정) → 블렌딩.

| 변형 | 블렌딩     | 노출 보정 |
|------|------------|-----------|
| V1   | overwrite  | —         |
| V2   | feather    | —         |
| V3   | multiband  | —         |
| V4   | multiband  | gain      |
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.bev.blend import blend, feather_weights, last_writer_labels, winner_labels
from src.bev.gain import GainCompensator, apply_gains
from src.bev.ground_homography import BEVGrid
from src.bev.warp import Lut, build_lut, remap
from src.constants import CAMERAS
from src.data.nuscenes_lite import Frame

VARIANTS: dict[str, tuple[str, bool]] = {
    "V1": ("overwrite", False),
    "V2": ("feather", False),
    "V3": ("multiband", False),
    "V4": ("multiband", True),
}


@dataclass
class BEVResult:
    bev: np.ndarray
    warped: list[np.ndarray]
    masks: list[np.ndarray]
    weights: list[np.ndarray]
    labels: np.ndarray
    gains: np.ndarray | None
    seam_labels: np.ndarray            # 합성 결과에서 픽셀이 실제로 온 채널(이음새 지표용)
    timings: dict[str, float] = field(default_factory=dict)


def load_images(frame: Frame) -> dict[str, np.ndarray]:
    return {c: cv2.imread(str(frame.images[c]), cv2.IMREAD_COLOR) for c in CAMERAS}


def _calib_key(frame: Frame) -> tuple:
    return tuple((c, frame.cameras[c].K.tobytes(), frame.cameras[c].ego_from_cam.R.tobytes(),
                  frame.cameras[c].ego_from_cam.t.tobytes()) for c in CAMERAS)


class SurroundBEV:
    def __init__(self, grid: BEVGrid | None = None, variant: str = "V4",
                 time_comp: bool = False, max_range: float | None = None,
                 levels: int = 5, gain_alpha: float = 1.0):
        self.grid = grid or BEVGrid()
        self.variant = variant
        self.method, self.use_gain = VARIANTS[variant]
        self.time_comp = time_comp
        self.max_range = max_range
        self.levels = levels
        self.gain = GainCompensator(alpha=gain_alpha)
        self._cache: dict[tuple, tuple[dict[str, Lut], list[np.ndarray], np.ndarray]] = {}

    def luts(self, frame: Frame) -> tuple[dict[str, Lut], list[np.ndarray], np.ndarray]:
        """채널별 LUT, feather 가중치, 최대 가중 채널 라벨.

        시간 보정이 꺼져 있으면 캘리브레이션당 1회만 계산한다.
        """
        if not self.time_comp:
            key = _calib_key(frame)
            if key not in self._cache:
                self._cache[key] = self._build(frame)
            return self._cache[key]
        return self._build(frame)

    def _build(self, frame: Frame):
        luts = {}
        for c in CAMERAS:
            T = frame.cam_from_ref_ego(c) if self.time_comp else None
            luts[c] = build_lut(frame.cameras[c], self.grid, T, max_range=self.max_range)
        weights = feather_weights([luts[c].valid for c in CAMERAS])
        return luts, weights, winner_labels(weights)

    def compose(self, frame: Frame, images: dict[str, np.ndarray] | None = None) -> BEVResult:
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        images = images if images is not None else load_images(frame)
        t1 = time.perf_counter()
        luts, weights, labels = self.luts(frame)
        t2 = time.perf_counter()
        warped = [remap(images[c], luts[c]) for c in CAMERAS]
        masks = [luts[c].valid for c in CAMERAS]
        t3 = time.perf_counter()
        gains = None
        if self.use_gain:
            gains = self.gain(warped, masks)
            warped = apply_gains(warped, gains)
        t4 = time.perf_counter()
        bev = blend(warped, masks, weights, self.method, self.levels, labels)
        t5 = time.perf_counter()
        tm.update(load=t1 - t0, lut=t2 - t1, warp=t3 - t2, gain=t4 - t3, blend=t5 - t4,
                  total=t5 - t1)
        seam = last_writer_labels(masks) if self.method == "overwrite" else labels
        return BEVResult(bev, warped, masks, weights, labels, gains, seam,
                         {k: v * 1e3 for k, v in tm.items()})
