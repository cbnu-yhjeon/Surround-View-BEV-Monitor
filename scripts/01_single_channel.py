"""마일스톤 1: 채널별 단일 BEV + 6채널 덮어쓰기 합성 + 원통 파노라마.

    python scripts/01_single_channel.py --scene scene-0061 --index 0
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import _common  # noqa: F401
from src.bev.ground_homography import BEVGrid
from src.bev.pipeline import SurroundBEV, load_images
from src.bev.warp import build_lut, remap
from src.constants import CAMERAS
from src.data import NuScenesLite
from src.overlay.box_projection import draw_ego, draw_range_rings
from src.panorama.cylindrical import CylindricalPanorama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    ns = NuScenesLite()
    frame = ns.keyframes(args.scene)[args.index]
    images = load_images(frame)
    grid = BEVGrid()
    d = _common.out_dir("single_channel")

    tiles = []
    for c in CAMERAS:
        lut = build_lut(frame.cameras[c], grid)
        bev = remap(images[c], lut)
        draw_range_rings(bev, grid)
        draw_ego(bev, grid)
        cv2.putText(bev, c, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(d / f"{c}.jpg"), bev)
        src = cv2.resize(images[c], (800, 450))
        tiles.append(np.vstack([src, bev]))
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(d / "contact_sheet.jpg"), cv2.resize(sheet, None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    res = SurroundBEV(grid, variant="V1").compose(frame, images)
    bev = res.bev.copy()
    draw_range_rings(bev, grid)
    draw_ego(bev, grid)
    cv2.imwrite(str(d / "six_channel_V1.jpg"), bev)

    for method, gain in (("overwrite", False), ("multiband", True)):
        pano = CylindricalPanorama(method=method, gain=gain).compose(frame, images)
        cv2.imwrite(str(d / f"panorama_{method}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved →", d)


if __name__ == "__main__":
    main()
