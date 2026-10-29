"""속도: 룩업 테이블(cv2.remap) 사전 계산 전후 + 단계별 합성 시간 (README 지표).

비교 (6채널 워핑만, 영상 디코딩 제외)
  per_frame_coords  매 프레임 LUT 좌표를 새로 계산 + remap   (시간 보정 켤 때의 비용)
  warpPerspective   매 프레임 호모그래피 워핑 (LUT 없음)
  remap_float       사전 계산 float32 LUT
  remap_fixed       사전 계산 고정소수점 LUT (convertMaps CV_16SC2)
  remap_fixed_half  위 + 입력 영상 ½ 축소 (LUT도 ½ 스케일)
그리고 V1–V4 전체 합성 시간, 해상도(5 cm / 10 cm)별.

    python scripts/04_benchmark_speed.py --repeat 20
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import _common  # noqa: F401
from src.bev.ground_homography import BEVGrid
from src.bev.pipeline import VARIANTS, SurroundBEV, load_images
from src.bev.warp import build_lut, remap, warp_perspective
from src.calib.camera import CameraModel
from src.constants import CAMERAS
from src.data import NuScenesLite


def timeit(fn, repeat: int) -> dict:
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": float(np.median(ts)), "p10_ms": float(np.percentile(ts, 10)),
            "p90_ms": float(np.percentile(ts, 90))}


def half_camera(cam: CameraModel) -> CameraModel:
    K = cam.K.copy()
    K[:2] *= 0.5
    return CameraModel(cam.name, K, cam.ego_from_cam, cam.width // 2, cam.height // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="cv2.setNumThreads")
    args = ap.parse_args()
    if args.threads is not None:
        cv2.setNumThreads(args.threads)

    ns = NuScenesLite()
    f = ns.keyframes("scene-0103")[10]
    images = load_images(f)
    half = {c: cv2.resize(images[c], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            for c in CAMERAS}
    d = _common.out_dir("speed")
    results = {"machine": _common.machine_note(), "cv2_threads": cv2.getNumThreads(),
               "warp": {}, "compose": {}}

    for res in (0.05, 0.10):
        grid = BEVGrid(res=res)
        tag = f"{grid.width}x{grid.height} ({res * 100:.0f} cm/px)"
        luts = {c: build_lut(f.cameras[c], grid) for c in CAMERAS}
        luts_h = {c: build_lut(half_camera(f.cameras[c]), grid) for c in CAMERAS}
        for lt in list(luts.values()) + list(luts_h.values()):
            lt.fixed()

        def per_frame():
            for c in CAMERAS:
                remap(images[c], build_lut(f.cameras[c], grid, f.cam_from_ref_ego(c)), fixed=False)

        def wp():
            for c in CAMERAS:
                warp_perspective(images[c], f.cameras[c], grid)

        def rf():
            for c in CAMERAS:
                remap(images[c], luts[c], fixed=False)

        def rx():
            for c in CAMERAS:
                remap(images[c], luts[c], fixed=True)

        def rxh():
            for c in CAMERAS:
                remap(half[c], luts_h[c], fixed=True)

        results["warp"][tag] = {name: timeit(fn, args.repeat) for name, fn in
                                [("per_frame_coords", per_frame), ("warpPerspective", wp),
                                 ("remap_float", rf), ("remap_fixed", rx),
                                 ("remap_fixed_half", rxh)]}
        results["compose"][tag] = {}
        for v in VARIANTS:
            for tc in (False, True):
                p = SurroundBEV(grid, variant=v, time_comp=tc)
                key = v + (" +timecomp" if tc else "")
                results["compose"][tag][key] = timeit(lambda: p.compose(f, images), args.repeat)
        print(tag, "done", flush=True)

    _common.save_json(results, d / "summary.json")
    lines = ["## 속도 (6채널, JPEG 디코딩 제외, 중앙값 ms)", "",
             f"측정 환경: {results['machine']}, cv2 threads={results['cv2_threads']}", ""]
    for tag, r in results["warp"].items():
        lines += [f"### 워핑 — {tag}", "",
                  _common.md_table([{"method": k, **v} for k, v in r.items()],
                                   ["method", "median_ms", "p10_ms", "p90_ms"],
                                   {"median_ms": ".1f", "p10_ms": ".1f", "p90_ms": ".1f"}), ""]
    for tag, r in results["compose"].items():
        lines += [f"### 전체 합성 — {tag}", "",
                  _common.md_table([{"variant": k, **v} for k, v in r.items()],
                                   ["variant", "median_ms", "p10_ms", "p90_ms"],
                                   {"median_ms": ".1f", "p10_ms": ".1f", "p90_ms": ".1f"}), ""]
    (d / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
