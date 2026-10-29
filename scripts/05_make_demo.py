"""마일스톤 5: 관제 UI 데모 영상.

CAM_FRONT 전 프레임(≈12 Hz)을 기준으로 6채널을 묶고(sweep 모드), keyframe 사이 박스는
보간한다. 장면마다 프로세스 하나씩 병렬로 렌더링한다(이득 평활은 장면 안에서 순차).

    python scripts/05_make_demo.py --scenes scene-0103 scene-0553 scene-1077
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2

import _common  # noqa: F401

DEFAULT_SCENES = ["scene-0103", "scene-0553", "scene-0916", "scene-1077"]


def render_scene(scene: str, version: str, step: int, max_frames: int | None,
                 width: int, variant: str) -> str:
    cv2.setNumThreads(2)
    from src.data import NuScenesLite
    from src.ui.monitor import MonitorRenderer

    ns = NuScenesLite(version)
    frames = ns.sweeps(scene)[::step]
    if max_frames:
        frames = frames[:max_frames]
    d = _common.out_dir("demo")
    r = MonitorRenderer(ns, variant=variant)
    fps = 12.0 / step
    height = int(round(width * 9 / 16))
    vw = cv2.VideoWriter(str(d / f"{scene}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, height))
    snap_at = {len(frames) // 3, 2 * len(frames) // 3}
    for i, f in enumerate(frames):
        img = r.render(f)
        if i in snap_at:
            cv2.imwrite(str(d / f"{scene}_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vw.write(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
    vw.release()
    return f"{scene}: {len(frames)} frames @ {fps:.1f} fps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--step", type=int, default=1, help="sweep 프레임 간격 (1 = 12 Hz)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--variant", default="V4")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(render_scene, s, args.version, args.step, args.max_frames,
                          args.width, args.variant) for s in args.scenes]
        for fu in futs:
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main()
