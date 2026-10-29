"""마일스톤 3: 블렌딩 V1–V4 비교 + 이음새 정렬(시간 보정 전후).

    python scripts/02_compare_blending.py --stride 2

출력 (results/blending/)
  per_frame.csv      프레임 × 변형별 지표
  alignment.csv      프레임 × 채널쌍 × 시간보정 여부별 어긋남
  summary.json/.md   주간/야간별 평균
  compare_*.jpg      변형 비교 그림
"""

from __future__ import annotations

import argparse
import csv
import time

import cv2
import numpy as np

import _common  # noqa: F401
from src.bev.ground_homography import BEVGrid
from src.bev.pipeline import VARIANTS, SurroundBEV, load_images
from src.data import NuScenesLite
from src.eval.metrics import overlap_photometric, overlap_sharpness, pair_misalignment, seam_step
from src.overlay.box_projection import EGO_CENTER_X, draw_ego


def near_mask(grid: BEVGrid, radius: float) -> np.ndarray:
    x, y = grid.ground_coords()
    return np.hypot(x - EGO_CENTER_X, y) <= radius


def figure(frames, ns, grid, d, tag):
    """변형별 전체 BEV + 앞-우 이음새 확대."""
    f = frames[len(frames) // 2]
    images = load_images(f)
    row, crops = [], []
    uv0 = grid.to_pixel(np.array([[12.0, 1.0]]))[0].astype(int)      # 좌상단 (전방, 좌측)
    uv1 = grid.to_pixel(np.array([[2.0, -9.0]]))[0].astype(int)
    for v in VARIANTS:
        bev = SurroundBEV(grid, variant=v).compose(f, images).bev
        crop = bev[uv0[1]:uv1[1], uv0[0]:uv1[0]].copy()
        full = bev.copy()
        draw_ego(full, grid)
        cv2.rectangle(full, tuple(uv0), tuple(uv1), (80, 220, 255), 3)
        full = cv2.resize(full, (480, 480))
        cv2.putText(full, v, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 220, 255), 3, cv2.LINE_AA)
        row.append(full)
        crops.append(cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST))
    sheet = np.vstack([np.hstack(row), np.hstack(crops)])
    cv2.imwrite(str(d / f"compare_{tag}_{f.scene}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--stride", type=int, default=2, help="keyframe 간격")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--near", type=float, default=20.0, help="정렬 지표 평가 반경(m)")
    args = ap.parse_args()

    ns = NuScenesLite(args.version)
    grid = BEVGrid()
    d = _common.out_dir("blending")
    scenes = args.scenes or sorted(ns.scenes)
    near = near_mask(grid, args.near)
    pipes = {v: SurroundBEV(grid, variant=v) for v in VARIANTS}
    align_pipes = {False: pipes["V1"], True: SurroundBEV(grid, variant="V1", time_comp=True)}

    rows, align_rows = [], []
    t_start = time.time()
    for scene in scenes:
        night = ns.is_night(scene)
        frames = ns.keyframes(scene)[:: args.stride]
        for k, f in enumerate(frames):
            images = load_images(f)
            for v, p in pipes.items():
                r = p.compose(f, images)
                rows.append({
                    "scene": scene, "night": int(night), "frame": k, "variant": v,
                    "photometric": overlap_photometric(r.warped, r.masks),
                    "seam_step": seam_step(r.bev, r.seam_labels),
                    "sharpness": overlap_sharpness(r.bev, r.warped, r.masks, r.labels),
                    "ms_total": r.timings["total"],
                })
            for tc, p in align_pipes.items():
                r = p.compose(f, images)
                for m in pair_misalignment(r.warped, r.masks, grid.res, near):
                    align_rows.append({"scene": scene, "night": int(night), "frame": k,
                                       "time_comp": int(tc), "speed": f.speed(),
                                       "sync_ms": max(abs(x) for x in f.sync_error_ms().values()),
                                       **m})
        print(f"{scene}: {len(frames)} frames ({time.time() - t_start:.0f}s)", flush=True)
        if scene in ("scene-0061", "scene-1077"):
            figure(frames, ns, grid, d, "night" if night else "day")

    with open(d / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if align_rows:
        with open(d / "alignment.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, align_rows[0].keys())
            w.writeheader()
            w.writerows(align_rows)

    summary = []
    for cond, sel in (("all", lambda r: True), ("day", lambda r: not r["night"]),
                      ("night", lambda r: r["night"])):
        for v in VARIANTS:
            rs = [r for r in rows if r["variant"] == v and sel(r)]
            if not rs:
                continue
            summary.append({"cond": cond, "variant": v, "n": len(rs),
                            **{m: float(np.nanmean([r[m] for r in rs]))
                               for m in ("photometric", "seam_step", "sharpness")},
                            "ms_total_median": float(np.median([r["ms_total"] for r in rs]))})
    align_summary = []
    for tc in (0, 1):
        rs = [r for r in align_rows if r["time_comp"] == tc]
        if rs:
            align_summary.append({
                "time_comp": tc, "n_pairs": len(rs),
                "shift_m_median": float(np.median([r["shift_m"] for r in rs])),
                "shift_m_mean": float(np.mean([r["shift_m"] for r in rs])),
                "shift_px_median": float(np.median([r["shift_px"] for r in rs])),
                "fast_shift_m_median": float(np.median([r["shift_m"] for r in rs if r["speed"] > 5]
                                                       or [np.nan])),
            })
    _common.save_json({"blending": summary, "alignment": align_summary,
                       "machine": _common.machine_note(), "stride": args.stride,
                       "version": args.version}, d / "summary.json")
    md = ["## 블렌딩 비교 (V1–V4)", "",
          _common.md_table(summary, ["cond", "variant", "n", "photometric", "seam_step",
                                     "sharpness", "ms_total_median"],
                           {"photometric": ".2f", "seam_step": ".3f", "sharpness": ".3f",
                            "ms_total_median": ".1f"}),
          "", "## 이음새 정렬 (위상 상관, 자차 반경 %.0f m)" % args.near, "",
          _common.md_table(align_summary, ["time_comp", "n_pairs", "shift_m_median",
                                           "shift_m_mean", "shift_px_median",
                                           "fast_shift_m_median"],
                           {"shift_m_median": ".3f", "shift_m_mean": ".3f",
                            "shift_px_median": ".1f", "fast_shift_m_median": ".3f"})]
    (d / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
