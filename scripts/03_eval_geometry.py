"""기하 정확도: GT 3D 박스 바닥 중심의 BEV 위치 오차(m) (README 지표).

절차 (keyframe마다, 위험 범주 GT 박스마다)
  1. 접지 높이 z_g: GT 박스 바닥 높이(주석자가 LiDAR 점에 맞춘 박스라 객체가 실제로
     닿는 지면). 교차 검증용으로 박스 주변 LiDAR 하위 10% 높이 z_lidar 도 기록한다
     — 원거리에서는 지면 점이 드물어 주변 물체에 오염되므로 기준으로 쓰지 않는다.
  2. 실제 접지점 P = (x_b, y_b, z_g) 를, 그 박스가 가장 잘 보이는 채널의 **촬영 시각**
     자세로 투영 → 영상에 실제로 찍힌 접지점 픽셀 p.
     (객체 이동도 반영: p_obj + v·(t_cam − t_sample))
  3. p 를 IPM(지면 z=0 가정)으로 BEV에 되돌린 위치와 기준 시각(CAM_FRONT)의 참 위치를 비교.
     - static   : 캘리브레이션만 사용 (시간 보정 없음)
     - timecomp : 채널 촬영 시각 ↔ 기준 시각 자차 이동 보정
     - oracle_z : timecomp + 참 지면 높이 z_g 사용 → 평지 가정 오차를 분리
오차 원인을 나눠 보고한다: 거리, 자차 속도(고속 구간), 지면 높이 |z_g|(경사·턱).

    python scripts/03_eval_geometry.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import _common  # noqa: F401
from src.bev.ground_homography import image_to_ground
from src.data import NuScenesLite
from src.overlay.box_projection import EGO_CENTER_X, best_camera, is_risk_category

MODES = ("static", "timecomp", "oracle_z")


def ground_height(pts: np.ndarray | None, box, fallback: float) -> tuple[float, bool]:
    if pts is None:
        return fallback, False
    c = box.center
    r = 0.5 * float(np.hypot(box.size[0], box.size[1]))
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    # 박스 좌표계에서 박스 밖인 점만 (차량 자체 점 제외)
    local = (pts[:, :2] - c[:2]) @ box.R[:2, :2]
    inside = (np.abs(local[:, 0]) <= box.size[1] / 2 + 0.2) & (np.abs(local[:, 1]) <= box.size[0] / 2 + 0.2)
    sel = (d > 0) & (d < r + 1.5) & ~inside & (pts[:, 2] < fallback + 0.6)
    if sel.sum() < 8:
        return fallback, False
    return float(np.percentile(pts[sel, 2], 10)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--max-dist", type=float, default=30.0)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    ns = NuScenesLite(args.version)
    d = _common.out_dir("geometry")
    rows = []
    for scene in args.scenes or sorted(ns.scenes):
        for f in ns.keyframes(scene):
            pts = ns.lidar_points_ego(f)
            if pts is not None:
                pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < args.max_dist + 5]
            sample_ts = ns.tables["sample"][f.sample_token]["timestamp"]
            T_ref = f.ego_from_global
            for b in ns.boxes(f):
                if not is_risk_category(b.category):
                    continue
                v = b.velocity if np.all(np.isfinite(b.velocity)) else np.zeros(2)
                # 기준 시각(CAM_FRONT)의 참 위치
                p_ref_g = b.center + np.r_[v, 0] * (f.timestamp - sample_ts) / 1e6
                be = b.transformed(T_ref)
                be.center = T_ref.apply(p_ref_g)
                dist = float(np.hypot(be.center[0] - EGO_CENTER_X, be.center[1]))
                if dist > args.max_dist or dist < 2.5:
                    continue
                zg = float(be.bottom_center()[2])
                z_lidar, from_lidar = ground_height(pts, be, zg)
                truth = be.center[:2]
                cam_name = best_camera(f, np.r_[truth, zg])
                if cam_name is None:
                    continue
                cam = f.cameras[cam_name]
                # 촬영 시각의 객체 위치 → 촬영 시각 자차 → 카메라
                t_c = f.cam_timestamps[cam_name]
                p_c_g = b.center + np.r_[v, 0] * (t_c - sample_ts) / 1e6
                p_c_g[2] = 0.0
                P_ref = T_ref.apply(p_c_g)
                P_ref[2] = zg                                    # 접지점 높이는 LiDAR 지면
                cam_from_ref = f.cam_from_ref_ego(cam_name)
                uv, z = cam.project(P_ref[None], cam_from_ref)
                if not cam.in_image(uv, z)[0]:
                    continue
                est = {
                    "static": image_to_ground(cam, uv)[0],
                    "timecomp": image_to_ground(cam, uv, cam_from_ref)[0],
                    "oracle_z": image_to_ground(cam, uv, cam_from_ref, z0=zg)[0],
                }
                row = {"scene": scene, "night": int(ns.is_night(scene)), "camera": cam_name,
                       "category": b.category, "dist": dist, "speed": f.speed(),
                       "dt_ms": (t_c - f.timestamp) / 1e3, "z_ground": zg,
                       "z_lidar": z_lidar, "z_from_lidar": int(from_lidar)}
                for m, e in est.items():
                    row[f"err_{m}"] = float(np.linalg.norm(e - truth))
                rows.append(row)
        print(scene, len(rows), flush=True)

    with open(d / "per_box.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    def stats(sel, label):
        rs = [r for r in rows if sel(r)]
        out = {"group": label, "n": len(rs)}
        for m in MODES:
            e = np.array([r[f"err_{m}"] for r in rs]) if rs else np.array([np.nan])
            out[f"{m}_median"] = float(np.median(e))
            out[f"{m}_p90"] = float(np.percentile(e, 90))
        return out

    groups = [stats(lambda r: True, "all"),
              stats(lambda r: r["dist"] < 10, "dist <10m"),
              stats(lambda r: 10 <= r["dist"] < 20, "dist 10–20m"),
              stats(lambda r: r["dist"] >= 20, "dist ≥20m"),
              stats(lambda r: r["speed"] < 1, "ego stopped (<1 m/s)"),
              stats(lambda r: r["speed"] >= 8, "ego fast (≥8 m/s)"),
              stats(lambda r: abs(r["z_ground"]) <= 0.15, "flat (z_g within ±0.15 m)"),
              stats(lambda r: abs(r["z_ground"]) > 0.15, "non-flat (z_g beyond ±0.15 m)"),
              stats(lambda r: abs(r["z_ground"]) <= 0.15 and r["speed"] >= 8, "flat & fast"),
              stats(lambda r: abs(r["z_ground"]) <= 0.15 and r["dist"] < 10, "flat & <10m")]
    lid = [r for r in rows if r["z_from_lidar"] and r["dist"] < 15]
    agree = float(np.median([abs(r["z_lidar"] - r["z_ground"]) for r in lid])) if lid else float("nan")
    _common.save_json({"groups": groups, "n_lidar_ground": len(lid),
                       "lidar_vs_box_bottom_median_abs_m_within15m": agree,
                       "z_ground_percentiles_5_50_95": np.percentile(
                           [r["z_ground"] for r in rows], [5, 50, 95]).tolist(),
                       "version": args.version}, d / "summary.json")
    cols = ["group", "n"] + [f"{m}_{s}" for m in MODES for s in ("median", "p90")]
    md = "## 기하 정확도: GT 박스 접지점 BEV 위치 오차 (m)\n\n" + _common.md_table(
        groups, cols, {c: ".3f" for c in cols[2:]}) + "\n"
    md += f"\nLiDAR 지면 vs 박스 바닥 (15 m 이내) 중앙 절대차: {agree:.3f} m\n"
    (d / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
