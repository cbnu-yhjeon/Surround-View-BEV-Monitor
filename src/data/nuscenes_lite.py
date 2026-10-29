"""nuScenes 최소 로더 — devkit 없이 JSON 테이블만 읽는다.

devkit(nuscenes-devkit)은 numpy<2를 고정해서 다른 프로젝트와 venv를 같이 쓰면
버전 충돌이 난다(형제 저장소 requirements.txt 경고 참조). 이 프로젝트가 쓰는 것은
캘리브레이션 · 자차 자세 · 3D 박스 · 영상 경로뿐이라 직접 읽는 편이 가볍다.

프레임 묶기 (README 리스크: "카메라마다 타임스탬프가 조금씩 다름")
  - keyframe 모드: sample 하나 = 6채널 keyframe (2 Hz). GT 박스가 정확히 있다.
  - sweep 모드: CAM_FRONT의 모든 sample_data(≈12 Hz)를 기준으로, 다른 채널은
    타임스탬프가 가장 가까운 것을 묶는다. 박스는 앞뒤 keyframe 사이를 보간한다.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.calib.camera import CameraModel, Pose, quat_to_rot, yaw_of
from src.constants import CAMERAS, DATAROOT

_TABLES = ["scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose",
           "sample_annotation", "instance", "category", "attribute", "log"]


@dataclass
class Box:
    """전역 좌표계 3D 박스. size = (w, l, h), nuScenes 관례."""

    center: np.ndarray
    size: np.ndarray
    R: np.ndarray
    category: str
    instance: str
    velocity: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    attribute: str = ""

    def transformed(self, T: Pose) -> "Box":
        """T: 현재 좌표계 → 새 좌표계 (예: ego_from_global)."""
        v = self.velocity
        if np.all(np.isfinite(v)):
            v = (T.R[:2, :2] @ v)
        return Box(T.apply(self.center), self.size, T.R @ self.R, self.category,
                   self.instance, v, self.attribute)

    @property
    def yaw(self) -> float:
        return yaw_of(self.R)

    def corners(self) -> np.ndarray:
        """(8,3): 앞 4개는 바닥(z−h/2), 뒤 4개는 윗면. 순서: +x+y, +x−y, −x−y, −x+y."""
        w, l, h = self.size
        x = np.array([1, 1, -1, -1]) * l / 2
        y = np.array([1, -1, -1, 1]) * w / 2
        bottom = np.stack([x, y, -np.full(4, h / 2)], 1)
        top = bottom + [0, 0, h]
        return np.concatenate([bottom, top]) @ self.R.T + self.center

    def bottom_center(self) -> np.ndarray:
        return self.center - self.R[:, 2] * self.size[2] / 2


@dataclass
class Frame:
    """6채널 한 묶음. 기준 시각/자세는 CAM_FRONT."""

    scene: str
    sample_token: str | None          # keyframe이면 sample, sweep이면 None
    timestamp: int                    # μs, CAM_FRONT 기준
    images: dict[str, Path]
    cam_timestamps: dict[str, int]
    cam_ego_poses: dict[str, Pose]    # global_from_ego (각 채널 촬영 시각)
    cameras: dict[str, CameraModel]
    ref_pose: Pose                    # global_from_ego (기준 시각)
    ego_velocity: np.ndarray          # 전역 xy, m/s
    lidar_path: Path | None = None
    lidar_ego_from_sensor: Pose | None = None
    lidar_ego_pose: Pose | None = None

    @property
    def ego_from_global(self) -> Pose:
        return self.ref_pose.inverse()

    def sync_error_ms(self) -> dict[str, float]:
        return {c: (t - self.timestamp) / 1e3 for c, t in self.cam_timestamps.items()}

    def cam_from_ref_ego(self, cam: str) -> Pose:
        """시간 보정 외부 파라미터: 기준 시각 자차 → (촬영 시각 자차) → 카메라."""
        ego_t_from_ref = self.cam_ego_poses[cam].inverse() @ self.ref_pose
        return self.cameras[cam].cam_from_ego @ ego_t_from_ref

    def speed(self) -> float:
        return float(np.linalg.norm(self.ego_velocity))


class NuScenesLite:
    def __init__(self, version: str = "v1.0-mini", dataroot: Path | str = DATAROOT):
        self.dataroot = Path(dataroot)
        self.version = version
        meta = self.dataroot / version
        t: dict[str, dict[str, dict]] = {}
        for name in _TABLES:
            with open(meta / f"{name}.json") as f:
                t[name] = {r["token"]: r for r in json.load(f)}
        self.tables = t
        self.scenes = {s["name"]: s for s in t["scene"].values()}

        self._sensor_of_calib = {c["token"]: t["sensor"][c["sensor_token"]]["channel"]
                                 for c in t["calibrated_sensor"].values()}
        # sample → {channel: keyframe sample_data}
        self._sample_sd: dict[str, dict[str, dict]] = {}
        for sd in t["sample_data"].values():
            sd["channel"] = self._sensor_of_calib[sd["calibrated_sensor_token"]]
            if sd["is_key_frame"]:
                self._sample_sd.setdefault(sd["sample_token"], {})[sd["channel"]] = sd
        self._sample_anns: dict[str, list[dict]] = {}
        for a in t["sample_annotation"].values():
            self._sample_anns.setdefault(a["sample_token"], []).append(a)

    # ------------------------------------------------------------------ 기본 조회
    def scene_samples(self, scene: str) -> list[dict]:
        s = self.scenes[scene]
        out, tok = [], s["first_sample_token"]
        while tok:
            rec = self.tables["sample"][tok]
            out.append(rec)
            tok = rec["next"]
        return out

    def camera_model(self, sd: dict) -> CameraModel:
        cs = self.tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        return CameraModel(sd["channel"], np.asarray(cs["camera_intrinsic"], float),
                           Pose(quat_to_rot(cs["rotation"]), np.asarray(cs["translation"], float)),
                           sd["width"] or 1600, sd["height"] or 900)

    def ego_pose(self, sd: dict) -> Pose:
        return Pose.from_record(self.tables["ego_pose"][sd["ego_pose_token"]])

    def is_night(self, scene: str) -> bool:
        return "night" in self.scenes[scene]["description"].lower()

    def _chain(self, first_sd: dict) -> list[dict]:
        """한 채널의 scene 전체 sample_data(keyframe+sweep), 시간순."""
        sd = first_sd
        while sd["prev"]:
            sd = self.tables["sample_data"][sd["prev"]]
        out = []
        while sd:
            out.append(sd)
            sd = self.tables["sample_data"][sd["next"]] if sd["next"] else None
        return out

    # ------------------------------------------------------------------ 프레임
    def _ego_velocity(self, chain: list[dict], i: int) -> np.ndarray:
        """자차 속도(전역 xy) — 앞뒤 sample_data 자세 차분."""
        a, b = chain[max(i - 1, 0)], chain[min(i + 1, len(chain) - 1)]
        dt = (b["timestamp"] - a["timestamp"]) / 1e6
        if dt <= 0:
            return np.zeros(2)
        return (self.ego_pose(b).t[:2] - self.ego_pose(a).t[:2]) / dt

    def _make_frame(self, scene: str, sample_token: str | None, sds: dict[str, dict],
                    vel: np.ndarray, lidar_sd: dict | None) -> Frame:
        front = sds["CAM_FRONT"]
        f = Frame(
            scene=scene, sample_token=sample_token, timestamp=front["timestamp"],
            images={c: self.dataroot / sds[c]["filename"] for c in CAMERAS},
            cam_timestamps={c: sds[c]["timestamp"] for c in CAMERAS},
            cam_ego_poses={c: self.ego_pose(sds[c]) for c in CAMERAS},
            cameras={c: self.camera_model(sds[c]) for c in CAMERAS},
            ref_pose=self.ego_pose(front), ego_velocity=vel,
        )
        if lidar_sd is not None:
            cs = self.tables["calibrated_sensor"][lidar_sd["calibrated_sensor_token"]]
            f.lidar_path = self.dataroot / lidar_sd["filename"]
            f.lidar_ego_from_sensor = Pose.from_record(cs)
            f.lidar_ego_pose = self.ego_pose(lidar_sd)
        return f

    def keyframes(self, scene: str) -> list[Frame]:
        samples = self.scene_samples(scene)
        front_chain = self._chain(self._sample_sd[samples[0]["token"]]["CAM_FRONT"])
        idx = {sd["token"]: i for i, sd in enumerate(front_chain)}
        frames = []
        for s in samples:
            sds = self._sample_sd[s["token"]]
            vel = self._ego_velocity(front_chain, idx[sds["CAM_FRONT"]["token"]])
            frames.append(self._make_frame(scene, s["token"], sds, vel, sds.get("LIDAR_TOP")))
        return frames

    def sweeps(self, scene: str) -> list[Frame]:
        """CAM_FRONT 전 프레임 기준, 다른 채널은 가장 가까운 타임스탬프로 묶음."""
        first = self._sample_sd[self.scenes[scene]["first_sample_token"]]
        chains = {c: self._chain(first[c]) for c in CAMERAS}
        stamps = {c: [sd["timestamp"] for sd in chains[c]] for c in CAMERAS}
        frames = []
        for i, front in enumerate(chains["CAM_FRONT"]):
            sds = {"CAM_FRONT": front}
            for c in CAMERAS[1:]:
                j = bisect.bisect_left(stamps[c], front["timestamp"])
                cand = [k for k in (j - 1, j) if 0 <= k < len(stamps[c])]
                sds[c] = chains[c][min(cand, key=lambda k: abs(stamps[c][k] - front["timestamp"]))]
            tok = front["sample_token"] if front["is_key_frame"] else None
            vel = self._ego_velocity(chains["CAM_FRONT"], i)
            frames.append(self._make_frame(scene, tok, sds, vel, None))
        return frames

    # ------------------------------------------------------------------ 박스
    def _ann_velocity(self, ann: dict, max_dt: float = 1.5) -> np.ndarray:
        """devkit box_velocity와 같은 방식: 앞뒤 annotation 위치 차분."""
        sa = self.tables["sample_annotation"]
        prev = sa[ann["prev"]] if ann["prev"] else ann
        nxt = sa[ann["next"]] if ann["next"] else ann
        if prev is nxt:
            return np.full(2, np.nan)
        ts = self.tables["sample"]
        dt = (ts[nxt["sample_token"]]["timestamp"] - ts[prev["sample_token"]]["timestamp"]) / 1e6
        if dt <= 0:
            return np.full(2, np.nan)
        if dt > max_dt * (2 if (ann["prev"] and ann["next"]) else 1):
            return np.full(2, np.nan)
        return (np.asarray(nxt["translation"][:2]) - np.asarray(prev["translation"][:2])) / dt

    def _box(self, ann: dict) -> Box:
        inst = self.tables["instance"][ann["instance_token"]]
        attrs = [self.tables["attribute"][a]["name"] for a in ann["attribute_tokens"]]
        return Box(np.asarray(ann["translation"], float), np.asarray(ann["size"], float),
                   quat_to_rot(ann["rotation"]),
                   self.tables["category"][inst["category_token"]]["name"],
                   ann["instance_token"], self._ann_velocity(ann), attrs[0] if attrs else "")

    def boxes(self, frame: Frame) -> list[Box]:
        """전역 좌표계 GT 박스. sweep 프레임은 앞뒤 keyframe 사이 선형 보간."""
        if frame.sample_token is not None:
            return [self._box(a) for a in self._sample_anns.get(frame.sample_token, [])]
        samples = self.scene_samples(frame.scene)
        times = [s["timestamp"] for s in samples]
        j = bisect.bisect_right(times, frame.timestamp)
        if j == 0 or j >= len(samples):
            s = samples[min(max(j - 1, 0), len(samples) - 1)]
            return [self._box(a) for a in self._sample_anns.get(s["token"], [])]
        s0, s1 = samples[j - 1], samples[j]
        a = (frame.timestamp - s0["timestamp"]) / (s1["timestamp"] - s0["timestamp"])
        b0 = {b.instance: b for b in (self._box(x) for x in self._sample_anns.get(s0["token"], []))}
        b1 = {b.instance: b for b in (self._box(x) for x in self._sample_anns.get(s1["token"], []))}
        out = []
        for inst, x0 in b0.items():
            x1 = b1.get(inst)
            if x1 is None:
                if a < 0.5:
                    out.append(x0)
                continue
            dyaw = (x1.yaw - x0.yaw + np.pi) % (2 * np.pi) - np.pi
            yaw = x0.yaw + a * dyaw
            c, s_ = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            out.append(Box((1 - a) * x0.center + a * x1.center, x0.size, R, x0.category,
                           inst, x0.velocity, x0.attribute))
        out += [x1 for inst, x1 in b1.items() if inst not in b0 and a >= 0.5]
        return out

    def lidar_points_ego(self, frame: Frame) -> np.ndarray | None:
        """LIDAR_TOP 점(N,3)을 **기준 시각 자차 좌표계**로."""
        if frame.lidar_path is None:
            return None
        pts = np.fromfile(frame.lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(float)
        T = frame.ego_from_global @ frame.lidar_ego_pose @ frame.lidar_ego_from_sensor
        return T.apply(pts)
