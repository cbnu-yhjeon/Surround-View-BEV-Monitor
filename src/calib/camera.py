"""nuScenes 캘리브레이션 → K, R, t (Lec 10: 내부·외부 파라미터, 좌표계 변환).

nuScenes `calibrated_sensor`는 **카메라 → 자차** 변환(센서 좌표를 자차 좌표로 옮기는
회전·이동)을 준다. 투영에는 그 역, 즉 **자차 → 카메라** 가 필요하다.

    X_ego = R_ce · X_cam + t_ce            (calibrated_sensor)
    X_cam = R_ceᵀ · (X_ego − t_ce)         (투영용: R = R_ceᵀ, t = −R_ceᵀ t_ce)

좌표계
  - 자차: x 전방, y 좌측, z 위. 원점은 뒤 차축 중심의 지면.
  - 카메라: x 오른쪽, y 아래, z 광축(전방).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    """nuScenes 쿼터니언 [w, x, y, z] → 3×3 회전 행렬."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass(frozen=True)
class Pose:
    """강체 변환 child → parent:  X_parent = R · X_child + t."""

    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def from_record(rec: dict) -> "Pose":
        return Pose(quat_to_rot(rec["rotation"]), np.asarray(rec["translation"], float))

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    def inverse(self) -> "Pose":
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def __matmul__(self, other: "Pose") -> "Pose":
        """(a @ b): b의 child → a의 parent."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """pts (..., 3) 변환."""
        return pts @ self.R.T + self.t

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T


@dataclass(frozen=True)
class CameraModel:
    """핀홀 카메라. `ego_from_cam`은 calibrated_sensor 그대로(카메라 → 자차)."""

    name: str
    K: np.ndarray
    ego_from_cam: Pose
    width: int = 1600
    height: int = 900

    @property
    def cam_from_ego(self) -> Pose:
        return self.ego_from_cam.inverse()

    @property
    def center_ego(self) -> np.ndarray:
        """자차 좌표계에서 카메라 광학 중심."""
        return self.ego_from_cam.t

    @property
    def axis_ego(self) -> np.ndarray:
        """자차 좌표계에서 광축 방향(단위 벡터)."""
        return self.ego_from_cam.R[:, 2]

    @property
    def yaw(self) -> float:
        a = self.axis_ego
        return float(np.arctan2(a[1], a[0]))

    @property
    def hfov(self) -> float:
        return float(2 * np.arctan(self.width / (2 * self.K[0, 0])))

    def project(self, pts_ego: np.ndarray, cam_from_ego: Pose | None = None):
        """자차 점 (N,3) → 픽셀 (N,2), 깊이 (N,).

        `cam_from_ego`를 주면 기본 외부 파라미터 대신 사용한다(시간 보정 포함 변환).
        """
        T = cam_from_ego or self.cam_from_ego
        pc = T.apply(np.atleast_2d(pts_ego))
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (pc @ self.K.T)[:, :2] / z[:, None]
        return uv, z

    def in_image(self, uv: np.ndarray, z: np.ndarray, min_depth: float = 0.5) -> np.ndarray:
        return ((z > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] <= self.width - 1)
                & (uv[:, 1] >= 0) & (uv[:, 1] <= self.height - 1))
