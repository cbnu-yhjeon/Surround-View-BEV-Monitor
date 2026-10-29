"""프로젝트 전역 상수.

채널 순서는 적응형 카메라 뷰 연구(Adaptive-Camera-View-Selection-System)의
`src/utils/constants.py`와 동일하게 시계 방향으로 고정한다. 두 프로젝트의 attention
뷰 인덱스가 같은 카메라를 가리키게 하기 위함이다.
"""

from __future__ import annotations

import os
from pathlib import Path

CAMERAS: list[str] = [
    "CAM_FRONT",        # 0
    "CAM_FRONT_RIGHT",  # 1
    "CAM_BACK_RIGHT",   # 2
    "CAM_BACK",         # 3
    "CAM_BACK_LEFT",    # 4
    "CAM_FRONT_LEFT",   # 5
]
NUM_CAMERAS: int = len(CAMERAS)
CAMERA_INDEX: dict[str, int] = {c: i for i, c in enumerate(CAMERAS)}

# 관제 UI 썸네일용 짧은 이름
CAMERA_SHORT: dict[str, str] = {
    "CAM_FRONT": "F", "CAM_FRONT_RIGHT": "FR", "CAM_BACK_RIGHT": "BR",
    "CAM_BACK": "B", "CAM_BACK_LEFT": "BL", "CAM_FRONT_LEFT": "FL",
}

DATAROOT: Path = Path(os.environ.get("NUSCENES_ROOT", Path.home() / "data" / "nuscenes"))
RESULTS: Path = Path(__file__).resolve().parents[1] / "results"

# nuScenes 수집 차량(Renault Zoe) 차체 — 자차 원점은 뒤 차축 중심(지면).
# 휠베이스 2.588 m, 전장 4.084 m, 전폭 1.730 m 에서 여유를 조금 둔 값.
EGO_X_MIN, EGO_X_MAX = -1.0, 3.6
EGO_Y_HALF = 1.0

# 영상 하단에 자차 차체(트렁크 범퍼)가 찍히는 채널 — 해당 행 수만큼 IPM에서 제외.
BODY_CROP_ROWS: dict[str, int] = {"CAM_BACK": 60}
