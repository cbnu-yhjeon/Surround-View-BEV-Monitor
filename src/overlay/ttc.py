"""TTC(Time-To-Collision) — 적응형 카메라 뷰 연구의 정의를 그대로 따른다.

원본: Adaptive-Camera-View-Selection-System `src/labeling/ttc.py` (cpa 정의).

    p_rel = p_obj − p_ego,  v_rel = v_obj − v_ego
    t*    = −(p_rel·v_rel) / ‖v_rel‖²        (최접근 시각)
    DCA   = ‖p_rel + v_rel·t*‖               (그때의 거리)
    TTC   = t*  (접근 중일 때)  |  ∞ (멀어지는 중)

고전적 radial TTC는 옆으로 스쳐 지나갈 객체(길가 주차 차량)도 충돌 임박으로
판정한다. cpa는 최접근 거리 DCA를 따로 주므로, DCA가 차체 폭보다 크면 "스쳐 지나감"
으로 낮춰 표시할 수 있다.
"""

from __future__ import annotations

import numpy as np

TTC_INF = float("inf")

# 관제 화면 색상 단계 (BGR). 경계값(초)은 형제 연구의 τ=3 s 주변에서 고른 표시용 값.
TTC_LEVELS: list[tuple[float, str, tuple[int, int, int]]] = [
    (1.5, "critical", (40, 40, 235)),
    (3.0, "warning", (0, 140, 255)),
    (5.0, "caution", (0, 220, 255)),
]
SAFE = ("safe", (90, 200, 90))
PASSING = ("passing", (200, 200, 120))


def closest_approach(p_rel: np.ndarray, v_rel: np.ndarray) -> tuple[float, float]:
    """(t*, DCA). 멀어지는 중이거나 상대정지면 (0, 현재 거리)."""
    r = float(np.linalg.norm(p_rel))
    vv = float(np.dot(v_rel, v_rel))
    if vv < 1e-9:
        return 0.0, r
    t = -float(np.dot(p_rel, v_rel)) / vv
    if t <= 0.0:
        return 0.0, r
    return t, float(np.linalg.norm(p_rel + v_rel * t))


def compute_ttc(p_obj, p_ego, v_obj, v_ego, v_r_min: float = 0.1) -> tuple[float, float]:
    """(TTC, DCA). 수평면(x, y) 기준. 접근 중이 아니면 TTC = inf."""
    p_rel = np.asarray(p_obj, float)[:2] - np.asarray(p_ego, float)[:2]
    v_rel = np.asarray(v_obj, float)[:2] - np.asarray(v_ego, float)[:2]
    r = float(np.linalg.norm(p_rel))
    if r < 1e-6:
        return 0.0, 0.0
    v_r = -float(np.dot(p_rel, v_rel)) / r
    t, dca = closest_approach(p_rel, v_rel)
    if v_r <= v_r_min:
        return TTC_INF, dca
    return t, dca


def risk_level(ttc: float, dca: float, dca_hit: float) -> tuple[str, tuple[int, int, int]]:
    """TTC 단계. 최접근 거리가 dca_hit 보다 크면 스쳐 지나가는 객체로 본다."""
    if not np.isfinite(ttc):
        return SAFE
    if dca > dca_hit:
        return PASSING
    for th, name, color in TTC_LEVELS:
        if ttc < th:
            return name, color
    return SAFE
