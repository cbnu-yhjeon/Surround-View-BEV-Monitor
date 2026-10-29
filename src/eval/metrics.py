"""이음새 품질 · 기하 정확도 지표 (README 지표 표).

이음새 품질
  - `overlap_photometric`: 겹침 영역에서 두 채널의 밝기 평균 절대 차(MAD). 노출
    보정 전후를 비교한다. 블렌딩은 이 값을 바꾸지 않는다(합성 전 지표).
  - `seam_step`: 합성 결과에서 이음새를 가로지르는 이웃 픽셀 밝기 차 / 이음새가 아닌
    이웃 픽셀 밝기 차. 1이면 이음새가 보이지 않음, 클수록 계단이 보인다.
  - `overlap_sharpness`: 겹침 영역에서 합성 결과의 |Laplacian| / 담당 채널 원본의
    |Laplacian|. 1보다 작으면 흐려지거나 이중상(ghost)으로 고주파가 깎였다.
  - `pair_misalignment`: 겹침 영역에서 두 채널 BEV 사이의 평행이동(위상 상관).
    차선 마킹 같은 지면 텍스처가 이음새에서 얼마나 어긋나는지(px, m).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.constants import CAMERAS


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def overlap_photometric(warped: list[np.ndarray], masks: list[np.ndarray],
                        step: int = 2) -> float:
    grays = [_gray(w)[::step, ::step] for w in warped]
    ms = [m[::step, ::step] for m in masks]
    tot, n = 0.0, 0
    for i in range(len(grays)):
        for j in range(i + 1, len(grays)):
            ov = ms[i] & ms[j]
            k = int(ov.sum())
            if k:
                tot += float(np.abs(grays[i][ov] - grays[j][ov]).sum())
                n += k
    return tot / n if n else float("nan")


def seam_step(bev: np.ndarray, labels: np.ndarray) -> float:
    g = _gray(bev)
    ratios_num, ratios_den = [], []
    for axis in (0, 1):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        la, lb = labels[tuple(a)], labels[tuple(b)]
        d = np.abs(g[tuple(a)] - g[tuple(b)])
        both = (la >= 0) & (lb >= 0)
        seam = both & (la != lb)
        ratios_num.append(d[seam])
        ratios_den.append(d[both & ~seam])
    num = np.concatenate(ratios_num)
    den = np.concatenate(ratios_den)
    if num.size == 0 or den.size == 0:
        return float("nan")
    return float(num.mean() / max(den.mean(), 1e-6))


def overlap_sharpness(bev: np.ndarray, warped: list[np.ndarray], masks: list[np.ndarray],
                      labels: np.ndarray) -> float:
    count = np.sum(np.stack(masks), 0)
    ov = (count >= 2) & (labels >= 0)
    if not ov.any():
        return float("nan")
    lap_bev = np.abs(cv2.Laplacian(_gray(bev), cv2.CV_32F))
    ref = np.zeros_like(lap_bev)
    for i, w in enumerate(warped):
        sel = ov & (labels == i)
        if sel.any():
            ref[sel] = np.abs(cv2.Laplacian(_gray(w), cv2.CV_32F))[sel]
    return float(lap_bev[ov].mean() / max(ref[ov].mean(), 1e-6))


def _highpass(g: np.ndarray) -> np.ndarray:
    return g - cv2.GaussianBlur(g, (0, 0), 3)


def pair_misalignment(warped: list[np.ndarray], masks: list[np.ndarray], res: float,
                      near: np.ndarray | None = None, min_pixels: int = 1500,
                      min_response: float = 0.1) -> list[dict]:
    """겹치는 채널 쌍마다 위상 상관으로 구한 평행이동.

    near: 평가할 영역 마스크(예: 자차 12 m 이내). 원거리는 IPM 해상도가 낮고
    높이 있는 물체의 시차가 커서 정렬 지표가 의미 없다. 위상 상관 응답이
    min_response 미만이면(텍스처 부족·반복 패턴) 신뢰할 수 없어 버린다.
    """
    out = []
    grays = [_gray(w) for w in warped]
    for i in range(len(warped)):
        for j in range(i + 1, len(warped)):
            ov = masks[i] & masks[j]
            if near is not None:
                ov &= near
            k = int(ov.sum())
            if k < min_pixels:
                continue
            ys, xs = np.nonzero(ov)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            win = ov[y0:y1, x0:x1].astype(np.float32)
            win = cv2.GaussianBlur(win, (0, 0), 4) * win   # 마스크 경계 누설 완화
            a = _highpass(grays[i][y0:y1, x0:x1]) * win
            b = _highpass(grays[j][y0:y1, x0:x1]) * win
            if a.std() < 1.0 or b.std() < 1.0:
                continue                                   # 텍스처 없음(밤, 빈 노면)
            (dx, dy), resp = cv2.phaseCorrelate(a, b)
            if resp < min_response:
                continue
            px = float(np.hypot(dx, dy))
            out.append({"pair": f"{CAMERAS[i]}|{CAMERAS[j]}", "pixels": k,
                        "dx": float(dx), "dy": float(dy), "shift_px": px,
                        "shift_m": px * res, "response": float(resp)})
    return out
