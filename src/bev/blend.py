"""겹침 영역 블렌딩 (Lec 9: Feathering · 다중 대역 블렌딩).

실험 V1–V3 (README 실험 설계):
  V1 overwrite   — 채널 순서대로 덮어쓰기. 이음새에서 밝기·기하가 계단처럼 끊긴다.
  V2 feather     — 마스크 경계까지 거리(distance transform)를 가중치로 정규화 합.
                   이음새는 부드러워지지만 어긋난 고주파(차선)는 이중상(ghost)이 된다.
  V3 multiband   — Burt & Adelson 1983. 라플라시안 피라미드의 대역마다 다른 폭으로
                   섞는다: 저주파(밝기)는 넓게, 고주파(경계)는 좁게.
                   Brown & Lowe 2007 처럼 가중치가 가장 큰 채널만 1인 이진 마스크를
                   쓰므로 고주파는 한 채널에서만 온다 → ghost 없음.
"""

from __future__ import annotations

import cv2
import numpy as np

METHODS = ("overwrite", "feather", "multiband")


def feather_weights(masks: list[np.ndarray], power: float = 1.0) -> list[np.ndarray]:
    """마스크 경계까지 거리 가중치 (정규화 전, float32)."""
    out = []
    for m in masks:
        d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        out.append(d ** power if power != 1.0 else d)
    return out


def winner_labels(weights: list[np.ndarray]) -> np.ndarray:
    """픽셀마다 가중치 최대 채널 인덱스, 아무도 없으면 −1."""
    best = np.zeros_like(weights[0])
    lab = np.full(weights[0].shape, -1, np.int32)
    for i, w in enumerate(weights):
        better = w > best
        lab[better] = i
        np.maximum(best, w, out=best)
    return lab


def last_writer_labels(masks: list[np.ndarray]) -> np.ndarray:
    """덮어쓰기에서 픽셀마다 최종적으로 남는 채널."""
    lab = np.full(masks[0].shape, -1, np.int32)
    for i, m in enumerate(masks):
        lab[m] = i
    return lab


def blend_overwrite(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(images[0])
    for img, m in zip(images, masks):
        out[m] = img[m]
    return out


def blend_feather(images: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(images[0].shape, np.float32)
    wsum = np.zeros(images[0].shape[:2], np.float32)
    for img, w in zip(images, weights):
        acc += img.astype(np.float32) * w[..., None]
        wsum += w
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 다중 대역
def _fill_invalid(img: np.ndarray, valid: np.ndarray, levels: int) -> np.ndarray:
    """유효 영역 밖을 주변 값으로 채운다(pull-push).

    워핑 영상 밖은 검정이라 그대로 피라미드를 만들면 경계의 검정이 저주파 대역을
    타고 이웃 채널 영역까지 번진다.
    """
    v = valid.astype(np.float32)
    im = img.astype(np.float32)
    im[~valid] = 0
    stack = [(im, v)]
    for _ in range(levels):
        im, v = cv2.pyrDown(im), cv2.pyrDown(v)
        stack.append((im, v))
    im, v = stack[-1]
    filled = im / np.maximum(v, 1e-6)[..., None]
    for im_k, v_k in reversed(stack[:-1]):
        up = cv2.pyrUp(filled, dstsize=(im_k.shape[1], im_k.shape[0]))
        a = np.minimum(v_k, 1.0)[..., None]
        own = im_k / np.maximum(v_k, 1e-6)[..., None]
        filled = up + a * (own - up)
    return filled


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    g = [img]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    lap = []
    for k in range(levels):
        up = cv2.pyrUp(g[k + 1], dstsize=(g[k].shape[1], g[k].shape[0]))
        lap.append(cv2.subtract(g[k], up))
    lap.append(g[-1])
    return lap


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    g = [img]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    return g


def _roi(sel: np.ndarray, margin: int, align: int) -> tuple[int, int, int, int]:
    """sel 을 감싸는 영역 + 여유, 피라미드가 어긋나지 않게 align 배수로 정렬."""
    ys, xs = np.nonzero(sel)
    H, W = sel.shape
    y0 = max((ys.min() - margin) // align * align, 0)
    x0 = max((xs.min() - margin) // align * align, 0)
    y1 = min(-(-(ys.max() + 1 + margin) // align) * align, H)
    x1 = min(-(-(xs.max() + 1 + margin) // align) * align, W)
    return y0, y1, x0, x1


def blend_multiband(images: list[np.ndarray], masks: list[np.ndarray],
                    weights: list[np.ndarray], levels: int = 5,
                    labels: np.ndarray | None = None) -> np.ndarray:
    """채널마다 담당 영역(ROI)만 피라미드로 분해해 전역 피라미드에 누적한다.

    ROI 시작점과 크기를 2^levels 배수로 맞추면 ROI의 k단계 피라미드가 전역 피라미드의
    (y0/2^k, x0/2^k) 위치에 정확히 겹친다. 캔버스도 2^levels 배수로 패딩한다.
    """
    if labels is None:
        labels = winner_labels(weights)
    H0, W0 = labels.shape
    align = 2 ** levels
    H, W = -(-H0 // align) * align, -(-W0 // align) * align
    pad = ((0, H - H0), (0, W - W0))
    labels = np.pad(labels, pad, constant_values=-1)
    acc = [np.zeros((H >> k, W >> k, 3), np.float32) for k in range(levels + 1)]
    wsum = [np.zeros((H >> k, W >> k), np.float32) for k in range(levels + 1)]
    margin = 3 * align
    for i, (img, m) in enumerate(zip(images, masks)):
        sel = labels == i
        if not sel.any():
            continue
        y0, y1, x0, x1 = _roi(sel, margin, align)
        im = np.pad(img, pad + ((0, 0),))[y0:y1, x0:x1]
        mv = np.pad(m, pad)[y0:y1, x0:x1]
        lap = _laplacian_pyramid(_fill_invalid(im, mv, levels), levels)
        gm = _gaussian_pyramid(sel[y0:y1, x0:x1].astype(np.float32), levels)
        for k in range(levels + 1):
            ys, xs = slice(y0 >> k, (y0 >> k) + gm[k].shape[0]), slice(x0 >> k, (x0 >> k) + gm[k].shape[1])
            acc[k][ys, xs] += lap[k] * gm[k][..., None]
            wsum[k][ys, xs] += gm[k]
    bands = [a / np.maximum(w, 1e-6)[..., None] for a, w in zip(acc, wsum)]
    out = bands[-1]
    for k in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(bands[k].shape[1], bands[k].shape[0])) + bands[k]
    out = out[:H0, :W0]
    out[labels[:H0, :W0] < 0] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def blend(images: list[np.ndarray], masks: list[np.ndarray], weights: list[np.ndarray],
          method: str, levels: int = 5, labels: np.ndarray | None = None) -> np.ndarray:
    if method == "overwrite":
        return blend_overwrite(images, masks)
    if method == "feather":
        return blend_feather(images, weights)
    if method == "multiband":
        return blend_multiband(images, masks, weights, levels, labels)
    raise ValueError(f"unknown blend method: {method}")
