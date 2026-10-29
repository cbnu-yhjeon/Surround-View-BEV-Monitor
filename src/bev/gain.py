"""Gain compensation (Lec 9; Brown & Lowe, IJCV 2007 §6).

채널마다 이득 g_i 하나를 두고, 겹치는 영역의 평균 밝기가 같아지도록 최소제곱으로 푼다.

    e = ½ Σ_i Σ_j N_ij [ (g_i·Ī_ij − g_j·Ī_ji)² / σ_N²  +  (1 − g_i)² / σ_g² ]

  N_ij : i, j 겹침 픽셀 수,  Ī_ij : 겹침 영역에서 i 영상의 평균 밝기.
  두 번째 항이 없으면 g = 0 이 자명해이므로, 이득이 1 근처에 머물도록 붙잡아 둔다.
∂e/∂g_i = 0 이 g에 대한 선형 방정식이라 6×6 연립방정식 하나로 끝난다. 색 채널별로 푼다.
"""

from __future__ import annotations

import cv2
import numpy as np


def overlap_stats(images: list[np.ndarray], masks: list[np.ndarray], step: int = 4):
    """N (n,n), Ī (n,n,C). step 간격으로 표본추출해 속도를 아낀다.

    마스크 행렬 M (n,P) 와 영상 X (n,P,C) 로 쓰면 N = M·Mᵀ, Σ_ij = (X_c ⊙ M)·Mᵀ.
    """
    M = np.stack([m[::step, ::step].ravel() for m in masks]).astype(np.float32)
    X = np.stack([im[::step, ::step].reshape(M.shape[1], -1) for im in images]).astype(np.float32)
    N = M @ M.T
    np.fill_diagonal(N, 0)
    S = np.einsum("ipc,jp->ijc", X * M[..., None], M)
    I = S / np.maximum(N, 1)[..., None]
    return N.astype(np.float64), I.astype(np.float64)


def solve_gains(N: np.ndarray, I: np.ndarray, sigma_n: float = 10.0,
                sigma_g: float = 0.1) -> np.ndarray:
    """(n, C) 이득."""
    n, _, C = I.shape
    g = np.ones((n, C))
    for c in range(C):
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j or N[i, j] == 0:
                    continue
                A[i, i] += N[i, j] * (I[i, j, c] ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
                A[i, j] -= N[i, j] * I[i, j, c] * I[j, i, c] / sigma_n ** 2
                b[i] += N[i, j] / sigma_g ** 2
        has = np.diag(A) > 0
        if has.any():
            g[has, c] = np.linalg.solve(A[np.ix_(has, has)], b[has])
    return g


def apply_gains(images: list[np.ndarray], gains: np.ndarray) -> list[np.ndarray]:
    return [cv2.multiply(im, tuple(float(x) for x in g) + (0.0,)) for im, g in zip(images, gains)]


class GainCompensator:
    """프레임 간 이득 깜빡임을 막기 위한 지수 평활(alpha=1 이면 평활 없음)."""

    def __init__(self, alpha: float = 1.0, sigma_n: float = 10.0, sigma_g: float = 0.1):
        self.alpha, self.sigma_n, self.sigma_g = alpha, sigma_n, sigma_g
        self.gains: np.ndarray | None = None

    def __call__(self, images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
        N, I = overlap_stats(images, masks)
        g = solve_gains(N, I, self.sigma_n, self.sigma_g)
        if self.gains is None or self.alpha >= 1.0:
            self.gains = g
        else:
            self.gains = self.alpha * g + (1 - self.alpha) * self.gains
        return self.gains
