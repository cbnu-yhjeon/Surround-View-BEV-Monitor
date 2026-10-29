import numpy as np

from src.bev.blend import blend, feather_weights, winner_labels
from src.bev.gain import apply_gains, overlap_stats, solve_gains
from src.eval.metrics import overlap_photometric, seam_step
from src.overlay.ttc import compute_ttc, risk_level


def _two_views(h=128, w=192):
    rng = np.random.default_rng(0)
    base = (rng.random((h, w, 3)) * 120 + 60).astype(np.uint8)
    m1 = np.zeros((h, w), bool)
    m2 = np.zeros((h, w), bool)
    m1[:, : w * 2 // 3] = True
    m2[:, w // 3:] = True
    return base, [m1, m2]


def test_feather_and_multiband_reproduce_identical_inputs():
    base, masks = _two_views()
    ims = [np.where(m[..., None], base, 0).astype(np.uint8) for m in masks]
    w = feather_weights(masks)
    for method in ("feather", "multiband"):
        out = blend(ims, masks, w, method, levels=4)
        diff = np.abs(out.astype(int) - base.astype(int))
        assert diff.mean() < 1.5, method


def test_winner_labels_partition():
    _, masks = _two_views()
    lab = winner_labels(feather_weights(masks))
    assert set(np.unique(lab)) == {0, 1}
    assert lab[0, 0] == 0 and lab[0, -1] == 1


def test_gain_compensation_equalizes_exposure():
    base, masks = _two_views()
    ims = [np.where(masks[0][..., None], base * 1.0, 0), np.where(masks[1][..., None], base * 0.7, 0)]
    ims = [np.clip(i, 0, 255).astype(np.uint8) for i in ims]
    N, I = overlap_stats(ims, masks, step=1)
    g = solve_gains(N, I)
    # σ_g 사전항이 이득을 1 근처로 붙잡으므로 30% 차이는 일부만 메워진다 (Brown & Lowe)
    gap0 = abs(I[0, 1, 0] - I[1, 0, 0])
    gap1 = abs(g[0, 0] * I[0, 1, 0] - g[1, 0] * I[1, 0, 0])
    assert gap1 < 0.4 * gap0
    assert g[0, 0] < 1.0 < g[1, 0]
    # 사전항을 약하게 하면 거의 완전히 맞춰진다
    g_free = solve_gains(N, I, sigma_g=10.0)
    assert abs(g_free[0, 0] * I[0, 1, 0] - g_free[1, 0] * I[1, 0, 0]) < 1.0
    before = overlap_photometric(ims, masks)
    after = overlap_photometric(apply_gains(ims, g), masks)
    assert after < 0.5 * before


def test_seam_step_detects_hard_seam():
    h, w = 64, 64
    flat = np.full((h, w, 3), 100, np.uint8)
    flat += (np.random.default_rng(0).random((h, w, 1)) * 4).astype(np.uint8)
    lab = np.zeros((h, w), np.int32)
    lab[:, w // 2:] = 1
    stepped = flat.copy()
    stepped[:, w // 2:] += 40
    assert seam_step(stepped, lab) > 5 * seam_step(flat, lab)


def test_ttc_head_on_and_passing():
    ttc, dca = compute_ttc([20, 0], [0, 0], [-5, 0], [5, 0])          # 정면, 상대 10 m/s
    assert abs(ttc - 2.0) < 1e-9 and dca < 1e-9
    ttc, dca = compute_ttc([20, 6], [0, 0], [0, 0], [10, 0])          # 6 m 옆 주차 차량
    assert abs(ttc - 2.0) < 1e-9 and abs(dca - 6.0) < 1e-9
    assert risk_level(ttc, dca, dca_hit=2.4)[0] == "passing"
    assert risk_level(1.0, 0.5, dca_hit=2.4)[0] == "critical"
    assert np.isinf(compute_ttc([20, 0], [0, 0], [10, 0], [5, 0])[0])  # 멀어짐
