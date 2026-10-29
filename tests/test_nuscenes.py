"""v1.0-mini 가 있을 때만 도는 통합 테스트."""

import numpy as np
import pytest

from src.constants import CAMERAS, DATAROOT

pytestmark = pytest.mark.skipif(not (DATAROOT / "v1.0-mini").exists(), reason="nuScenes mini 없음")


@pytest.fixture(scope="module")
def ns():
    from src.data import NuScenesLite
    return NuScenesLite("v1.0-mini")


def test_keyframes_have_six_cameras(ns):
    frames = ns.keyframes("scene-0061")
    assert len(frames) == 39
    f = frames[0]
    assert set(f.images) == set(CAMERAS)
    assert all(p.exists() for p in f.images.values())
    assert max(abs(v) for v in f.sync_error_ms().values()) < 60


def test_camera_yaws_are_clockwise(ns):
    f = ns.keyframes("scene-0061")[0]
    yaws = [np.degrees(f.cameras[c].yaw) for c in CAMERAS]
    assert abs(yaws[0]) < 5 and yaws[1] < -40 and abs(abs(yaws[3]) - 180) < 5 and yaws[5] > 40


def test_gt_box_projects_into_some_camera(ns):
    from src.overlay.box_projection import assess
    f = ns.keyframes("scene-0061")[5]
    risks = assess(f, ns.boxes(f))
    assert risks
    assert any(r.camera is not None for r in risks)


def test_sweep_frames_interpolate_boxes(ns):
    sw = ns.sweeps("scene-0061")
    assert len(sw) > 3 * len(ns.keyframes("scene-0061"))
    f = next(x for x in sw if x.sample_token is None)
    assert ns.boxes(f)


def test_bev_compose_all_variants(ns):
    from src.bev.ground_homography import BEVGrid
    from src.bev.pipeline import VARIANTS, SurroundBEV, load_images
    f = ns.keyframes("scene-0061")[0]
    ims = load_images(f)
    for v in VARIANTS:
        r = SurroundBEV(BEVGrid(res=0.1), variant=v).compose(f, ims)
        assert r.bev.shape == (400, 400, 3)
        assert r.bev.mean() > 20
