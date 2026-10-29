import numpy as np
import pytest

from src.bev.ground_homography import BEVGrid, bev_homography, ground_homography, image_to_ground
from src.bev.warp import build_lut
from src.calib.camera import CameraModel, Pose, quat_to_rot


def _front_camera(pitch_deg: float = 5.0, name: str = "CAM_FRONT") -> CameraModel:
    """자차 x 전방을 보는 카메라. 카메라 축: x 오른쪽, y 아래, z 전방."""
    base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # cam → ego (pitch 0)
    p = np.radians(pitch_deg)                                      # 아래로 숙임
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    K = np.array([[1266.0, 0, 816], [0, 1266.0, 491], [0, 0, 1]])
    return CameraModel(name, K, Pose(Ry @ base, np.array([1.7, 0.0, 1.5])))


def test_quat_identity_and_rotation():
    assert np.allclose(quat_to_rot([1, 0, 0, 0]), np.eye(3))
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]               # z축 90°
    assert np.allclose(quat_to_rot(q) @ [1, 0, 0], [0, 1, 0])


def test_pose_inverse_compose():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    T = Pose(quat_to_rot(q), rng.normal(size=3))
    pts = rng.normal(size=(10, 3))
    assert np.allclose((T.inverse() @ T).apply(pts), pts)


def test_ground_homography_matches_projection():
    cam = _front_camera()
    pts = np.array([[8.0, 1.0, 0.0], [15.0, -3.0, 0.0], [6.0, 0.5, 0.0]])
    uv, _ = cam.project(pts)
    H = ground_homography(cam)
    p = np.column_stack([pts[:, :2], np.ones(3)]) @ H.T
    assert np.allclose(p[:, :2] / p[:, 2:], uv, atol=1e-6)
    assert np.allclose(image_to_ground(cam, uv), pts[:, :2], atol=1e-6)


def test_bev_grid_roundtrip():
    g = BEVGrid()
    xy = np.array([[0.0, 0.0], [10.0, -5.0], [-3.2, 7.7]])
    uv = g.to_pixel(xy)
    assert np.allclose(g.to_ground(uv), xy)
    p = np.column_stack([uv, np.ones(3)]) @ g.S().T
    assert np.allclose(p[:, :2], xy)
    # 위쪽이 전방, 오른쪽이 차량 우측
    assert g.to_pixel(np.array([[10.0, 0.0]]))[0, 1] < g.to_pixel(np.array([[0.0, 0.0]]))[0, 1]
    assert g.to_pixel(np.array([[0.0, -5.0]]))[0, 0] > g.to_pixel(np.array([[0.0, 0.0]]))[0, 0]


def test_lut_agrees_with_projection():
    cam = _front_camera()
    g = BEVGrid(res=0.1)
    lut = build_lut(cam, g)
    assert lut.valid.any()
    vs, us = np.nonzero(lut.valid)
    idx = np.random.default_rng(1).choice(len(us), 50)
    xy = g.to_ground(np.column_stack([us[idx], vs[idx]]).astype(float))
    uv, z = cam.project(np.column_stack([xy, np.zeros(len(xy))]))
    assert np.all(z > 0)
    assert np.allclose(np.column_stack([lut.map_x[vs[idx], us[idx]], lut.map_y[vs[idx], us[idx]]]),
                       uv, atol=2e-2)
    # 카메라 뒤(자차 뒤쪽 지면)는 유효하지 않다
    assert not lut.valid[g.to_pixel(np.array([[-10.0, 0.0]])).astype(int)[0, ::-1][0],
                         g.to_pixel(np.array([[-10.0, 0.0]])).astype(int)[0, 0]]


def test_time_compensated_homography_shifts_ground():
    """기준 시각 이후 자차가 1 m 전진했으면 같은 지면점이 1 m 가깝게 보여야 한다."""
    cam = _front_camera()
    moved = Pose(np.eye(3), np.array([1.0, 0, 0]))         # ego_t → ego_ref
    T = cam.cam_from_ego @ moved.inverse()                  # ego_ref → ego_t → cam
    p_ref = np.array([[10.0, 0.0, 0.0]])
    uv_comp, _ = cam.project(p_ref, T)
    uv_near, _ = cam.project(np.array([[9.0, 0.0, 0.0]]))
    assert np.allclose(uv_comp, uv_near)
    HS = bev_homography(cam, BEVGrid(), T)
    assert np.isfinite(HS).all()


@pytest.mark.parametrize("pitch", [0.0, 10.0])
def test_hfov(pitch):
    cam = _front_camera(pitch)
    assert 60 < np.degrees(cam.hfov) < 70
