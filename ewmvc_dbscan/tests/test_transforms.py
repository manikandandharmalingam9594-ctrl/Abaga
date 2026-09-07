"""
Unit tests for src/transforms.py module.
"""

import pytest
import numpy as np
import pandas as pd

from src.transforms import rotate_points, sensor_to_vehicle, vehicle_to_global, add_global_coordinates


def test_rotate_points_180_deg():
    points = np.array([[1.0, 2.0], [-3.0, 4.0]])
    angle_rad = np.radians(180.0)

    rotated = rotate_points(points, angle_rad)
    expected = np.array([[-1.0, -2.0], [3.0, -4.0]])

    np.testing.assert_allclose(rotated, expected, atol=1e-6)


def test_sensor_to_vehicle():
    rel_x = np.array([1.0, -2.0])
    rel_y = np.array([3.0, -4.0])

    vx, vy = sensor_to_vehicle(rel_x, rel_y, rotation_deg=180.0)

    np.testing.assert_allclose(vx, [-1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(vy, [-3.0, 4.0], atol=1e-6)


def test_vehicle_to_global_yaw_zero():
    # yaw = 0 deg -> global_x = vehicle_x + x_veh, global_y = vehicle_y + y_veh
    veh_pts = np.array([[5.0, 2.0]])
    gx, gy = vehicle_to_global(veh_pts, vehicle_x=10.0, vehicle_y=20.0, vehicle_yaw=0.0)

    np.testing.assert_allclose(gx, [15.0], atol=1e-6)
    np.testing.assert_allclose(gy, [22.0], atol=1e-6)


def test_vehicle_to_global_yaw_ninety_deg():
    # yaw = pi/2 rad -> cos = 0, sin = 1
    # global_x = vehicle_x - y_veh, global_y = vehicle_y + x_veh
    veh_pts = np.array([[5.0, 2.0]])
    gx, gy = vehicle_to_global(veh_pts, vehicle_x=10.0, vehicle_y=20.0, vehicle_yaw=np.pi / 2)

    np.testing.assert_allclose(gx, [8.0], atol=1e-6)
    np.testing.assert_allclose(gy, [25.0], atol=1e-6)


def test_add_global_coordinates():
    df = pd.DataFrame({
        "rel_x_sensor": [1.0],
        "rel_y_sensor": [2.0],
        "vehicle_x": [10.0],
        "vehicle_y": [20.0],
        "vehicle_yaw": [0.0],
    })

    augmented = add_global_coordinates(df, rotation_deg=180.0)

    assert "rel_x_vehicle" in augmented.columns
    assert "rel_y_vehicle" in augmented.columns
    assert "global_x" in augmented.columns
    assert "global_y" in augmented.columns

    np.testing.assert_allclose(augmented["rel_x_vehicle"].values, [-1.0], atol=1e-6)
    np.testing.assert_allclose(augmented["rel_y_vehicle"].values, [-2.0], atol=1e-6)
    np.testing.assert_allclose(augmented["global_x"].values, [9.0], atol=1e-6)
    np.testing.assert_allclose(augmented["global_y"].values, [18.0], atol=1e-6)
