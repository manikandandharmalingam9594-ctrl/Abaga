"""
Unit tests for src/uncertainty.py module.
"""

import pytest
import numpy as np
import pandas as pd

from src.uncertainty import compute_observation_covariance, add_uncertainty_propagation


def test_compute_observation_covariance_yaw_zero():
    # yaw = 0 rad, x_v = 2.0, y_v = 1.0
    # a = -y_v = -1.0, b = x_v = 2.0
    # cov_xx = var_px + a^2 * var_pyaw + var_sx = 0.05^2 + (-1)^2 * 0.01^2 + 0.1^2
    #         = 0.0025 + 0.0001 + 0.01 = 0.0126
    # cov_yy = var_py + b^2 * var_pyaw + var_sy = 0.05^2 + (2)^2 * 0.01^2 + 0.1^2
    #         = 0.0025 + 0.0004 + 0.01 = 0.0129
    # cov_xy = a * b * var_pyaw = (-1) * (2) * 0.0001 = -0.0002
    cov_xx, cov_xy, cov_yy = compute_observation_covariance(
        rel_x_vehicle=2.0,
        rel_y_vehicle=1.0,
        vehicle_yaw=0.0,
        sigma_sensor_x=0.10,
        sigma_sensor_y=0.10,
        sigma_pose_x=0.05,
        sigma_pose_y=0.05,
        sigma_pose_yaw=0.01,
    )

    np.testing.assert_allclose(cov_xx, 0.0126, atol=1e-6)
    np.testing.assert_allclose(cov_yy, 0.0129, atol=1e-6)
    np.testing.assert_allclose(cov_xy, -0.0002, atol=1e-6)


def test_compute_observation_covariance_yaw_ninety():
    # yaw = pi/2 rad, x_v = 2.0, y_v = 1.0
    # cos = 0, sin = 1 => a = -x_v = -2.0, b = -y_v = -1.0
    # cov_xx = var_px + (-2)^2 * var_pyaw + var_sy
    # cov_yy = var_py + (-1)^2 * var_pyaw + var_sx
    # cov_xy = (-2)*(-1)*var_pyaw + 0 = 2 * var_pyaw
    cov_xx, cov_xy, cov_yy = compute_observation_covariance(
        rel_x_vehicle=2.0,
        rel_y_vehicle=1.0,
        vehicle_yaw=np.pi / 2,
        sigma_sensor_x=0.10,
        sigma_sensor_y=0.10,
        sigma_pose_x=0.05,
        sigma_pose_y=0.05,
        sigma_pose_yaw=0.01,
    )

    np.testing.assert_allclose(cov_xx, 0.0025 + 4 * 0.0001 + 0.01, atol=1e-6)
    np.testing.assert_allclose(cov_yy, 0.0025 + 1 * 0.0001 + 0.01, atol=1e-6)
    np.testing.assert_allclose(cov_xy, 2 * 0.0001, atol=1e-6)


def test_positive_diagonal_and_symmetry():
    cov_xx, cov_xy, cov_yy = compute_observation_covariance(
        rel_x_vehicle=np.array([1.0, -5.0, 10.0]),
        rel_y_vehicle=np.array([2.0, 3.0, -4.0]),
        vehicle_yaw=np.array([0.1, -1.2, 2.5]),
        sigma_sensor_x=0.15,
        sigma_sensor_y=0.08,
        sigma_pose_x=0.04,
        sigma_pose_y=0.04,
        sigma_pose_yaw=0.02,
    )

    assert np.all(cov_xx > 0), "cov_xx must be strictly positive"
    assert np.all(cov_yy > 0), "cov_yy must be strictly positive"
    # Cauchy-Schwarz inequality for covariance: cov_xy^2 < cov_xx * cov_yy
    assert np.all((cov_xy ** 2) <= (cov_xx * cov_yy)), "Covariance matrix must be positive semi-definite"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        compute_observation_covariance(
            rel_x_vehicle=1.0, rel_y_vehicle=1.0, vehicle_yaw=0.0, sigma_sensor_x=-0.1
        )


def test_add_uncertainty_propagation_dataframe():
    df = pd.DataFrame({
        "rel_x_vehicle": [1.0, 2.0],
        "rel_y_vehicle": [3.0, 4.0],
        "vehicle_yaw": [0.0, 0.5],
    })

    augmented = add_uncertainty_propagation(df)

    assert "cov_xx" in augmented.columns
    assert "cov_xy" in augmented.columns
    assert "cov_yy" in augmented.columns
    assert "sigma_x" in augmented.columns
    assert "sigma_y" in augmented.columns
    assert len(augmented) == 2
    assert not augmented[["cov_xx", "cov_xy", "cov_yy"]].isna().any().any()
