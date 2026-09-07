"""
Unit tests for src/landmark.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.landmark import (
    fuse_observations_inverse_covariance,
    compute_landmarks,
)


def test_1_singleton_observation():
    """TEST 1: Singleton observation yields landmark position and covariance equal to observation."""
    gx = [10.5]
    gy = [-4.2]
    c_xx = [0.08]
    c_xy = [0.01]
    c_yy = [0.12]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    assert lx == pytest.approx(10.5, abs=1e-8)
    assert ly == pytest.approx(-4.2, abs=1e-8)
    assert lcov_xx == pytest.approx(0.08, abs=1e-8)
    assert lcov_xy == pytest.approx(0.01, abs=1e-8)
    assert lcov_yy == pytest.approx(0.12, abs=1e-8)
    assert lsig_x == pytest.approx(np.sqrt(0.08), abs=1e-8)
    assert lsig_y == pytest.approx(np.sqrt(0.12), abs=1e-8)
    assert w_shift == pytest.approx(0.0, abs=1e-8)


def test_2_two_observations_identical_covariance():
    """TEST 2: Two observations with identical covariance yield landmark position equal to arithmetic mean."""
    gx = [2.0, 4.0]
    gy = [1.0, 5.0]
    c_xx = [0.1, 0.1]
    c_xy = [0.0, 0.0]
    c_yy = [0.1, 0.1]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    assert lx == pytest.approx(3.0, abs=1e-8)
    assert ly == pytest.approx(3.0, abs=1e-8)
    # Total precision W = diag(10, 10) + diag(10, 10) = diag(20, 20) => covariance = diag(0.05, 0.05)
    assert lcov_xx == pytest.approx(0.05, abs=1e-8)
    assert lcov_yy == pytest.approx(0.05, abs=1e-8)
    assert w_shift == pytest.approx(0.0, abs=1e-8)


def test_3_two_observations_different_covariance():
    """TEST 3: Lower-uncertainty observation receives greater influence."""
    # p1 = [0, 0], var1 = 0.1 (weight = 10)
    # p2 = [1, 0], var2 = 0.4 (weight = 2.5)
    # fused_x = (10*0 + 2.5*1) / (10 + 2.5) = 2.5 / 12.5 = 0.2
    gx = [0.0, 1.0]
    gy = [0.0, 0.0]
    c_xx = [0.1, 0.4]
    c_xy = [0.0, 0.0]
    c_yy = [0.1, 0.4]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    assert lx == pytest.approx(0.2, abs=1e-8)
    assert ly == pytest.approx(0.0, abs=1e-8)
    # Fused variance = 1 / (10 + 2.5) = 1 / 12.5 = 0.08
    assert lcov_xx == pytest.approx(0.08, abs=1e-8)
    assert lcov_yy == pytest.approx(0.08, abs=1e-8)
    # Shift vs ordinary mean (0.5, 0) = |0.2 - 0.5| = 0.3
    assert w_shift == pytest.approx(0.3, abs=1e-8)


def test_4_known_analytical_example():
    """TEST 4: Known 2D analytical example with non-diagonal covariance."""
    # p1 = [10, 20], Sigma1 = [[2, 1], [1, 2]] => det = 3, inv = [[2/3, -1/3], [-1/3, 2/3]]
    # p2 = [12, 22], Sigma2 = [[2, 1], [1, 2]]
    # Total W = 2 * [[2/3, -1/3], [-1/3, 2/3]] = [[4/3, -2/3], [-2/3, 4/3]]
    # Position should be midpoint [11, 21]
    gx = [10.0, 12.0]
    gy = [20.0, 22.0]
    c_xx = [2.0, 2.0]
    c_xy = [1.0, 1.0]
    c_yy = [2.0, 2.0]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    assert lx == pytest.approx(11.0, abs=1e-8)
    assert ly == pytest.approx(21.0, abs=1e-8)
    assert lcov_xx == pytest.approx(1.0, abs=1e-8)
    assert lcov_xy == pytest.approx(0.5, abs=1e-8)
    assert lcov_yy == pytest.approx(1.0, abs=1e-8)


def test_5_covariance_symmetry():
    """TEST 5: Covariance symmetry: lcov_xy is symmetric."""
    gx = [1.0, 2.0, 3.0]
    gy = [4.0, 5.0, 6.0]
    c_xx = [0.15, 0.12, 0.09]
    c_xy = [0.03, -0.02, 0.01]
    c_yy = [0.18, 0.14, 0.11]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    cov_matrix = np.array([[lcov_xx, lcov_xy], [lcov_xy, lcov_yy]])
    assert np.allclose(cov_matrix, cov_matrix.T)


def test_6_invalid_covariance_handling():
    """TEST 6: Singular / non-positive definite covariance is handled safely without producing NaN."""
    gx = [0.0, 1.0]
    gy = [0.0, 0.0]
    c_xx = [0.0, 0.1]
    c_xy = [0.0, 0.0]
    c_yy = [0.0, 0.1]

    lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
        gx, gy, c_xx, c_xy, c_yy
    )

    assert not np.isnan(lx) and not np.isnan(ly)
    assert not np.isnan(lcov_xx) and not np.isnan(lcov_yy)


def test_7_actual_dataset():
    """TEST 7: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")

    f_obs = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    f_con = os.path.join(out_dir, "ewmvc_consensus.csv")

    if not os.path.exists(f_obs) or not os.path.exists(f_con):
        pytest.skip("Input CSV files missing.")

    obs_df = pd.read_csv(f_obs)
    con_df = pd.read_csv(f_con)

    # Run without label column
    obs_no_labels = obs_df.drop(columns=["label"], errors="ignore")
    landmarks_df = compute_landmarks(obs_no_labels, con_df)

    accepted_count = len(con_df[con_df["accepted"] == True])
    assert len(landmarks_df) == accepted_count, f"Expected {accepted_count} landmarks, got {len(landmarks_df)}"

    assert landmarks_df["candidate_id"].nunique() == len(landmarks_df)
    assert not landmarks_df[["landmark_x", "landmark_y", "landmark_cov_xx", "landmark_cov_yy"]].isna().any().any()

    assert (landmarks_df["landmark_sigma_x"] >= 0.0).all()
    assert (landmarks_df["landmark_sigma_y"] >= 0.0).all()
    assert "consensus_score" in landmarks_df.columns

    # Label independence check
    if "label" in obs_df.columns:
        landmarks_with_labels = compute_landmarks(obs_df, con_df)
        pd.testing.assert_frame_equal(landmarks_df, landmarks_with_labels)
