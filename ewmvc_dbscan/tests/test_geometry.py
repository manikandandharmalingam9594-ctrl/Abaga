"""
Unit tests for src/geometry.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.geometry import (
    compute_spatial_dispersion,
    calculate_geometry_score,
    compute_geometry_features,
)


def test_1_identical_points():
    """TEST 1: All observations at exactly the same point yield sigma_spatial = 0 and geometry_score = 1."""
    gx = np.array([5.0, 5.0, 5.0, 5.0])
    gy = np.array([-2.0, -2.0, -2.0, -2.0])

    mu_x, mu_y, sigma_sp, r_med, r_p95, r_max = compute_spatial_dispersion(gx, gy)
    score = calculate_geometry_score(sigma_sp, sigma0_m=0.75)

    assert mu_x == pytest.approx(5.0, abs=1e-8)
    assert mu_y == pytest.approx(-2.0, abs=1e-8)
    assert sigma_sp == pytest.approx(0.0, abs=1e-8)
    assert r_max == pytest.approx(0.0, abs=1e-8)
    assert score == pytest.approx(1.0, abs=1e-8)


def test_2_known_symmetric_points():
    """TEST 2: Known symmetric points around center verify RMS dispersion mathematically."""
    # 4 points at (1,0), (-1,0), (0,1), (0,-1) => centroid = (0,0)
    # r_i = 1.0 for all points => mean(r_i^2) = 1.0 => sigma_spatial = 1.0
    gx = np.array([1.0, -1.0, 0.0, 0.0])
    gy = np.array([0.0, 0.0, 1.0, -1.0])

    mu_x, mu_y, sigma_sp, r_med, r_p95, r_max = compute_spatial_dispersion(gx, gy)

    assert mu_x == pytest.approx(0.0, abs=1e-8)
    assert mu_y == pytest.approx(0.0, abs=1e-8)
    assert sigma_sp == pytest.approx(1.0, abs=1e-8)
    assert r_med == pytest.approx(1.0, abs=1e-8)
    assert r_max == pytest.approx(1.0, abs=1e-8)


def test_3_sigma_spatial_equals_sigma0():
    """TEST 3: sigma_spatial = sigma0 yields geometry_score approx exp(-0.5) = 0.60653066."""
    sigma_sp = 0.75
    sigma0 = 0.75

    score = calculate_geometry_score(sigma_sp, sigma0_m=sigma0)
    expected = np.exp(-0.5)  # 0.6065306597126334

    assert score == pytest.approx(expected, abs=1e-8)
    assert score == pytest.approx(0.60653066, abs=1e-6)


def test_4_large_spatial_dispersion():
    """TEST 4: Large spatial dispersion yields low but positive geometry_score in (0, 1]."""
    sigma_sp = 5.0
    sigma0 = 0.75

    score = calculate_geometry_score(sigma_sp, sigma0_m=sigma0)

    assert score > 0.0
    assert score < 0.01
    assert not np.isnan(score)


def test_5_singleton_component():
    """TEST 5: Singleton component yields sigma_spatial = 0 and geometry_score = 1."""
    df_singleton = pd.DataFrame({
        "candidate_id": [0],
        "global_x": [12.34],
        "global_y": [-56.78],
    })

    res_df = compute_geometry_features(df_singleton, sigma0_m=0.75)

    assert len(res_df) == 1
    assert res_df.iloc[0]["sigma_spatial_m"] == pytest.approx(0.0, abs=1e-8)
    assert res_df.iloc[0]["geometry_score"] == pytest.approx(1.0, abs=1e-8)


def test_6_actual_dataset():
    """TEST 6: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_csv = os.path.join(base_dir, "outputs", "ewmvc_candidate_components.csv")

    if not os.path.exists(cand_csv):
        pytest.skip(f"Input candidate CSV '{cand_csv}' does not exist.")

    cand_df = pd.read_csv(cand_csv)

    # Run without label column
    cand_no_labels = cand_df.drop(columns=["label"], errors="ignore")
    res_df = compute_geometry_features(cand_no_labels, sigma0_m=0.75)

    assert len(res_df) == 141, "Expected exactly 141 candidate components"
    assert res_df["candidate_id"].nunique() == 141

    # Check bounds (0, 1] and zero NaNs
    assert not res_df.isna().any().any()
    assert (res_df["geometry_score"] > 0.0).all()
    assert (res_df["geometry_score"] <= 1.0).all()

    # Singletons check
    singletons = res_df[res_df["num_observations"] == 1]
    assert (singletons["sigma_spatial_m"] == 0.0).all()
    assert (singletons["geometry_score"] == 1.0).all()

    # Label independence
    if "label" in cand_df.columns:
        res_with_labels = compute_geometry_features(cand_df, sigma0_m=0.75)
        pd.testing.assert_frame_equal(res_df, res_with_labels)
