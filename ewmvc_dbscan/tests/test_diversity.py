"""
Unit tests for src/diversity.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.diversity import (
    calculate_diversity_score,
    compute_pairwise_baselines,
    compute_diversity_features,
)


def test_1_singleton_candidate():
    """TEST 1: Singleton candidate (one observation at one vehicle position) yields max_baseline_m = 0 and diversity = 0."""
    df_singleton = pd.DataFrame({
        "candidate_id": [0],
        "timestamp": [1.0],
        "vehicle_x": [10.0],
        "vehicle_y": [20.0],
    })

    res_df = compute_diversity_features(df_singleton, saturation_baseline_m=5.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["max_baseline_m"] == pytest.approx(0.0, abs=1e-8)
    assert res_df.iloc[0]["viewpoint_diversity"] == pytest.approx(0.0, abs=1e-8)
    assert res_df.iloc[0]["num_unique_timestamps"] == 1


def test_2_two_viewpoints_half_saturation():
    """TEST 2: Two viewpoints separated by 2.5 m with saturation_baseline_m = 5.0 yield diversity = 0.5."""
    df = pd.DataFrame({
        "candidate_id": [0, 0],
        "timestamp": [0.0, 1.0],
        "vehicle_x": [0.0, 2.5],
        "vehicle_y": [0.0, 0.0],
    })

    res_df = compute_diversity_features(df, saturation_baseline_m=5.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["max_baseline_m"] == pytest.approx(2.5, abs=1e-8)
    assert res_df.iloc[0]["viewpoint_diversity"] == pytest.approx(0.5, abs=1e-8)


def test_3_two_viewpoints_exact_saturation():
    """TEST 3: Two viewpoints separated by 5.0 m yield diversity = 1.0."""
    df = pd.DataFrame({
        "candidate_id": [0, 0],
        "timestamp": [0.0, 1.0],
        "vehicle_x": [0.0, 3.0],
        "vehicle_y": [0.0, 4.0],
    })  # dist = sqrt(3^2 + 4^2) = 5.0

    res_df = compute_diversity_features(df, saturation_baseline_m=5.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["max_baseline_m"] == pytest.approx(5.0, abs=1e-8)
    assert res_df.iloc[0]["viewpoint_diversity"] == pytest.approx(1.0, abs=1e-8)


def test_4_two_viewpoints_above_saturation():
    """TEST 4: Two viewpoints separated by 10.0 m yield diversity = 1.0 (saturates)."""
    score = calculate_diversity_score(max_baseline_m=10.0, saturation_baseline_m=5.0)
    assert score == pytest.approx(1.0, abs=1e-8)


def test_5_duplicate_timestamps():
    """TEST 5: Duplicate timestamps do NOT create artificial viewpoint diversity."""
    # Timestamp 0.0 -> 3 observations at vehicle_x=0.0
    # Timestamp 0.1 -> 2 observations at vehicle_x=1.0
    df = pd.DataFrame({
        "candidate_id": [0, 0, 0, 0, 0],
        "timestamp": [0.0, 0.0, 0.0, 0.1, 0.1],
        "vehicle_x": [0.0, 0.0, 0.0, 1.0, 1.0],
        "vehicle_y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })

    res_df = compute_diversity_features(df, saturation_baseline_m=5.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["observation_count"] == 5
    assert res_df.iloc[0]["num_unique_timestamps"] == 2
    assert res_df.iloc[0]["max_baseline_m"] == pytest.approx(1.0, abs=1e-8)
    assert res_df.iloc[0]["viewpoint_diversity"] == pytest.approx(0.2, abs=1e-8)


def test_6_known_three_view_geometry():
    """TEST 6: Known three-view geometry (0,0), (3,0), (0,4) yields max baseline 5.0 m and diversity 1.0."""
    uv_x = np.array([0.0, 3.0, 0.0])
    uv_y = np.array([0.0, 0.0, 4.0])

    max_b, mean_b, median_b, min_b = compute_pairwise_baselines(uv_x, uv_y)
    score = calculate_diversity_score(max_b, saturation_baseline_m=5.0)

    # Pairs: (0,0)-(3,0)=3, (0,0)-(0,4)=4, (3,0)-(0,4)=5
    assert max_b == pytest.approx(5.0, abs=1e-8)
    assert min_b == pytest.approx(3.0, abs=1e-8)
    assert mean_b == pytest.approx((3.0 + 4.0 + 5.0) / 3.0, abs=1e-8)  # 4.0
    assert score == pytest.approx(1.0, abs=1e-8)


def test_7_actual_dataset():
    """TEST 7: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_csv = os.path.join(base_dir, "outputs", "ewmvc_candidate_components.csv")

    if not os.path.exists(cand_csv):
        pytest.skip(f"Input candidate CSV '{cand_csv}' does not exist.")

    cand_df = pd.read_csv(cand_csv)

    # Run without label column
    cand_no_labels = cand_df.drop(columns=["label"], errors="ignore")
    res_df = compute_diversity_features(cand_no_labels, saturation_baseline_m=5.0)

    assert len(res_df) == 141, "Expected exactly 141 candidate components"
    assert res_df["candidate_id"].nunique() == 141

    # Check bounds [0, 1] and zero NaNs
    assert not res_df.isna().any().any()
    assert (res_df["viewpoint_diversity"] >= 0.0).all()
    assert (res_df["viewpoint_diversity"] <= 1.0).all()

    # Singletons check (max_baseline = 0, score = 0)
    singletons = res_df[res_df["observation_count"] == 1]
    assert (singletons["max_baseline_m"] == 0.0).all()
    assert (singletons["viewpoint_diversity"] == 0.0).all()
    assert (singletons["num_unique_timestamps"] == 1).all()

    # Check required columns
    required_cols = [
        "candidate_id", "observation_count", "num_unique_timestamps",
        "max_baseline_m", "mean_pairwise_baseline_m", "median_pairwise_baseline_m",
        "min_pairwise_baseline_m", "viewpoint_diversity"
    ]
    for col in required_cols:
        assert col in res_df.columns

    # Label independence
    if "label" in cand_df.columns:
        res_with_labels = compute_diversity_features(cand_df, saturation_baseline_m=5.0)
        pd.testing.assert_frame_equal(res_df, res_with_labels)
