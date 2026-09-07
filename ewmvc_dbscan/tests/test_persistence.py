"""
Unit tests for src/persistence.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.persistence import (
    calculate_persistence_score,
    compute_persistence_features,
)


def test_1_singleton_candidate():
    """TEST 1: Singleton candidate (one observation at t=5) yields duration = 0 and persistence_score = 0."""
    df_singleton = pd.DataFrame({
        "candidate_id": [0],
        "timestamp": [5.0],
    })

    res_df = compute_persistence_features(df_singleton, expected_duration_s=10.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["duration_s"] == pytest.approx(0.0, abs=1e-8)
    assert res_df.iloc[0]["persistence_score"] == pytest.approx(0.0, abs=1e-8)
    assert res_df.iloc[0]["num_unique_timestamps"] == 1


def test_2_time_span_calculation():
    """TEST 2: Observations at t = 0, 1, 2, 3, 4 seconds yield duration = 4 seconds (t_max - t_min)."""
    df = pd.DataFrame({
        "candidate_id": [0, 0, 0, 0, 0],
        "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
    })

    res_df = compute_persistence_features(df, expected_duration_s=10.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["duration_s"] == pytest.approx(4.0, abs=1e-8)
    assert res_df.iloc[0]["persistence_score"] == pytest.approx(0.4, abs=1e-8)
    assert res_df.iloc[0]["num_unique_timestamps"] == 5


def test_3_duration_equals_expected_duration():
    """TEST 3: duration_s exactly equal to expected_duration_s yields persistence_score = 1.0."""
    score = calculate_persistence_score(duration_s=10.0, expected_duration_s=10.0)
    assert score == pytest.approx(1.0, abs=1e-8)


def test_4_duration_greater_than_expected_duration():
    """TEST 4: duration_s greater than expected_duration_s yields persistence_score = 1.0."""
    score = calculate_persistence_score(duration_s=25.0, expected_duration_s=10.0)
    assert score == pytest.approx(1.0, abs=1e-8)


def test_5_duplicate_timestamps():
    """TEST 5: Duplicate timestamps (0, 0, 0.1, 0.1, 0.2) yield duration = 0.2s and num_unique_timestamps = 3."""
    df = pd.DataFrame({
        "candidate_id": [0, 0, 0, 0, 0],
        "timestamp": [0.0, 0.0, 0.1, 0.1, 0.2],
    })

    res_df = compute_persistence_features(df, expected_duration_s=10.0)

    assert len(res_df) == 1
    assert res_df.iloc[0]["observation_count"] == 5
    assert res_df.iloc[0]["num_unique_timestamps"] == 3
    assert res_df.iloc[0]["duration_s"] == pytest.approx(0.2, abs=1e-8)
    assert res_df.iloc[0]["persistence_score"] == pytest.approx(0.02, abs=1e-8)


def test_6_actual_dataset():
    """TEST 6: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_csv = os.path.join(base_dir, "outputs", "ewmvc_candidate_components.csv")

    if not os.path.exists(cand_csv):
        pytest.skip(f"Input candidate CSV '{cand_csv}' does not exist.")

    cand_df = pd.read_csv(cand_csv)

    # Run without label column
    cand_no_labels = cand_df.drop(columns=["label"], errors="ignore")
    res_df = compute_persistence_features(cand_no_labels, expected_duration_s=10.0)

    assert len(res_df) == 141, "Expected exactly 141 candidate components"
    assert res_df["candidate_id"].nunique() == 141

    # Check bounds [0, 1] and zero NaNs
    assert not res_df.isna().any().any()
    assert (res_df["persistence_score"] >= 0.0).all()
    assert (res_df["persistence_score"] <= 1.0).all()

    # Singletons check (duration = 0, score = 0, unique_timestamps = 1)
    singletons = res_df[res_df["observation_count"] == 1]
    assert (singletons["duration_s"] == 0.0).all()
    assert (singletons["persistence_score"] == 0.0).all()
    assert (singletons["num_unique_timestamps"] == 1).all()

    # Label independence
    if "label" in cand_df.columns:
        res_with_labels = compute_persistence_features(cand_df, expected_duration_s=10.0)
        pd.testing.assert_frame_equal(res_df, res_with_labels)
