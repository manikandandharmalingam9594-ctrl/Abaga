"""
Unit tests for src/entropy.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.entropy import (
    compute_viewpoint_angle,
    assign_viewpoint_bin,
    calculate_normalized_entropy,
    compute_viewpoint_entropy,
)


def test_1_all_observations_in_one_viewpoint_bin():
    """TEST 1: All observations in one viewpoint bin should yield normalized entropy = 0."""
    bin_counts = np.array([10, 0, 0, 0, 0, 0, 0, 0])
    h_norm, unique_bins = calculate_normalized_entropy(bin_counts, num_bins=8)

    assert unique_bins == 1
    assert h_norm == pytest.approx(0.0, abs=1e-8)


def test_2_two_equally_populated_bins():
    """TEST 2: Two equally populated bins should yield normalized entropy approx 0.333333 (ln(2)/ln(8))."""
    bin_counts = np.array([5, 5, 0, 0, 0, 0, 0, 0])
    h_norm, unique_bins = calculate_normalized_entropy(bin_counts, num_bins=8)

    expected = np.log(2) / np.log(8)  # 1/3 = 0.333333...
    assert unique_bins == 2
    assert h_norm == pytest.approx(expected, abs=1e-6)
    assert h_norm == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_3_eight_equally_populated_bins():
    """TEST 3: Eight equally populated bins should yield normalized entropy approx 1.0."""
    bin_counts = np.array([4, 4, 4, 4, 4, 4, 4, 4])
    h_norm, unique_bins = calculate_normalized_entropy(bin_counts, num_bins=8)

    assert unique_bins == 8
    assert h_norm == pytest.approx(1.0, abs=1e-6)


def test_4_singleton_component():
    """TEST 4: Singleton component should yield normalized entropy = 0."""
    df_singleton = pd.DataFrame({
        "candidate_id": [0],
        "global_x": [10.0],
        "global_y": [5.0],
        "vehicle_x": [0.0],
        "vehicle_y": [0.0],
    })

    entropy_df = compute_viewpoint_entropy(df_singleton, num_bins=8)

    assert len(entropy_df) == 1
    assert entropy_df.iloc[0]["num_observations"] == 1
    assert entropy_df.iloc[0]["unique_viewpoint_bins"] == 1
    assert entropy_df.iloc[0]["viewpoint_entropy"] == pytest.approx(0.0, abs=1e-8)


def test_5_angle_wrapping_around_179_and_minus_179_degrees():
    """TEST 5: Angle wrapping around +179 and -179 degrees is stable and valid."""
    # +179 deg = +3.1241 rad, -179 deg = -3.1241 rad
    angles = np.array([3.124139, -3.124139, 3.14159, -3.14159])
    bins = assign_viewpoint_bin(angles, num_bins=8)

    # All these angles fall into the bin containing -pi/+pi (Bin 0 or 7 depending on convention, but valid range [0, 7])
    assert (bins >= 0).all() and (bins < 8).all()

    # Viewpoint angle computation check
    angles_calc = compute_viewpoint_angle(
        vehicle_x=np.array([10.0, -10.0]),
        vehicle_y=np.array([0.0, 0.0]),
        candidate_x=0.0,
        candidate_y=0.0,
    )
    # atan2(0 - 0, 0 - 10) = atan2(0, -10) = pi -> wrapped to -pi
    # atan2(0 - 0, 0 - (-10)) = atan2(0, 10) = 0.0
    assert (angles_calc >= -np.pi).all() and (angles_calc <= np.pi).all()


def test_6_actual_candidate_dataset():
    """TEST 6: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_csv = os.path.join(base_dir, "outputs", "ewmvc_candidate_components.csv")

    if not os.path.exists(cand_csv):
        pytest.skip(f"Input file '{cand_csv}' does not exist.")

    cand_df = pd.read_csv(cand_csv)

    # Run without label column
    cand_no_labels = cand_df.drop(columns=["label"], errors="ignore")
    entropy_df = compute_viewpoint_entropy(cand_no_labels, num_bins=8)

    assert len(entropy_df) == 141, "Expected exactly 141 candidate components"
    assert entropy_df["candidate_id"].nunique() == 141

    # Check bounds
    assert (entropy_df["viewpoint_entropy"] >= 0.0).all()
    assert (entropy_df["viewpoint_entropy"] <= 1.0).all()

    # Check singletons have entropy = 0
    singletons = entropy_df[entropy_df["num_observations"] == 1]
    assert (singletons["viewpoint_entropy"] == 0.0).all()
    assert (singletons["unique_viewpoint_bins"] == 1).all()

    # Verify label independence: running with or without label column gives identical results
    if "label" in cand_df.columns:
        entropy_with_labels = compute_viewpoint_entropy(cand_df, num_bins=8)
        pd.testing.assert_frame_equal(entropy_df, entropy_with_labels)
