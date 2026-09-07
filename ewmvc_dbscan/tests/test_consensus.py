"""
Unit tests for src/consensus.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.consensus import (
    validate_weights,
    validate_feature_values,
    compute_consensus_score,
)


def test_1_all_features_one():
    """TEST 1: All features = 1.0 yield consensus_score = 1.0."""
    df_h = pd.DataFrame({"candidate_id": [0], "viewpoint_entropy": [1.0]})
    df_g = pd.DataFrame({"candidate_id": [0], "geometry_score": [1.0]})
    df_p = pd.DataFrame({"candidate_id": [0], "persistence_score": [1.0]})
    df_d = pd.DataFrame({"candidate_id": [0], "viewpoint_diversity": [1.0]})

    res_df, _, _ = compute_consensus_score(df_h, df_g, df_p, df_d, acceptance_threshold=0.60)

    assert len(res_df) == 1
    assert res_df.iloc[0]["consensus_score"] == pytest.approx(1.0, abs=1e-8)
    assert res_df.iloc[0]["accepted"] is True or res_df.iloc[0]["accepted"] == 1


def test_2_low_zero_features():
    """TEST 2: H=0, G=0.1, P=0, D=0 yields consensus_score = 0.3*0 + 0.3*0.1 + 0.2*0 + 0.2*0 = 0.03."""
    df_h = pd.DataFrame({"candidate_id": [0], "viewpoint_entropy": [0.0]})
    df_g = pd.DataFrame({"candidate_id": [0], "geometry_score": [0.1]})
    df_p = pd.DataFrame({"candidate_id": [0], "persistence_score": [0.0]})
    df_d = pd.DataFrame({"candidate_id": [0], "viewpoint_diversity": [0.0]})

    res_df, _, _ = compute_consensus_score(df_h, df_g, df_p, df_d, acceptance_threshold=0.60)

    assert len(res_df) == 1
    assert res_df.iloc[0]["consensus_score"] == pytest.approx(0.03, abs=1e-8)
    assert not bool(res_df.iloc[0]["accepted"])


def test_3_known_values():
    """TEST 3: H=0.5, G=0.8, P=1.0, D=0.4 yields consensus_score = 0.67."""
    # C = 0.3(0.5) + 0.3(0.8) + 0.2(0.4) + 0.2(1.0) = 0.15 + 0.24 + 0.08 + 0.20 = 0.67
    df_h = pd.DataFrame({"candidate_id": [0], "viewpoint_entropy": [0.5]})
    df_g = pd.DataFrame({"candidate_id": [0], "geometry_score": [0.8]})
    df_p = pd.DataFrame({"candidate_id": [0], "persistence_score": [1.0]})
    df_d = pd.DataFrame({"candidate_id": [0], "viewpoint_diversity": [0.4]})

    res_df, _, _ = compute_consensus_score(df_h, df_g, df_p, df_d, acceptance_threshold=0.60)

    assert len(res_df) == 1
    assert res_df.iloc[0]["consensus_score"] == pytest.approx(0.67, abs=1e-8)


def test_4_acceptance_threshold():
    """TEST 4: Acceptance threshold checks for 0.59 (rejected), 0.60 (accepted), 0.61 (accepted)."""
    df_h = pd.DataFrame({"candidate_id": [0, 1, 2], "viewpoint_entropy": [0.59, 0.60, 0.61]})
    df_g = pd.DataFrame({"candidate_id": [0, 1, 2], "geometry_score": [0.59, 0.60, 0.61]})
    df_p = pd.DataFrame({"candidate_id": [0, 1, 2], "persistence_score": [0.59, 0.60, 0.61]})
    df_d = pd.DataFrame({"candidate_id": [0, 1, 2], "viewpoint_diversity": [0.59, 0.60, 0.61]})

    res_df, _, _ = compute_consensus_score(df_h, df_g, df_p, df_d, acceptance_threshold=0.60)

    assert not bool(res_df.iloc[0]["accepted"])
    assert bool(res_df.iloc[1]["accepted"])
    assert bool(res_df.iloc[2]["accepted"])


def test_5_weight_validation():
    """TEST 5: Weight validation raises ValueError if sum != 1.0 or any weight < 0."""
    with pytest.raises(ValueError):
        validate_weights({"entropy": 0.3, "geometry": 0.3, "diversity": 0.2, "persistence": 0.1})  # sum = 0.9

    with pytest.raises(ValueError):
        validate_weights({"entropy": -0.1, "geometry": 0.5, "diversity": 0.3, "persistence": 0.3})  # negative weight


def test_6_contribution_consistency():
    """TEST 6: Contribution sum equals consensus_score within numerical tolerance."""
    df_h = pd.DataFrame({"candidate_id": [0, 1], "viewpoint_entropy": [0.2, 0.7]})
    df_g = pd.DataFrame({"candidate_id": [0, 1], "geometry_score": [0.9, 0.4]})
    df_p = pd.DataFrame({"candidate_id": [0, 1], "persistence_score": [0.3, 0.8]})
    df_d = pd.DataFrame({"candidate_id": [0, 1], "viewpoint_diversity": [0.5, 0.6]})

    res_df, _, _ = compute_consensus_score(df_h, df_g, df_p, df_d, acceptance_threshold=0.60)

    for i in range(len(res_df)):
        c_sum = (
            res_df.iloc[i]["entropy_contribution"]
            + res_df.iloc[i]["geometry_contribution"]
            + res_df.iloc[i]["diversity_contribution"]
            + res_df.iloc[i]["persistence_contribution"]
        )
        assert c_sum == pytest.approx(res_df.iloc[i]["consensus_score"], abs=1e-8)


def test_7_actual_dataset():
    """TEST 7: Actual candidate dataset validation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")

    f_entropy = os.path.join(out_dir, "ewmvc_candidate_entropy.csv")
    f_geometry = os.path.join(out_dir, "ewmvc_candidate_geometry.csv")
    f_persistence = os.path.join(out_dir, "ewmvc_candidate_persistence.csv")
    f_diversity = os.path.join(out_dir, "ewmvc_candidate_diversity.csv")

    for f_path in [f_entropy, f_geometry, f_persistence, f_diversity]:
        if not os.path.exists(f_path):
            pytest.skip(f"Feature CSV '{f_path}' missing.")

    df_h = pd.read_csv(f_entropy)
    df_g = pd.read_csv(f_geometry)
    df_p = pd.read_csv(f_persistence)
    df_d = pd.read_csv(f_diversity)

    res_df, corr_df, ablation_df = compute_consensus_score(
        df_h, df_g, df_p, df_d, acceptance_threshold=0.60
    )

    assert len(res_df) == 141, "Expected exactly 141 candidate components"
    assert res_df["candidate_id"].nunique() == 141
    assert not res_df.isna().any().any()

    assert (res_df["consensus_score"] >= 0.0).all()
    assert (res_df["consensus_score"] <= 1.0).all()

    assert res_df["accepted"].dtype == bool or res_df["accepted"].dtype == np.bool_ or set(res_df["accepted"].unique()).issubset({True, False, 1, 0})

    # Check contribution sum consistency
    contrib_sum = (
        res_df["entropy_contribution"]
        + res_df["geometry_contribution"]
        + res_df["diversity_contribution"]
        + res_df["persistence_contribution"]
    )
    np.testing.assert_allclose(contrib_sum.values, res_df["consensus_score"].values, atol=1e-8)

    # Check correlation matrix shape (4x4)
    assert corr_df.shape == (4, 4)
    assert (corr_df.values >= -1.0).all() and (corr_df.values <= 1.0).all()

    # Check ablation table shape (4 rows)
    assert len(ablation_df) == 4
