"""
Unit tests for src/graph_builder.py module.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.graph_builder import (
    compute_mahalanobis_distance_sq,
    build_mahalanobis_graph,
    generate_graph_summary,
)


def test_1_identical_points_zero_mahalanobis():
    """Test 1: Identical points with equal covariance should have d²_M = 0."""
    p1 = np.array([2.5, 3.5])
    p2 = np.array([2.5, 3.5])
    cov1 = np.array([[0.05, 0.01], [0.01, 0.04]])
    cov2 = np.array([[0.05, 0.01], [0.01, 0.04]])

    d_sq = compute_mahalanobis_distance_sq(p1, cov1, p2, cov2)
    assert d_sq == pytest.approx(0.0, abs=1e-8)


def test_2_known_covariance_analytically_expected_mahalanobis():
    """Test 2: Two points with known covariance produce analytically expected Mahalanobis distance."""
    # p1 = [0, 0], p2 = [1, 0] => dp = [1, 0]
    # cov1 = cov2 = diag(0.1, 0.1) => cov_sum = diag(0.2, 0.2)
    # inv(cov_sum) = diag(5.0, 5.0)
    # d²_M = [1, 0] * [5.0, 0; 0, 5.0] * [1, 0]^T = 5.0
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    cov1 = np.array([[0.1, 0.0], [0.0, 0.1]])
    cov2 = np.array([[0.1, 0.0], [0.0, 0.1]])

    d_sq = compute_mahalanobis_distance_sq(p1, cov1, p2, cov2)
    assert d_sq == pytest.approx(5.0, abs=1e-6)


def test_3_increasing_covariance_reduces_mahalanobis():
    """Test 3: Increasing covariance should reduce Mahalanobis distance for the same spatial separation."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])

    # Small covariance (high certainty) -> larger d²_M
    cov_small = np.array([[0.1, 0.0], [0.0, 0.1]])
    d_sq_small_cov = compute_mahalanobis_distance_sq(p1, cov_small, p2, cov_small)

    # Large covariance (high uncertainty) -> smaller d²_M
    cov_large = np.array([[0.5, 0.0], [0.0, 0.5]])
    d_sq_large_cov = compute_mahalanobis_distance_sq(p1, cov_large, p2, cov_large)

    assert d_sq_small_cov == pytest.approx(5.0, abs=1e-6)
    assert d_sq_large_cov == pytest.approx(1.0, abs=1e-6)
    assert d_sq_large_cov < d_sq_small_cov


def test_4_points_beyond_threshold_do_not_form_edge():
    """Test 4: Points beyond the compatibility threshold should not form an edge."""
    # d²_M = 7.5 > chi2_threshold (5.991)
    df = pd.DataFrame({
        "global_x": [0.0, 1.5],
        "global_y": [0.0, 0.0],
        "cov_xx": [0.15, 0.15],
        "cov_xy": [0.0, 0.0],
        "cov_yy": [0.15, 0.15],
    })

    # dp = [1.5, 0], cov_sum = diag(0.3, 0.3), inv = diag(1/0.3, 1/0.3)
    # d²_M = (1.5^2) / 0.3 = 2.25 / 0.3 = 7.5 > 5.991
    adj, candidate_ids, degrees, n_components = build_mahalanobis_graph(
        df, chi_square_threshold=5.991, candidate_radius_m=2.0
    )

    assert adj.nnz == 0
    assert n_components == 2
    assert (degrees == 0).all()
    assert candidate_ids[0] != candidate_ids[1]


def test_5_points_within_threshold_form_edge():
    """Test 5: Points within the threshold should form an edge."""
    # d²_M = 5.0 <= chi2_threshold (5.991)
    df = pd.DataFrame({
        "global_x": [0.0, 1.0],
        "global_y": [0.0, 0.0],
        "cov_xx": [0.1, 0.1],
        "cov_xy": [0.0, 0.0],
        "cov_yy": [0.1, 0.1],
    })

    adj, candidate_ids, degrees, n_components = build_mahalanobis_graph(
        df, chi_square_threshold=5.991, candidate_radius_m=2.0
    )

    assert adj.nnz == 2  # Undirected (0->1, 1->0)
    assert n_components == 1
    assert (degrees == 1).all()
    assert candidate_ids[0] == candidate_ids[1]


def test_6_symmetric_pairwise_distance():
    """Test 6: Symmetric pairwise distance: d_M(i,j) = d_M(j,i)."""
    p1 = np.array([1.2, -3.4])
    p2 = np.array([2.1, 0.5])
    cov1 = np.array([[0.08, 0.02], [0.02, 0.12]])
    cov2 = np.array([[0.10, -0.01], [-0.01, 0.09]])

    d_ij = compute_mahalanobis_distance_sq(p1, cov1, p2, cov2)
    d_ji = compute_mahalanobis_distance_sq(p2, cov2, p1, cov1)

    assert d_ij == pytest.approx(d_ji, abs=1e-8)


def test_7_singular_or_invalid_covariance_handled_safely():
    """Test 7: Singular/invalid covariance is handled safely without crashing."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    cov_singular = np.array([[0.0, 0.0], [0.0, 0.0]])
    cov_valid = np.array([[0.1, 0.0], [0.0, 0.1]])

    # Should not crash, should return inf or valid float
    d_sq = compute_mahalanobis_distance_sq(p1, cov_singular, p2, cov_valid)
    assert np.isinf(d_sq) or d_sq >= 0.0


def test_8_connected_components_extracted_from_small_synthetic_graph():
    """Test 8: Connected components are correctly extracted from a small synthetic graph."""
    # Group A: nodes 0, 1 (close together)
    # Group B: nodes 2, 3 (close together, far from Group A)
    df = pd.DataFrame({
        "global_x": [0.0, 0.2, 10.0, 10.2],
        "global_y": [0.0, 0.1, 10.0, 10.1],
        "cov_xx": [0.1, 0.1, 0.1, 0.1],
        "cov_xy": [0.0, 0.0, 0.0, 0.0],
        "cov_yy": [0.1, 0.1, 0.1, 0.1],
    })

    adj, candidate_ids, degrees, n_components = build_mahalanobis_graph(
        df, chi_square_threshold=5.991, candidate_radius_m=1.5
    )

    assert n_components == 2
    assert candidate_ids[0] == candidate_ids[1]
    assert candidate_ids[2] == candidate_ids[3]
    assert candidate_ids[0] != candidate_ids[2]


def test_9_candidate_ids_assigned_correctly():
    """Test 9: Candidate IDs are assigned correctly including singletons."""
    # 3 nodes: node 0 & 1 connected, node 2 singleton
    df = pd.DataFrame({
        "global_x": [0.0, 0.1, 50.0],
        "global_y": [0.0, 0.0, 50.0],
        "cov_xx": [0.1, 0.1, 0.1],
        "cov_xy": [0.0, 0.0, 0.0],
        "cov_yy": [0.1, 0.1, 0.1],
    })

    adj, candidate_ids, degrees, n_components = build_mahalanobis_graph(
        df, chi_square_threshold=5.991, candidate_radius_m=1.5
    )

    assert len(candidate_ids) == 3
    assert n_components == 2
    assert candidate_ids[0] == candidate_ids[1]
    assert candidate_ids[2] != candidate_ids[0]
    assert degrees[2] == 0


def test_10_graph_construction_does_not_use_label_column():
    """Test 10: Graph construction does NOT use the 'label' column."""
    df_with_labels = pd.DataFrame({
        "global_x": [0.0, 0.2],
        "global_y": [0.0, 0.0],
        "cov_xx": [0.1, 0.1],
        "cov_xy": [0.0, 0.0],
        "cov_yy": [0.1, 0.1],
        "label": ["real", "ghost"],
    })

    df_no_labels = df_with_labels.drop(columns=["label"])

    adj1, cids1, degs1, n1 = build_mahalanobis_graph(df_with_labels, chi_square_threshold=5.991)
    adj2, cids2, degs2, n2 = build_mahalanobis_graph(df_no_labels, chi_square_threshold=5.991)

    assert (adj1.toarray() == adj2.toarray()).all()
    assert (cids1 == cids2).all()
    assert (degs1 == degs2).all()
    assert n1 == n2
