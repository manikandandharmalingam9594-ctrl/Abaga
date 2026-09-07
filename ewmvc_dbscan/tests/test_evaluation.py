"""
Unit tests for src/evaluate.py module.
"""

import pytest
import pandas as pd

from src.evaluate import evaluate_dbscan, save_evaluation_metrics


def test_evaluate_dbscan_synthetic():
    # Synthetic dataset:
    # 4 points: 2 real in cluster 0, 1 ghost in cluster 0, 1 ghost as noise (-1)
    clustered = pd.DataFrame({
        "cluster_id": [0, 0, 0, -1],
        "label": ["real", "real", "ghost", "ghost"],
    })

    cone_map = pd.DataFrame({
        "cluster_id": [0],
        "observation_count": [3],
    })

    metrics = evaluate_dbscan(clustered, cone_map)

    assert metrics["total_observations"] == 4
    assert metrics["clustered_observations"] == 3
    assert metrics["noise_observations"] == 1
    assert metrics["noise_fraction"] == 0.25
    assert metrics["total_ghost_observations"] == 2
    assert metrics["ghost_observations_inside_clusters"] == 1
    assert metrics["ghost_observations_rejected_as_noise"] == 1
    assert metrics["ghost_rejection_rate_by_noise"] == 0.5
    assert metrics["number_of_clusters"] == 1
    assert metrics["largest_cluster_size"] == 3
    assert metrics["average_cluster_size"] == 3.0


def test_evaluate_dbscan_zero_division_safety():
    # No ghosts present in dataset
    clustered = pd.DataFrame({
        "cluster_id": [0, 0],
        "label": ["real", "real"],
    })
    cone_map = pd.DataFrame({
        "cluster_id": [0],
        "observation_count": [2],
    })

    metrics = evaluate_dbscan(clustered, cone_map)

    assert metrics["total_ghost_observations"] == 0
    assert metrics["ghost_rejection_rate_by_noise"] == 0.0
    assert metrics["noise_fraction"] == 0.0
