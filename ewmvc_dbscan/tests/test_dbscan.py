"""
Unit tests for src/dbscan_mapper.py module.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates
from src.dbscan_mapper import run_dbscan, summarize_clusters


def test_run_dbscan_basic():
    df = pd.DataFrame({
        "global_x": [0.0, 0.1, 0.2, 10.0],
        "global_y": [0.0, 0.1, 0.2, 10.0],
    })

    clustered = run_dbscan(df, eps_m=1.0, min_samples=3)

    assert "cluster_id" in clustered.columns
    # First 3 points should form cluster 0, 4th point is noise (-1)
    assert clustered.loc[0, "cluster_id"] == 0
    assert clustered.loc[1, "cluster_id"] == 0
    assert clustered.loc[2, "cluster_id"] == 0
    assert clustered.loc[3, "cluster_id"] == -1


def test_run_dbscan_invalid_input():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError):
        run_dbscan(df)


def test_summarize_clusters_weighted_centroid():
    # Synthetic cluster: Point 1 (x=0, conf=0.2), Point 2 (x=10, conf=0.8)
    clustered = pd.DataFrame({
        "cluster_id": [0, 0],
        "global_x": [0.0, 10.0],
        "global_y": [0.0, 0.0],
        "confidence": [0.2, 0.8],
        "label": ["real", "real"],
        "timestamp": [0.0, 0.1],
    })

    summary = summarize_clusters(clustered)

    assert len(summary) == 1
    # Weighted centroid = (0.2*0 + 0.8*10) / 1.0 = 8.0
    np.testing.assert_allclose(summary.loc[0, "global_x"], 8.0, atol=1e-6)
    np.testing.assert_allclose(summary.loc[0, "global_y"], 0.0, atol=1e-6)
    assert summary.loc[0, "observation_count"] == 2
    assert summary.loc[0, "unique_frames"] == 2


def test_summarize_clusters_equal_confidence():
    clustered = pd.DataFrame({
        "cluster_id": [0, 0],
        "global_x": [0.0, 10.0],
        "global_y": [0.0, 10.0],
        "confidence": [0.5, 0.5],
        "label": ["real", "ghost"],
        "timestamp": [1.0, 2.0],
    })

    summary = summarize_clusters(clustered)

    assert len(summary) == 1
    np.testing.assert_allclose(summary.loc[0, "global_x"], 5.0, atol=1e-6)
    np.testing.assert_allclose(summary.loc[0, "global_y"], 5.0, atol=1e-6)
    assert summary.loc[0, "real_observations"] == 1
    assert summary.loc[0, "ghost_observations"] == 1
    np.testing.assert_allclose(summary.loc[0, "ghost_fraction"], 0.5, atol=1e-6)


def test_dbscan_on_actual_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    t_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")

    p, t = load_logs(p_path, t_path)
    p_interp = interpolate_telemetry(p, t)
    df_global = add_global_coordinates(p_interp, rotation_deg=180.0)

    clustered = run_dbscan(df_global, eps_m=1.0, min_samples=4)
    cone_map = summarize_clusters(clustered)

    noise_count = (clustered["cluster_id"] == -1).sum()

    assert len(cone_map) == 72, "DBSCAN on actual dataset must yield 72 clusters"
    assert noise_count == 66, "DBSCAN on actual dataset must yield 66 noise observations"
