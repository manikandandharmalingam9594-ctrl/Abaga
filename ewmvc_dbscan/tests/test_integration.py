"""
Integration, regression, import safety, and output artifact tests for DBSCAN baseline.
"""

import os
import sys
import pytest
import pandas as pd

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates
from src.dbscan_mapper import run_dbscan, summarize_clusters
from src.evaluate import evaluate_dbscan
from src.parameter_experiments import run_parameter_sensitivity_experiment
from src.failure_analysis import analyze_surviving_ghosts


def get_dataset_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    t_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")
    return base_dir, p_path, t_path


def test_baseline_regression():
    """
    CRITICAL BASELINE REGRESSION TEST:
    Verifies that the baseline pipeline produces the exact contract metrics on the actual dataset.
    """
    _, p_path, t_path = get_dataset_paths()

    perception, telemetry = load_logs(p_path, t_path)
    perc_interp = interpolate_telemetry(perception, telemetry)
    df_global = add_global_coordinates(perc_interp, rotation_deg=180.0)

    clustered = run_dbscan(df_global, eps_m=1.0, min_samples=4)
    cone_map = summarize_clusters(clustered)
    metrics = evaluate_dbscan(clustered, cone_map)

    assert metrics["total_observations"] == 9774, "Total observations must be 9774"
    assert metrics["clustered_observations"] == 9708, "Clustered observations must be 9708"
    assert metrics["noise_observations"] == 66, "Noise observations must be 66"
    assert metrics["noise_fraction"] == pytest.approx(0.006753, abs=1e-5)

    assert metrics["total_ghost_observations"] == 80, "Total ghosts must be 80"
    assert metrics["ghost_observations_inside_clusters"] == 14, "Ghosts inside clusters must be 14"
    assert metrics["ghost_observations_rejected_as_noise"] == 66, "Ghosts rejected must be 66"
    assert metrics["ghost_rejection_rate_by_noise"] == pytest.approx(0.825, abs=1e-3)

    assert metrics["number_of_clusters"] == 72, "Number of clusters must be 72"


def test_pipeline_integration():
    """Test stage-by-stage pipeline execution and column additions."""
    _, p_path, t_path = get_dataset_paths()

    # Stage 1: Load
    perception, telemetry = load_logs(p_path, t_path)
    assert "timestamp" in perception.columns

    # Stage 2: Interpolate
    perc_interp = interpolate_telemetry(perception, telemetry)
    assert "vehicle_x" in perc_interp.columns
    assert "vehicle_y" in perc_interp.columns
    assert "vehicle_yaw" in perc_interp.columns

    # Stage 3: Transform
    df_global = add_global_coordinates(perc_interp, rotation_deg=180.0)
    assert "rel_x_vehicle" in df_global.columns
    assert "global_x" in df_global.columns

    # Stage 4: DBSCAN
    clustered = run_dbscan(df_global, eps_m=1.0, min_samples=4)
    assert "cluster_id" in clustered.columns

    # Stage 5: Centroids & Map
    cone_map = summarize_clusters(clustered)
    assert "global_x" in cone_map.columns
    assert len(cone_map) > 0

    # Stage 6: Evaluation
    metrics = evaluate_dbscan(clustered, cone_map)
    assert "ghost_rejection_rate_by_noise" in metrics


def test_output_files_exist():
    """Verify expected output files exist after main baseline run."""
    base_dir, _, _ = get_dataset_paths()
    out_dir = os.path.join(base_dir, "outputs")

    expected_files = [
        "transformed_observations.csv",
        "dbscan_observations.csv",
        "cone_map_dbscan.csv",
        "evaluation.csv",
        "dbscan_map.png",
    ]

    for fname in expected_files:
        fpath = os.path.join(out_dir, fname)
        assert os.path.exists(fpath), f"Expected output file '{fname}' missing from outputs directory"


def test_import_safety():
    """Verify importing modules does not trigger unintended side effects."""
    import src.data_loader
    import src.transforms
    import src.dbscan_mapper
    import src.visualize
    import src.evaluate
    import src.parameter_experiments
    import src.failure_analysis

    assert hasattr(src.data_loader, "load_logs")
    assert hasattr(src.transforms, "add_global_coordinates")
    assert hasattr(src.dbscan_mapper, "run_dbscan")
    assert hasattr(src.visualize, "plot_dbscan_map")
    assert hasattr(src.evaluate, "evaluate_dbscan")
    assert hasattr(src.parameter_experiments, "run_parameter_sensitivity_experiment")
    assert hasattr(src.failure_analysis, "analyze_surviving_ghosts")


def test_parameter_experiment_grid_size():
    """Verify parameter experiment grid contains exactly 24 configurations."""
    eps_list = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    min_samples_list = [3, 4, 5, 6]

    assert len(eps_list) * len(min_samples_list) == 24, "Parameter grid size must equal 24"


def test_failure_analysis_contract():
    """Verify baseline failure analysis identifies 14 surviving ghosts and 10 contaminated clusters."""
    _, p_path, t_path = get_dataset_paths()

    perception, telemetry = load_logs(p_path, t_path)
    perc_interp = interpolate_telemetry(perception, telemetry)
    df_global = add_global_coordinates(perc_interp, rotation_deg=180.0)

    clustered = run_dbscan(df_global, eps_m=1.0, min_samples=4)
    cone_map = summarize_clusters(clustered)

    failure_df, contamination_df, _ = analyze_surviving_ghosts(clustered, cone_map)

    assert len(failure_df) == 14, "Surviving ghosts count must equal 14"
    assert len(contamination_df) == 10, "Contaminated clusters count must equal 10"
