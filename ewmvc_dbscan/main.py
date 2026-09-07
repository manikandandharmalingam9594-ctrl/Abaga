"""
Main End-to-End Pipeline Execution Script for Fixed-Epsilon DBSCAN Baseline.

Pipeline Stages:
1. Load perception and telemetry CSV logs & validate schemas and missing values.
2. Interpolate telemetry vehicle poses (vehicle_x, vehicle_y, vehicle_yaw) to perception timestamps.
3. Transform sensor coordinates -> vehicle coordinates (180 deg sensor rotation).
4. Transform vehicle coordinates -> global coordinates using vehicle SE(2) pose.
5. Save transformed observations to outputs/transformed_observations.csv.
6. Run standard Euclidean DBSCAN clustering (eps = 1.0 m, min_samples = 4).
7. Compute confidence-weighted cluster centroids and build cone map.
8. Run observation-level evaluation and calculate diagnostics.
9. Generate DBSCAN map visualization plot.
10. Save all output CSVs and visualization images.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates
from src.dbscan_mapper import run_dbscan, summarize_clusters, save_cluster_outputs
from src.evaluate import evaluate_dbscan, save_evaluation_metrics
from src.visualize import plot_dbscan_map


def load_config(config_path: str) -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")

    print("======================================================================")
    print("  STARTING END-TO-END FIXED-EPSILON DBSCAN CONE MAPPING PIPELINE")
    print("======================================================================")

    # 1. Load Configuration Parameters
    config = load_config(config_path)
    perception_rel_path = config["data"]["perception_file"]
    telemetry_rel_path = config["data"]["telemetry_file"]
    rotation_deg = float(config["sensor"]["rotation_deg"])
    eps_m = float(config["dbscan"]["eps_m"])
    min_samples = int(config["dbscan"]["min_samples"])
    output_dir_rel = config["output"]["directory"]

    perception_path = os.path.join(base_dir, perception_rel_path)
    telemetry_path = os.path.join(base_dir, telemetry_rel_path)
    output_dir = os.path.join(base_dir, output_dir_rel)
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Load and Validate Data
    print("\n[STAGE 1/8] Loading perception and telemetry logs...")
    perception, telemetry = load_logs(perception_path, telemetry_path)

    # Validate missing values
    perc_missing = perception.isna().sum().sum()
    telem_missing = telemetry.isna().sum().sum()
    print(f"  - Perception observations loaded: {len(perception)} (Missing values: {perc_missing})")
    print(f"  - Telemetry frames loaded        : {len(telemetry)} (Missing values: {telem_missing})")
    if perc_missing > 0 or telem_missing > 0:
        print("[WARNING] Missing values detected in input datasets!", file=sys.stderr)

    # Stage 2: Interpolate Telemetry Pose
    print("\n[STAGE 2/8] Interpolating telemetry poses to perception timestamps...")
    perception_interp = interpolate_telemetry(perception, telemetry)
    print("  - Telemetry vehicle_x, vehicle_y, and unwrapped vehicle_yaw associated.")

    # Stage 3: Coordinate Transformations (Sensor -> Vehicle -> Global)
    print(f"\n[STAGE 3/8] Transforming coordinates (Sensor rotation = {rotation_deg} deg)...")
    transformed_df = add_global_coordinates(perception_interp, rotation_deg=rotation_deg)
    print("  - Added columns: rel_x_vehicle, rel_y_vehicle, global_x, global_y.")

    # Stage 4: Save Transformed Observations
    transformed_obs_path = os.path.join(output_dir, "transformed_observations.csv")
    print(f"\n[STAGE 4/8] Saving transformed observations to '{transformed_obs_path}'...")
    transformed_df.to_csv(transformed_obs_path, index=False)

    # Stage 5: Run Standard Fixed-Epsilon DBSCAN
    print(f"\n[STAGE 5/8] Running Euclidean DBSCAN clustering (eps = {eps_m} m, min_samples = {min_samples})...")
    clustered_df = run_dbscan(transformed_df, eps_m=eps_m, min_samples=min_samples)
    noise_count = (clustered_df["cluster_id"] == -1).sum()
    clustered_count = (clustered_df["cluster_id"] != -1).sum()
    print(f"  - Observations clustered: {clustered_count}")
    print(f"  - Noise points identified: {noise_count}")

    # Stage 6: Summarize Cone Map Centroids
    print("\n[STAGE 6/8] Summarizing confidence-weighted cone centroids...")
    cone_map_df = summarize_clusters(clustered_df)
    print(f"  - Total cone clusters identified: {len(cone_map_df)}")

    # Stage 7: Run Observation-Level Evaluation
    print("\n[STAGE 7/8] Running observation-level evaluation...")
    metrics = evaluate_dbscan(clustered_df, cone_map_df)
    eval_csv_path = os.path.join(output_dir, "evaluation.csv")
    save_evaluation_metrics(metrics, eval_csv_path)

    # Stage 8: Generate Visualizations & Save Outputs
    print("\n[STAGE 8/8] Saving output CSVs and generating DBSCAN map plot...")
    save_cluster_outputs(clustered_df, cone_map_df, output_dir=output_dir)

    dbscan_map_path = os.path.join(output_dir, "dbscan_map.png")
    plot_dbscan_map(clustered_df, cone_map_df, dbscan_map_path)

    # Print Final Summary Table
    print("\n======================================================================")
    print("  FIXED-EPSILON DBSCAN BASELINE PIPELINE SUMMARY")
    print("======================================================================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<38}: {v:.6f}")
        else:
            print(f"  {k:<38}: {v}")
    print("----------------------------------------------------------------------")
    print(f"  Baseline Configuration                : eps={eps_m} m, min_samples={min_samples}")
    print(f"  Sensor Rotation                       : {rotation_deg} deg")
    print(f"  Transformed Observations              : {transformed_obs_path}")
    print(f"  DBSCAN Clustered Observations         : {os.path.join(output_dir, 'dbscan_observations.csv')}")
    print(f"  Summarized Cone Map                   : {os.path.join(output_dir, 'cone_map_dbscan.csv')}")
    print(f"  Evaluation Report                     : {eval_csv_path}")
    print(f"  DBSCAN Map Image                      : {dbscan_map_path}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
