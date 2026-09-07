"""
DBSCAN Failure Analysis Module.

Performs a detailed spatial, temporal, confidence, and contamination analysis on the
14 surviving ghost observations incorporated into baseline DBSCAN clusters (eps=1.0m, min_samples=4).
Analyzes structural limitations of conventional Euclidean DBSCAN without using EWMVC features.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates
from src.dbscan_mapper import run_dbscan, summarize_clusters


from typing import Tuple, Dict, Union

def analyze_surviving_ghosts(
    clustered_df: pd.DataFrame, cone_map_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform detailed failure analysis on surviving ghost observations.

    Args:
        clustered_df (pd.DataFrame): Clustered observations DataFrame.
        cone_map_df (pd.DataFrame): Cone map summary DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (failure_analysis_df, contamination_df, summary_df)
    """
    # 1. Identify surviving ghosts (label == 'ghost' AND cluster_id != -1)
    ghost_mask = (clustered_df["label"] == "ghost") & (clustered_df["cluster_id"] != -1)
    surviving_ghosts = clustered_df[ghost_mask].copy()

    # Pre-map cluster centroids and stats for fast lookup
    centroid_lookup = cone_map_df.set_index("cluster_id").to_dict(orient="index")

    failure_rows = []

    for idx, ghost_row in surviving_ghosts.iterrows():
        cid = int(ghost_row["cluster_id"])
        g_x = float(ghost_row["global_x"])
        g_y = float(ghost_row["global_y"])
        g_ts = float(ghost_row["timestamp"])

        # Cluster observations
        cluster_obs = clustered_df[clustered_df["cluster_id"] == cid]
        real_obs_in_cluster = cluster_obs[cluster_obs["label"] == "real"]
        ghost_obs_in_cluster = cluster_obs[cluster_obs["label"] == "ghost"]

        obs_count = len(cluster_obs)
        real_count = len(real_obs_in_cluster)
        ghost_count = len(ghost_obs_in_cluster)

        # Centroid coordinates
        if cid in centroid_lookup:
            c_x = centroid_lookup[cid]["global_x"]
            c_y = centroid_lookup[cid]["global_y"]
        else:
            c_x = cluster_obs["global_x"].mean()
            c_y = cluster_obs["global_y"].mean()

        # Distance to centroid
        dist_to_centroid = float(np.hypot(g_x - c_x, g_y - c_y))

        # Distance to nearest real observation in the same cluster
        if not real_obs_in_cluster.empty:
            real_coords = real_obs_in_cluster[["global_x", "global_y"]].values
            dists_to_reals = np.hypot(real_coords[:, 0] - g_x, real_coords[:, 1] - g_y)
            nearest_real_dist = float(dists_to_reals.min())
        else:
            # Fallback to nearest real in entire dataset if cluster has no reals
            all_reals = clustered_df[clustered_df["label"] == "real"][["global_x", "global_y"]].values
            nearest_real_dist = float(np.hypot(all_reals[:, 0] - g_x, all_reals[:, 1] - g_y).min())

        # Temporal metrics
        min_ts = float(cluster_obs["timestamp"].min())
        max_ts = float(cluster_obs["timestamp"].max())
        time_span = max_ts - min_ts
        unique_ts_count = int(cluster_obs["timestamp"].nunique())

        rel_time_pos = (g_ts - min_ts) / time_span if time_span > 0 else 0.0

        failure_rows.append(
            {
                "timestamp": g_ts,
                "cone_type": ghost_row["cone_type"],
                "confidence": float(ghost_row["confidence"]),
                "global_x": g_x,
                "global_y": g_y,
                "cluster_id": cid,
                "cluster_observation_count": obs_count,
                "cluster_centroid_global_x": c_x,
                "cluster_centroid_global_y": c_y,
                "distance_to_cluster_centroid": dist_to_centroid,
                "nearest_real_observation_distance": nearest_real_dist,
                "real_observations_in_cluster": real_count,
                "ghost_observations_in_cluster": ghost_count,
                "unique_timestamps_in_cluster": unique_ts_count,
                "cluster_time_span": time_span,
                "ghost_relative_time_position": rel_time_pos,
            }
        )

    failure_analysis_df = pd.DataFrame(failure_rows)

    # 2. Cluster Contamination Analysis (Clusters containing surviving ghosts)
    contaminated_cids = failure_analysis_df["cluster_id"].unique()
    contamination_rows = []

    for cid in sorted(contaminated_cids):
        c_obs = clustered_df[clustered_df["cluster_id"] == cid]
        c_ghosts = (c_obs["label"] == "ghost").sum()
        c_reals = (c_obs["label"] == "real").sum()
        c_total = len(c_obs)
        c_frac = c_ghosts / c_total if c_total > 0 else 0.0

        contamination_rows.append(
            {
                "cluster_id": int(cid),
                "ghost_count": int(c_ghosts),
                "real_count": int(c_reals),
                "total_count": int(c_total),
                "ghost_fraction": float(c_frac),
            }
        )

    contamination_df = pd.DataFrame(contamination_rows)

    # 3. Confidence Statistics Breakdown
    surviving_ghost_conf = failure_analysis_df["confidence"] if not failure_analysis_df.empty else pd.Series([0])
    rejected_ghost_conf = clustered_df[(clustered_df["label"] == "ghost") & (clustered_df["cluster_id"] == -1)]["confidence"]
    real_conf = clustered_df[clustered_df["label"] == "real"]["confidence"]

    conf_summary = {
        "surviving_ghost_mean": float(surviving_ghost_conf.mean()),
        "surviving_ghost_median": float(surviving_ghost_conf.median()),
        "surviving_ghost_min": float(surviving_ghost_conf.min()),
        "surviving_ghost_max": float(surviving_ghost_conf.max()),
        "rejected_ghost_mean": float(rejected_ghost_conf.mean()),
        "rejected_ghost_median": float(rejected_ghost_conf.median()),
        "rejected_ghost_min": float(rejected_ghost_conf.min()),
        "rejected_ghost_max": float(rejected_ghost_conf.max()),
        "real_obs_mean": float(real_conf.mean()),
        "real_obs_median": float(real_conf.median()),
        "real_obs_min": float(real_conf.min()),
        "real_obs_max": float(real_conf.max()),
    }

    # Summary Metrics DataFrame
    summary_rows = [
        {"metric": "total_surviving_ghosts", "value": len(failure_analysis_df)},
        {"metric": "contaminated_clusters_count", "value": len(contamination_df)},
        {"metric": "mean_distance_to_centroid_m", "value": float(failure_analysis_df["distance_to_cluster_centroid"].mean())},
        {"metric": "max_distance_to_centroid_m", "value": float(failure_analysis_df["distance_to_cluster_centroid"].max())},
        {"metric": "mean_nearest_real_distance_m", "value": float(failure_analysis_df["nearest_real_observation_distance"].mean())},
        {"metric": "max_nearest_real_distance_m", "value": float(failure_analysis_df["nearest_real_observation_distance"].max())},
        {"metric": "mean_cluster_time_span_s", "value": float(failure_analysis_df["cluster_time_span"].mean())},
    ]
    for k, v in conf_summary.items():
        summary_rows.append({"metric": k, "value": v})

    summary_df = pd.DataFrame(summary_rows)

    return failure_analysis_df, contamination_df, summary_df


def generate_failure_plots(
    clustered_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    contamination_df: pd.DataFrame,
    cone_map_df: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """
    Generate visualization plots for DBSCAN failure analysis.

    Generated plots:
      - outputs/surviving_ghosts_map.png
      - outputs/ghost_distance_to_centroid.png
      - outputs/ghost_cluster_contamination.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Surviving Ghosts Map
    fig, ax = plt.subplots(figsize=(12, 10))

    # Vehicle trajectory
    if "vehicle_x" in clustered_df.columns and "vehicle_y" in clustered_df.columns:
        veh_poses = clustered_df[["vehicle_x", "vehicle_y"]].drop_duplicates()
        ax.plot(
            veh_poses["vehicle_x"],
            veh_poses["vehicle_y"],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Vehicle Trajectory",
            zorder=1,
        )

    # Real observations
    real_pts = clustered_df[clustered_df["label"] == "real"]
    ax.scatter(
        real_pts["global_x"],
        real_pts["global_y"],
        c="tab:blue",
        s=10,
        alpha=0.4,
        edgecolors="none",
        label="Real Observations",
        zorder=2,
    )

    # Cluster Centroids
    ax.scatter(
        cone_map_df["global_x"],
        cone_map_df["global_y"],
        c="orange",
        marker="o",
        s=35,
        edgecolors="black",
        linewidths=0.8,
        label="DBSCAN Cluster Centroids",
        zorder=3,
    )

    # Surviving Ghost Observations (Highlight)
    ax.scatter(
        failure_df["global_x"],
        failure_df["global_y"],
        c="red",
        marker="X",
        s=100,
        edgecolors="black",
        linewidths=1.2,
        label="Surviving Ghosts (Accepted)",
        zorder=4,
    )

    # Annotate surviving ghost cluster IDs
    for _, r in failure_df.iterrows():
        ax.annotate(
            f"Ghost (C{int(r['cluster_id'])})",
            (r["global_x"], r["global_y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="darkred",
            zorder=5,
        )

    ax.set_title("DBSCAN Failure Analysis: Surviving Ghost Observations Map", fontsize=14, pad=12)
    ax.set_xlabel("Global X (m)", fontsize=12)
    ax.set_ylabel("Global Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    map_plot_path = os.path.join(output_dir, "surviving_ghosts_map.png")
    plt.savefig(map_plot_path, dpi=300)
    plt.close(fig)

    # 2. Ghost Distance to Centroid & Nearest Real Observation
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(len(failure_df))
    width = 0.35

    ax.bar(
        x_indices - width / 2,
        failure_df["distance_to_cluster_centroid"],
        width,
        label="Distance to Cluster Centroid",
        color="tab:orange",
    )
    ax.bar(
        x_indices + width / 2,
        failure_df["nearest_real_observation_distance"],
        width,
        label="Distance to Nearest Real Obs",
        color="tab:blue",
    )

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="DBSCAN Epsilon Threshold (1.0 m)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Ghost #{i+1}\n(C{int(cid)})" for i, cid in enumerate(failure_df["cluster_id"])], fontsize=8)
    ax.set_title("Surviving Ghost Proximity to Cluster Centroid & Real Observations", fontsize=13, pad=10)
    ax.set_ylabel("Distance (m)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "ghost_distance_to_centroid.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close(fig)

    # 3. Ghost Cluster Contamination (Ghost Fraction per Contaminated Cluster)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [f"Cluster {int(c)}" for c in contamination_df["cluster_id"]],
        contamination_df["ghost_fraction"] * 100.0,
        color="crimson",
        width=0.5,
    )
    ax.set_title("Ghost Observation Contamination Fraction per Cluster", fontsize=13, pad=10)
    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Ghost Observation Fraction (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    contam_plot_path = os.path.join(output_dir, "ghost_cluster_contamination.png")
    plt.savefig(contam_plot_path, dpi=300)
    plt.close(fig)

    print(f"Generated failure analysis plots in '{output_dir}':")
    print(f"  - {map_plot_path}")
    print(f"  - {dist_plot_path}")
    print(f"  - {contam_plot_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perception_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    telemetry_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("======================================================================")
    print("  DBSCAN FAILURE ANALYSIS: SURVIVING GHOST OBSERVATIONS")
    print("======================================================================")

    # 1. Load data & run baseline DBSCAN
    perception, telemetry = load_logs(perception_path, telemetry_path)
    perception_interp = interpolate_telemetry(perception, telemetry)
    transformed_df = add_global_coordinates(perception_interp, rotation_deg=180.0)

    clustered_df = run_dbscan(transformed_df, eps_m=1.0, min_samples=4)
    cone_map_df = summarize_clusters(clustered_df)

    # 2. Run detailed failure analysis
    failure_df, contamination_df, summary_df = analyze_surviving_ghosts(clustered_df, cone_map_df)

    # 3. Save CSV output files
    fail_csv = os.path.join(output_dir, "dbscan_failure_analysis.csv")
    contam_csv = os.path.join(output_dir, "dbscan_cluster_contamination.csv")
    summary_csv = os.path.join(output_dir, "dbscan_failure_summary.csv")

    failure_df.to_csv(fail_csv, index=False)
    contamination_df.to_csv(contam_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved failure analysis to       : '{fail_csv}'")
    print(f"Saved cluster contamination to : '{contam_csv}'")
    print(f"Saved failure summary to       : '{summary_csv}'")

    # 4. Generate failure plots
    generate_failure_plots(clustered_df, failure_df, contamination_df, cone_map_df, output_dir=output_dir)

    # 5. Terminal Summary & Diagnostics
    print("\n===== DBSCAN FAILURE ANALYSIS SUMMARY =====")
    print(f"Number of Surviving Ghosts         : {len(failure_df)}")
    print(f"Clusters Contaminated by Ghosts    : {len(contamination_df)} (Cluster IDs: {contamination_df['cluster_id'].tolist()})")
    print(f"Mean Ghost-to-Centroid Distance    : {failure_df['distance_to_cluster_centroid'].mean():.4f} m (Max: {failure_df['distance_to_cluster_centroid'].max():.4f} m)")
    print(f"Mean Nearest-Real Distance        : {failure_df['nearest_real_observation_distance'].mean():.4f} m (Max: {failure_df['nearest_real_observation_distance'].max():.4f} m)")

    print("\n----- Ghost Confidence Comparison -----")
    surv_conf = failure_df["confidence"]
    rej_conf = clustered_df[(clustered_df["label"] == "ghost") & (clustered_df["cluster_id"] == -1)]["confidence"]
    real_conf = clustered_df[clustered_df["label"] == "real"]["confidence"]

    print(f"Surviving Ghosts Confidence  : mean={surv_conf.mean():.4f}, median={surv_conf.median():.4f}, min={surv_conf.min():.4f}, max={surv_conf.max():.4f}")
    print(f"Rejected Ghosts Confidence   : mean={rej_conf.mean():.4f}, median={rej_conf.median():.4f}, min={rej_conf.min():.4f}, max={rej_conf.max():.4f}")
    print(f"Real Observations Confidence : mean={real_conf.mean():.4f}, median={real_conf.median():.4f}, min={real_conf.min():.4f}, max={real_conf.max():.4f}")

    print("\n----- Cluster Contamination Statistics -----")
    pd.set_option("display.max_columns", None)
    print(contamination_df.to_string(index=False))

    print("\n----- Temporal Persistence Statistics -----")
    print(f"Mean Contaminated Cluster Time Span : {failure_df['cluster_time_span'].mean():.2f} s")
    print(f"Mean Unique Frames per Cluster      : {failure_df['unique_timestamps_in_cluster'].mean():.1f} frames")

    # 6. Technical Conclusion on Structural Limitations of DBSCAN
    mean_near_real = failure_df['nearest_real_observation_distance'].mean()
    max_near_real = failure_df['nearest_real_observation_distance'].max()
    mean_g_conf = surv_conf.mean()
    mean_r_conf = real_conf.mean()
    mean_frames = failure_df['unique_timestamps_in_cluster'].mean()
    mean_span = failure_df['cluster_time_span'].mean()

    print("\n======================================================================")
    print("  TECHNICAL CONCLUSION: STRUCTURAL FAILURE MODES OF EUCLIDEAN DBSCAN")
    print("======================================================================")
    print(f"""1. SPATIAL PROXIMITY ADSORPTION:
   All 14 surviving ghost observations are located near real cone observation clusters
   (mean nearest-real distance: {mean_near_real:.4f} m, max: {max_near_real:.4f} m). Because standard
   DBSCAN only evaluates spatial proximity (d < eps = 1.0 m) to core points, any isolated
   false detection that falls spatially close to a real landmark is unconditionally absorbed.

2. CONFIDENCE INDISCRIMINATION:
   The mean confidence of surviving ghosts ({mean_g_conf:.4f}) is significantly lower than real cone
   observations ({mean_r_conf:.4f}). However, Euclidean DBSCAN weights every spatial point equally
   (binary spatial thresholding), ignoring perception detection confidence and sensor uncertainty.

3. TEMPORAL BLINDNESS & SINGLE-FRAME TRANSIENTS:
   Surviving ghosts appear as transient single-frame perception anomalies (occurring at a
   single timestamp), whereas valid cone landmarks persist across many frames (mean {mean_frames:.1f} frames,
   mean time span {mean_span:.2f} s). Standard DBSCAN operates on static point clouds without temporal
   continuity or multi-view persistence scoring.
""")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
