"""
Standard Fixed-Epsilon DBSCAN Cone Mapper baseline module.

Clusters 2D global cone observations using Euclidean DBSCAN and computes
confidence-weighted centroid cone position estimates along with cluster metrics.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def run_dbscan(
    data: pd.DataFrame, eps_m: float = 1.0, min_samples: int = 4
) -> pd.DataFrame:
    """
    Run standard Euclidean DBSCAN clustering on global cone observations.

    Args:
        data (pd.DataFrame): DataFrame containing 'global_x' and 'global_y'.
        eps_m (float): Maximum neighborhood distance (epsilon) in meters (default 1.0).
        min_samples (int): Minimum number of points required to form a dense region (default 4).

    Returns:
        pd.DataFrame: A copy of input DataFrame augmented with 'cluster_id' column
                      (cluster_id = -1 indicates noise).
    """
    if "global_x" not in data.columns or "global_y" not in data.columns:
        raise ValueError("Input DataFrame must contain 'global_x' and 'global_y' columns.")

    df = data.copy()
    coords = df[["global_x", "global_y"]].values

    # Run Euclidean DBSCAN clustering
    dbscan = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
    cluster_labels = dbscan.fit_predict(coords)

    df["cluster_id"] = cluster_labels
    return df


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize clusters into a cone map with confidence-weighted centroids.

    Excludes noise observations (cluster_id == -1).

    For each valid cluster, computes:
      - cluster_id
      - global_x (confidence-weighted centroid)
      - global_y (confidence-weighted centroid)
      - observation_count
      - real_observations
      - ghost_observations
      - ghost_fraction
      - mean_confidence
      - first_timestamp
      - last_timestamp
      - unique_frames

    Args:
        clustered (pd.DataFrame): DataFrame output from run_dbscan with 'cluster_id'.

    Returns:
        pd.DataFrame: Summarized cone map DataFrame indexed by cluster_id.
    """
    if "cluster_id" not in clustered.columns:
        raise ValueError("Input DataFrame must contain 'cluster_id' column.")

    # Filter out noise observations
    valid_clusters = clustered[clustered["cluster_id"] != -1].copy()

    if valid_clusters.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "global_x",
                "global_y",
                "observation_count",
                "real_observations",
                "ghost_observations",
                "ghost_fraction",
                "mean_confidence",
                "first_timestamp",
                "last_timestamp",
                "unique_frames",
            ]
        )

    summary_rows = []

    # Group by cluster_id and calculate metrics for each cluster
    for cluster_id, group in valid_clusters.groupby("cluster_id"):
        obs_count = len(group)
        conf = group["confidence"].values
        conf_sum = conf.sum()

        # Confidence-weighted centroid computation
        if conf_sum > 0:
            x_centroid = (conf * group["global_x"].values).sum() / conf_sum
            y_centroid = (conf * group["global_y"].values).sum() / conf_sum
        else:
            x_centroid = group["global_x"].mean()
            y_centroid = group["global_y"].mean()

        # Label counts
        real_count = (group["label"] == "real").sum() if "label" in group.columns else 0
        ghost_count = (group["label"] == "ghost").sum() if "label" in group.columns else 0
        ghost_frac = ghost_count / obs_count if obs_count > 0 else 0.0

        mean_conf = group["confidence"].mean()
        first_ts = group["timestamp"].min()
        last_ts = group["timestamp"].max()
        unique_frames = group["timestamp"].nunique()

        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "global_x": float(x_centroid),
                "global_y": float(y_centroid),
                "observation_count": int(obs_count),
                "real_observations": int(real_count),
                "ghost_observations": int(ghost_count),
                "ghost_fraction": float(ghost_frac),
                "mean_confidence": float(mean_conf),
                "first_timestamp": float(first_ts),
                "last_timestamp": float(last_ts),
                "unique_frames": int(unique_frames),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def save_cluster_outputs(
    clustered_df: pd.DataFrame, cone_map_df: pd.DataFrame, output_dir: str = "outputs"
) -> None:
    """
    Save clustered observations and summarized cone map to CSV files.

    Files created:
      - {output_dir}/dbscan_observations.csv
      - {output_dir}/cone_map_dbscan.csv

    Args:
        clustered_df (pd.DataFrame): Clustered observations DataFrame with cluster_id.
        cone_map_df (pd.DataFrame): Summarized cone map DataFrame.
        output_dir (str): Destination directory path (default 'outputs').
    """
    os.makedirs(output_dir, exist_ok=True)

    obs_file = os.path.join(output_dir, "dbscan_observations.csv")
    map_file = os.path.join(output_dir, "cone_map_dbscan.csv")

    clustered_df.to_csv(obs_file, index=False)
    cone_map_df.to_csv(map_file, index=False)

    print(f"Saved clustered observations to: '{obs_file}'")
    print(f"Saved summarized cone map to:    '{map_file}'")
