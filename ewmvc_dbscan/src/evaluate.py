"""
Observation-Level DBSCAN Evaluation & Diagnostic Module.

IMPORTANT NOTICE:
The dataset contains observation-level labels ('real' vs 'ghost') but does NOT contain
unique ground-truth cone IDs, ground-truth cone coordinates, or a ground-truth landmark map.
Therefore, these metrics measure observation-level DBSCAN clustering behavior and ghost noise
filtering diagnostics. They are NOT landmark detection precision, recall, or F1 scores.
Do NOT fabricate landmark F1 scores.
"""

import os
from typing import Dict, Union
import pandas as pd


def evaluate_dbscan(
    clustered: pd.DataFrame, cone_map: pd.DataFrame
) -> Dict[str, Union[int, float]]:
    """
    Compute observation-level clustering diagnostics and ghost observation statistics.

    Args:
        clustered (pd.DataFrame): DataFrame containing 'cluster_id' and 'label'.
        cone_map (pd.DataFrame): Summarized cone map DataFrame with 'cluster_id' and 'observation_count'.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing all 11 diagnostic metrics.
    """
    if "cluster_id" not in clustered.columns:
        raise ValueError("Input DataFrame 'clustered' must contain 'cluster_id' column.")

    total_observations = len(clustered)
    noise_mask = clustered["cluster_id"] == -1
    clustered_mask = ~noise_mask

    clustered_obs = int(clustered_mask.sum())
    noise_obs = int(noise_mask.sum())
    noise_frac = noise_obs / total_observations if total_observations > 0 else 0.0

    # Ghost observation metrics
    if "label" in clustered.columns:
        ghost_mask = clustered["label"] == "ghost"
        total_ghost = int(ghost_mask.sum())
        ghost_inside_clusters = int((ghost_mask & clustered_mask).sum())
        ghost_rejected_noise = int((ghost_mask & noise_mask).sum())
        ghost_rejection_rate = (
            ghost_rejected_noise / total_ghost if total_ghost > 0 else 0.0
        )
    else:
        total_ghost = 0
        ghost_inside_clusters = 0
        ghost_rejected_noise = 0
        ghost_rejection_rate = 0.0

    # Cluster metrics
    num_clusters = len(cone_map)
    if num_clusters > 0 and "observation_count" in cone_map.columns:
        largest_cluster_size = int(cone_map["observation_count"].max())
        average_cluster_size = float(cone_map["observation_count"].mean())
    elif not clustered[clustered_mask].empty:
        counts = clustered[clustered_mask]["cluster_id"].value_counts()
        largest_cluster_size = int(counts.max())
        average_cluster_size = float(counts.mean())
    else:
        largest_cluster_size = 0
        average_cluster_size = 0.0

    metrics = {
        "total_observations": total_observations,
        "clustered_observations": clustered_obs,
        "noise_observations": noise_obs,
        "noise_fraction": noise_frac,
        "total_ghost_observations": total_ghost,
        "ghost_observations_inside_clusters": ghost_inside_clusters,
        "ghost_observations_rejected_as_noise": ghost_rejected_noise,
        "ghost_rejection_rate_by_noise": ghost_rejection_rate,
        "number_of_clusters": num_clusters,
        "largest_cluster_size": largest_cluster_size,
        "average_cluster_size": average_cluster_size,
    }

    return metrics


def save_evaluation_metrics(
    metrics: Dict[str, Union[int, float]], output_path: str
) -> pd.DataFrame:
    """
    Save evaluation metrics to CSV file.

    Args:
        metrics (Dict[str, Union[int, float]]): Metrics dictionary.
        output_path (str): File path to save evaluation.csv.

    Returns:
        pd.DataFrame: DataFrame representation of evaluation metrics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_metrics = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    df_metrics.to_csv(output_path, index=False)
    print(f"Evaluation metrics saved to: '{output_path}'")
    return df_metrics
