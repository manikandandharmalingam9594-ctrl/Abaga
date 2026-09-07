"""
EWMVC Viewpoint Diversity Module.

Calculates vehicle displacement baseline features and viewpoint-diversity scores for candidate landmark components.

Mathematical Formulations:
1. Unique Viewpoints: Deduplicate vehicle positions (vehicle_x, vehicle_y) per timestamp frame.
   Multiple observations occurring at the same timestamp belong to the SAME vehicle pose.
2. Pairwise Vehicle Displacement Baseline: d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2) for unique vehicle poses i, j.
3. Max Pairwise Baseline: max_baseline_m = max(d_ij) over all unique viewpoint pairs. (0 for 1 viewpoint).
4. Pairwise Diagnostic Baselines: mean_pairwise_baseline_m, median_pairwise_baseline_m, min_pairwise_baseline_m.
5. Viewpoint Diversity Score: viewpoint_diversity = min(1.0, max_baseline_m / saturation_baseline_m) in [0.0, 1.0].

NOTE ON DESIGN PARAMETER:
saturation_baseline_m (default 5.0 m) is a PROPOSED CONFIGURATION PARAMETER, not an empirically tuned optimum.
For singletons or observations from a single vehicle pose, max_baseline_m = 0 => viewpoint_diversity = 0.0.
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_diversity_score(
    max_baseline_m: float,
    saturation_baseline_m: float = 5.0,
) -> float:
    """
    Calculate viewpoint diversity score = min(1.0, max_baseline_m / saturation_baseline_m).

    Args:
        max_baseline_m: Maximum pairwise vehicle displacement baseline in meters.
        saturation_baseline_m: Target saturation displacement scale in meters (default: 5.0).

    Returns:
        float: Diversity score in range [0.0, 1.0].
    """
    if saturation_baseline_m <= 0:
        raise ValueError("Saturation baseline scale must be strictly positive.")

    if max_baseline_m <= 0:
        return 0.0

    score = float(min(1.0, max_baseline_m / float(saturation_baseline_m)))
    return float(np.clip(score, 0.0, 1.0))


def compute_pairwise_baselines(
    unique_vx: np.ndarray,
    unique_vy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute max, mean, median, and min pairwise vehicle displacement baselines.

    Args:
        unique_vx: Array of unique vehicle X positions.
        unique_vy: Array of unique vehicle Y positions.

    Returns:
        Tuple[float, float, float, float]:
            - max_baseline_m
            - mean_pairwise_baseline_m
            - median_pairwise_baseline_m
            - min_pairwise_baseline_m
    """
    unique_vx = np.asarray(unique_vx, dtype=float)
    unique_vy = np.asarray(unique_vy, dtype=float)

    n_viewpoints = len(unique_vx)
    if n_viewpoints <= 1:
        return 0.0, 0.0, 0.0, 0.0

    # Compute pairwise Euclidean distances between unique vehicle positions
    dx = unique_vx[:, None] - unique_vx[None, :]
    dy = unique_vy[:, None] - unique_vy[None, :]
    dist_matrix = np.sqrt(dx ** 2 + dy ** 2)

    # Extract upper triangle (i < j)
    triu_indices = np.triu_indices(n_viewpoints, k=1)
    pairwise_distances = dist_matrix[triu_indices]

    if len(pairwise_distances) == 0:
        return 0.0, 0.0, 0.0, 0.0

    max_b = float(np.max(pairwise_distances))
    mean_b = float(np.mean(pairwise_distances))
    median_b = float(np.median(pairwise_distances))
    min_b = float(np.min(pairwise_distances))

    return max_b, mean_b, median_b, min_b


def compute_diversity_features(
    candidate_df: pd.DataFrame,
    saturation_baseline_m: float = 5.0,
) -> pd.DataFrame:
    """
    Compute viewpoint diversity features for all candidate components.

    Args:
        candidate_df: DataFrame containing 'candidate_id', 'timestamp', 'vehicle_x', 'vehicle_y'.
        saturation_baseline_m: Target saturation displacement scale in meters (default: 5.0).

    Returns:
        pd.DataFrame: DataFrame containing per-candidate diversity metrics:
            - candidate_id
            - observation_count
            - num_unique_timestamps
            - max_baseline_m
            - mean_pairwise_baseline_m
            - median_pairwise_baseline_m
            - min_pairwise_baseline_m
            - viewpoint_diversity
    """
    required = ["candidate_id", "timestamp", "vehicle_x", "vehicle_y"]
    missing = [c for c in required if c not in candidate_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    results = []
    grouped = candidate_df.groupby("candidate_id", sort=True)

    for cand_id, group in grouped:
        n_obs = len(group)

        # Deduplicate vehicle positions per timestamp frame
        unique_frames = group.drop_duplicates(subset=["timestamp"])
        num_unique_t = len(unique_frames)

        uv_x = unique_frames["vehicle_x"].values
        uv_y = unique_frames["vehicle_y"].values

        max_b, mean_b, median_b, min_b = compute_pairwise_baselines(uv_x, uv_y)
        score = calculate_diversity_score(max_b, saturation_baseline_m=saturation_baseline_m)

        results.append({
            "candidate_id": cand_id,
            "observation_count": int(n_obs),
            "num_unique_timestamps": int(num_unique_t),
            "max_baseline_m": max_b,
            "mean_pairwise_baseline_m": mean_b,
            "median_pairwise_baseline_m": median_b,
            "min_pairwise_baseline_m": min_b,
            "viewpoint_diversity": score,
        })

    res_df = pd.DataFrame(results)
    return res_df


def plot_diversity_distribution(
    diversity_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot histogram of candidate viewpoint diversity score distribution.

    Args:
        diversity_df: DataFrame containing 'viewpoint_diversity'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        diversity_df["viewpoint_diversity"],
        bins=20,
        range=(0.0, 1.0),
        color="#8338ec",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_title("EWMVC Viewpoint Diversity Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Viewpoint Diversity Score [0, 1]", fontsize=12)
    ax.set_ylabel("Number of Candidate Components", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_diversity_vs_component_size(
    diversity_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot scatter plot of diversity score vs component size (observation count).

    Args:
        diversity_df: DataFrame containing 'observation_count' and 'viewpoint_diversity'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        diversity_df["observation_count"],
        diversity_df["viewpoint_diversity"],
        c="#3a86c8",
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title("EWMVC Viewpoint Diversity vs Component Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Component Size (Observation Count)", fontsize=12)
    ax.set_ylabel("Viewpoint Diversity Score [0, 1]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_diversity_vs_entropy(
    diversity_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot scatter plot of viewpoint diversity vs viewpoint entropy.

    Args:
        diversity_df: DataFrame containing 'candidate_id' and 'viewpoint_diversity'.
        entropy_df: DataFrame containing 'candidate_id' and 'viewpoint_entropy'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    merged = pd.merge(diversity_df, entropy_df, on="candidate_id")

    ax.scatter(
        merged["viewpoint_entropy"],
        merged["viewpoint_diversity"],
        c="#ff006e",
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title("EWMVC Viewpoint Diversity vs Viewpoint Entropy", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Viewpoint Entropy [0, 1]", fontsize=12)
    ax.set_ylabel("Viewpoint Diversity Score [0, 1]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir_rel = config["output"]["directory"]
    out_dir = os.path.join(base_dir, out_dir_rel)

    div_cfg = config.get("ewmvc", {}).get("diversity", {})
    sat_baseline = float(div_cfg.get("saturation_baseline_m", 5.0))

    in_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"Input candidate CSV file '{in_csv}' not found. Run Step 20B (src/graph_builder.py) first."
        )

    cand_df = pd.read_csv(in_csv)

    print("======================================================================")
    print("  EWMVC STEP 20F: VIEWPOINT DIVERSITY CALCULATION")
    print("======================================================================")
    print(f"Loaded candidate observations         : {len(cand_df)}")
    print(f"Unique candidate component IDs        : {cand_df['candidate_id'].nunique()}")
    print(f"Saturation baseline scale parameter   : {sat_baseline} m (Configuration Parameter)")

    # 1. Calculate Viewpoint Diversity Features
    diversity_df = compute_diversity_features(cand_df, saturation_baseline_m=sat_baseline)

    # 2. Save candidate diversity CSV
    out_diversity_csv = os.path.join(out_dir, "ewmvc_candidate_diversity.csv")
    diversity_df.to_csv(out_diversity_csv, index=False)
    print(f"Saved Candidate Diversity CSV       : '{out_diversity_csv}'")

    # 3. Generate diagnostic plots
    out_hist_png = os.path.join(out_dir, "ewmvc_diversity_distribution.png")
    plot_diversity_distribution(diversity_df, out_hist_png)
    print(f"Saved Diversity Distribution PNG    : '{out_hist_png}'")

    out_scatter_size_png = os.path.join(out_dir, "ewmvc_diversity_vs_component_size.png")
    plot_diversity_vs_component_size(diversity_df, out_scatter_size_png)
    print(f"Saved Diversity vs Size Plot PNG    : '{out_scatter_size_png}'")

    entropy_csv = os.path.join(out_dir, "ewmvc_candidate_entropy.csv")
    if os.path.exists(entropy_csv):
        entropy_df = pd.read_csv(entropy_csv)
        out_scatter_ent_png = os.path.join(out_dir, "ewmvc_diversity_vs_entropy.png")
        plot_diversity_vs_entropy(diversity_df, entropy_df, out_scatter_ent_png)
        print(f"Saved Diversity vs Entropy Plot PNG : '{out_scatter_ent_png}'")

    # Summary Statistics Report
    max_baselines = diversity_df["max_baseline_m"]
    mean_baselines = diversity_df["mean_pairwise_baseline_m"]
    scores = diversity_df["viewpoint_diversity"]
    unique_t = diversity_df["num_unique_timestamps"]

    print("\n--- Max Baseline (max_baseline_m) Measured Results ---")
    print(f"Total Candidate Components          : {len(diversity_df)}")
    print(f"Minimum Max Baseline                : {max_baselines.min():.6f} m")
    print(f"Maximum Max Baseline                : {max_baselines.max():.6f} m")
    print(f"Mean Max Baseline                   : {max_baselines.mean():.6f} m")
    print(f"Median Max Baseline                 : {max_baselines.median():.6f} m")

    print("\n--- Mean Pairwise Baseline (mean_pairwise_baseline_m) Measured Results ---")
    print(f"Minimum Mean Pairwise Baseline      : {mean_baselines.min():.6f} m")
    print(f"Maximum Mean Pairwise Baseline      : {mean_baselines.max():.6f} m")
    print(f"Mean Mean Pairwise Baseline         : {mean_baselines.mean():.6f} m")
    print(f"Median Mean Pairwise Baseline       : {mean_baselines.median():.6f} m")

    print("\n--- Viewpoint Diversity Score Measured Results ---")
    print(f"Minimum Diversity Score             : {scores.min():.6f}")
    print(f"Maximum Diversity Score             : {scores.max():.6f}")
    print(f"Mean Diversity Score                : {scores.mean():.6f}")
    print(f"Median Diversity Score              : {scores.median():.6f}")

    print("\n--- Score Extrema Counts ---")
    print(f"Candidates with Diversity Score == 0 : {int((scores == 0.0).sum())}")
    print(f"Candidates with Diversity Score > 0.5 : {int((scores > 0.5).sum())}")
    print(f"Candidates with Diversity Score >= 1.0: {int((scores >= 1.0).sum())}")

    print("\n--- Unique Timestamps Distribution ---")
    print(f"Candidates with 1 unique timestamp  : {int((unique_t == 1).sum())}")
    print(f"Candidates with >= 2 unique timestamps: {int((unique_t >= 2).sum())}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
