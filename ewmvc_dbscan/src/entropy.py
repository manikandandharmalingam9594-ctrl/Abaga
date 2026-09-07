"""
EWMVC Viewpoint Entropy Module.

Calculates the normalized viewpoint entropy for each candidate landmark component.
Viewpoint entropy measures the angular distribution of observation viewpoints (vehicle poses)
relative to a candidate landmark's provisional centroid.

Mathematical Formulations:
1. Provisional Centroid: (x_c, y_c) = (mean(global_x), mean(global_y)) for candidate component k.
2. Viewpoint Angle: theta_view = atan2(y_c - y_v, x_c - x_v), normalized to [-pi, pi).
3. 8 Angular Bins: 45 deg per bin (2*pi / 8).
4. Shannon Entropy: H = -sum(p_b * ln(p_b)) for occupied bins (p_b > 0).
5. Normalized Entropy: H_norm = H / ln(8) in [0.0, 1.0].
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_viewpoint_angle(
    vehicle_x: Union[float, np.ndarray, pd.Series],
    vehicle_y: Union[float, np.ndarray, pd.Series],
    candidate_x: float,
    candidate_y: float,
) -> np.ndarray:
    """
    Calculate viewpoint direction angle from vehicle position to provisional candidate position.

    Args:
        vehicle_x: Vehicle global X position(s).
        vehicle_y: Vehicle global Y position(s).
        candidate_x: Provisional candidate landmark X centroid.
        candidate_y: Provisional candidate landmark Y centroid.

    Returns:
        np.ndarray: Viewpoint angles in radians, normalized to [-pi, pi).
    """
    vx = np.asarray(vehicle_x, dtype=float)
    vy = np.asarray(vehicle_y, dtype=float)

    dx = candidate_x - vx
    dy = candidate_y - vy

    angles = np.arctan2(dy, dx)
    # Wrap to [-pi, pi)
    angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
    return angles


def assign_viewpoint_bin(
    angles: np.ndarray,
    num_bins: int = 8,
) -> np.ndarray:
    """
    Assign viewpoint angles in [-pi, pi) to discrete uniform angular bins.

    Args:
        angles: Array of viewpoint angles in radians.
        num_bins: Number of angular bins (default: 8).

    Returns:
        np.ndarray: Bin indices in range [0, num_bins - 1].
    """
    angles = np.asarray(angles, dtype=float)
    # Map [-pi, pi) -> [0, 2*pi) -> [0, num_bins)
    shifted = (angles + np.pi) % (2.0 * np.pi)
    bin_width = (2.0 * np.pi) / float(num_bins)
    bins = np.floor(shifted / bin_width).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def calculate_normalized_entropy(
    bin_counts: np.ndarray,
    num_bins: int = 8,
) -> Tuple[float, int]:
    """
    Calculate Shannon entropy normalized by ln(num_bins).

    Args:
        bin_counts: Array of counts per viewpoint bin.
        num_bins: Total number of angular bins (default: 8).

    Returns:
        Tuple[float, int]: (normalized_entropy in [0.0, 1.0], unique_occupied_bins_count).
    """
    counts = np.asarray(bin_counts, dtype=float)
    total = np.sum(counts)

    if total == 0:
        return 0.0, 0

    p_b = counts / total
    occupied = p_b[p_b > 0]
    unique_bins = len(occupied)

    if unique_bins <= 1:
        return 0.0, unique_bins

    h_shannon = -np.sum(occupied * np.log(occupied))
    max_h = np.log(float(num_bins))
    h_norm = float(h_shannon / max_h)

    # Numerical clamping to [0.0, 1.0]
    h_norm = float(np.clip(h_norm, 0.0, 1.0))
    return h_norm, unique_bins


def compute_viewpoint_entropy(
    candidate_df: pd.DataFrame,
    num_bins: int = 8,
) -> pd.DataFrame:
    """
    Compute normalized viewpoint entropy for all candidate components.

    Args:
        candidate_df (pd.DataFrame): Observation DataFrame containing
            'candidate_id', 'global_x', 'global_y', 'vehicle_x', 'vehicle_y'.
        num_bins (int): Number of angular bins (default: 8).

    Returns:
        pd.DataFrame: Table with per-candidate metrics:
            - candidate_id
            - candidate_x
            - candidate_y
            - num_observations
            - unique_viewpoint_bins
            - viewpoint_entropy
            - bin_0 .. bin_{num_bins-1}
    """
    required = ["candidate_id", "global_x", "global_y", "vehicle_x", "vehicle_y"]
    missing = [c for c in required if c not in candidate_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    results = []
    grouped = candidate_df.groupby("candidate_id", sort=True)

    for cand_id, group in grouped:
        n_obs = len(group)
        cand_x = float(group["global_x"].mean())
        cand_y = float(group["global_y"].mean())

        angles = compute_viewpoint_angle(
            group["vehicle_x"].values,
            group["vehicle_y"].values,
            cand_x,
            cand_y,
        )

        bins = assign_viewpoint_bin(angles, num_bins=num_bins)
        bin_counts = np.bincount(bins, minlength=num_bins)

        h_norm, unique_bins = calculate_normalized_entropy(bin_counts, num_bins=num_bins)

        row = {
            "candidate_id": cand_id,
            "candidate_x": cand_x,
            "candidate_y": cand_y,
            "num_observations": int(n_obs),
            "unique_viewpoint_bins": int(unique_bins),
            "viewpoint_entropy": h_norm,
        }
        for b in range(num_bins):
            row[f"bin_{b}"] = int(bin_counts[b])

        results.append(row)

    res_df = pd.DataFrame(results)
    return res_df


def plot_entropy_distribution(
    entropy_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot histogram of candidate viewpoint entropy distribution.

    Args:
        entropy_df: DataFrame containing 'viewpoint_entropy'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        entropy_df["viewpoint_entropy"],
        bins=20,
        range=(0.0, 1.0),
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_title("EWMVC Viewpoint Entropy Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Normalized Viewpoint Entropy [0, 1]", fontsize=12)
    ax.set_ylabel("Number of Candidate Components", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_entropy_vs_component_size(
    entropy_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot scatter plot of viewpoint entropy vs component size (number of observations).

    Args:
        entropy_df: DataFrame containing 'num_observations' and 'viewpoint_entropy'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        entropy_df["num_observations"],
        entropy_df["viewpoint_entropy"],
        c="#e05638",
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title("EWMVC Viewpoint Entropy vs Component Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Component Size (Number of Observations)", fontsize=12)
    ax.set_ylabel("Normalized Viewpoint Entropy [0, 1]", fontsize=12)
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

    entropy_cfg = config.get("ewmvc", {}).get("entropy", {})
    num_bins = int(entropy_cfg.get("num_angular_bins", 8))

    in_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"Input candidate CSV file '{in_csv}' not found. Run Step 20B (src/graph_builder.py) first."
        )

    cand_df = pd.read_csv(in_csv)

    print("======================================================================")
    print("  EWMVC STEP 20C: VIEWPOINT ENTROPY CALCULATION")
    print("======================================================================")
    print(f"Loaded candidate observations     : {len(cand_df)}")
    print(f"Unique candidate component IDs    : {cand_df['candidate_id'].nunique()}")
    print(f"Angular bins count                : {num_bins} (45 deg / bin)")

    # 1. Calculate Viewpoint Entropy
    entropy_df = compute_viewpoint_entropy(cand_df, num_bins=num_bins)

    # 2. Save candidate entropy CSV
    out_entropy_csv = os.path.join(out_dir, "ewmvc_candidate_entropy.csv")
    entropy_df.to_csv(out_entropy_csv, index=False)
    print(f"Saved Candidate Entropy CSV       : '{out_entropy_csv}'")

    # 3. Generate diagnostic plots
    out_hist_png = os.path.join(out_dir, "ewmvc_entropy_distribution.png")
    plot_entropy_distribution(entropy_df, out_hist_png)
    print(f"Saved Entropy Distribution PNG   : '{out_hist_png}'")

    out_scatter_png = os.path.join(out_dir, "ewmvc_entropy_vs_component_size.png")
    plot_entropy_vs_component_size(entropy_df, out_scatter_png)
    print(f"Saved Entropy vs Size Plot PNG   : '{out_scatter_png}'")

    # Summary Statistics Report
    entropies = entropy_df["viewpoint_entropy"]
    bin_dist = entropy_df["unique_viewpoint_bins"].value_counts().sort_index()

    print("\n--- Viewpoint Entropy Measured Results ---")
    print(f"Total Candidate Components        : {len(entropy_df)}")
    print(f"Minimum Viewpoint Entropy         : {entropies.min():.6f}")
    print(f"Maximum Viewpoint Entropy         : {entropies.max():.6f}")
    print(f"Mean Viewpoint Entropy            : {entropies.mean():.6f}")
    print(f"Median Viewpoint Entropy          : {entropies.median():.6f}")
    print(f"Components with Entropy approx 0  : {int((entropies < 1e-6).sum())}")
    print(f"Components with Entropy > 0.5     : {int((entropies > 0.5).sum())}")

    print("\n--- Unique Viewpoint Bins Distribution ---")
    for b in range(1, num_bins + 1):
        count = int(bin_dist.get(b, 0))
        print(f"  {b} bin(s)  : {count}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
