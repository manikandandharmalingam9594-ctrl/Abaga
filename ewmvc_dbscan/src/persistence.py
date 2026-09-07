"""
EWMVC Temporal Persistence Module.

Calculates temporal persistence features and scores for candidate landmark components.

Mathematical Formulations:
1. Time Span: duration_s = max(timestamp) - min(timestamp) for component k.
2. Unique Timestamps: num_unique_timestamps = count of unique timestamps in component k.
3. Observation Count: total observations in component k.
4. Persistence Score: persistence_score = min(1.0, duration_s / expected_duration_s).

NOTE ON DESIGN PARAMETER:
expected_duration_s (default 10.0 s) is a PROPOSED CONFIGURATION PARAMETER, not an empirically tuned optimum.
For singletons or observations occurring at a single timestamp, duration_s = 0 => persistence_score = 0.0.
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_persistence_score(
    duration_s: float,
    expected_duration_s: float = 10.0,
) -> float:
    """
    Calculate temporal persistence score = min(1.0, duration_s / expected_duration_s).

    Args:
        duration_s: Time duration in seconds (t_max - t_min).
        expected_duration_s: Target expected duration scale in seconds (default: 10.0).

    Returns:
        float: Persistence score in range [0.0, 1.0].
    """
    if expected_duration_s <= 0:
        raise ValueError("Expected duration scale must be strictly positive.")

    if duration_s <= 0:
        return 0.0

    score = float(min(1.0, duration_s / float(expected_duration_s)))
    return float(np.clip(score, 0.0, 1.0))


def compute_persistence_features(
    candidate_df: pd.DataFrame,
    expected_duration_s: float = 10.0,
) -> pd.DataFrame:
    """
    Compute temporal persistence features for all candidate components.

    Args:
        candidate_df: DataFrame containing 'candidate_id', 'timestamp'.
        expected_duration_s: Target expected duration scale in seconds (default: 10.0).

    Returns:
        pd.DataFrame: DataFrame containing per-candidate persistence metrics:
            - candidate_id
            - observation_count
            - num_unique_timestamps
            - t_min
            - t_max
            - duration_s
            - persistence_score
    """
    required = ["candidate_id", "timestamp"]
    missing = [c for c in required if c not in candidate_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    results = []
    grouped = candidate_df.groupby("candidate_id", sort=True)

    for cand_id, group in grouped:
        n_obs = len(group)
        timestamps = group["timestamp"].values

        t_min = float(np.min(timestamps))
        t_max = float(np.max(timestamps))
        duration_s = float(t_max - t_min)
        num_unique_t = int(len(np.unique(timestamps)))

        score = calculate_persistence_score(duration_s, expected_duration_s=expected_duration_s)

        results.append({
            "candidate_id": cand_id,
            "observation_count": int(n_obs),
            "num_unique_timestamps": num_unique_t,
            "t_min": t_min,
            "t_max": t_max,
            "duration_s": duration_s,
            "persistence_score": score,
        })

    res_df = pd.DataFrame(results)
    return res_df


def plot_persistence_distribution(
    persistence_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot histogram of candidate temporal persistence score distribution.

    Args:
        persistence_df: DataFrame containing 'persistence_score'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        persistence_df["persistence_score"],
        bins=20,
        range=(0.0, 1.0),
        color="#3a86c8",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_title("EWMVC Persistence Score Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Temporal Persistence Score [0, 1]", fontsize=12)
    ax.set_ylabel("Number of Candidate Components", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_persistence_vs_component_size(
    persistence_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot scatter plot of persistence score vs component size (observation count).

    Args:
        persistence_df: DataFrame containing 'observation_count' and 'persistence_score'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        persistence_df["observation_count"],
        persistence_df["persistence_score"],
        c="#f4a261",
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title("EWMVC Persistence Score vs Component Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Component Size (Observation Count)", fontsize=12)
    ax.set_ylabel("Temporal Persistence Score [0, 1]", fontsize=12)
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

    pers_cfg = config.get("ewmvc", {}).get("persistence", {})
    expected_dur = float(pers_cfg.get("expected_duration_s", 10.0))

    in_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"Input candidate CSV file '{in_csv}' not found. Run Step 20B (src/graph_builder.py) first."
        )

    cand_df = pd.read_csv(in_csv)

    print("======================================================================")
    print("  EWMVC STEP 20E: TEMPORAL PERSISTENCE CALCULATION")
    print("======================================================================")
    print(f"Loaded candidate observations       : {len(cand_df)}")
    print(f"Unique candidate component IDs      : {cand_df['candidate_id'].nunique()}")
    print(f"Expected duration scale parameter   : {expected_dur} s (Configuration Parameter)")

    # 1. Calculate Persistence Features
    persistence_df = compute_persistence_features(cand_df, expected_duration_s=expected_dur)

    # 2. Save candidate persistence CSV
    out_persistence_csv = os.path.join(out_dir, "ewmvc_candidate_persistence.csv")
    persistence_df.to_csv(out_persistence_csv, index=False)
    print(f"Saved Candidate Persistence CSV     : '{out_persistence_csv}'")

    # 3. Generate diagnostic plots
    out_hist_png = os.path.join(out_dir, "ewmvc_persistence_distribution.png")
    plot_persistence_distribution(persistence_df, out_hist_png)
    print(f"Saved Persistence Distribution PNG  : '{out_hist_png}'")

    out_scatter_png = os.path.join(out_dir, "ewmvc_persistence_vs_component_size.png")
    plot_persistence_vs_component_size(persistence_df, out_scatter_png)
    print(f"Saved Persistence vs Size Plot PNG  : '{out_scatter_png}'")

    # Summary Statistics Report
    durations = persistence_df["duration_s"]
    scores = persistence_df["persistence_score"]
    unique_t = persistence_df["num_unique_timestamps"]

    print("\n--- Time Span (duration_s) Measured Results ---")
    print(f"Total Candidate Components          : {len(persistence_df)}")
    print(f"Minimum Duration                    : {durations.min():.6f} s")
    print(f"Maximum Duration                    : {durations.max():.6f} s")
    print(f"Mean Duration                       : {durations.mean():.6f} s")
    print(f"Median Duration                     : {durations.median():.6f} s")

    print("\n--- Persistence Score Measured Results ---")
    print(f"Minimum Persistence Score           : {scores.min():.6f}")
    print(f"Maximum Persistence Score           : {scores.max():.6f}")
    print(f"Mean Persistence Score              : {scores.mean():.6f}")
    print(f"Median Persistence Score            : {scores.median():.6f}")

    print("\n--- Score Extrema Counts ---")
    print(f"Candidates with Persistence Score == 0 : {int((scores == 0.0).sum())}")
    print(f"Candidates with Persistence Score > 0.5 : {int((scores > 0.5).sum())}")
    print(f"Candidates with Persistence Score >= 1.0: {int((scores >= 1.0).sum())}")

    print("\n--- Unique Timestamps Distribution ---")
    print(f"Candidates with 1 unique timestamp  : {int((unique_t == 1).sum())}")
    print(f"Candidates with >= 2 unique timestamps: {int((unique_t >= 2).sum())}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
