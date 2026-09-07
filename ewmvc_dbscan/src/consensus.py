"""
EWMVC Multi-View Consensus Module.

Combines four multi-view evidence features (Entropy, Geometry, Persistence, Diversity)
into a normalized candidate reliability score:

    C = w_H * H + w_G * G + w_D * D + w_P * P

where:
- H: Viewpoint Entropy in [0.0, 1.0]
- G: Geometry Consistency Score in (0.0, 1.0]
- P: Temporal Persistence Score in [0.0, 1.0]
- D: Viewpoint Diversity Score in [0.0, 1.0]

Weights and Acceptance Threshold (from config.yaml):
- w_H = 0.30, w_G = 0.30, w_D = 0.20, w_P = 0.20 (sum = 1.0)
- acceptance_threshold = 0.60 (Diagnostic threshold for accepted candidate components)

NOTE ON DESIGN PARAMETERS:
Weights and threshold are PROPOSED DESIGN PARAMETERS, NOT empirically optimized against ground truth labels.
Accepted candidate components represent 'consensus-supported candidate landmarks', not confirmed ground truth cones.
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_weights(weights: Dict[str, float], tol: float = 1e-6) -> None:
    """
    Validate that consensus feature weights are non-negative and sum to 1.0 within tolerance.

    Args:
        weights: Dictionary mapping feature names to floats.
        tol: Floating point tolerance.

    Raises:
        ValueError: If any weight is negative or if sum of weights != 1.0.
    """
    for name, w in weights.items():
        if w < 0.0:
            raise ValueError(f"Consensus weight '{name}' cannot be negative (got {w}).")

    total_w = sum(weights.values())
    if abs(total_w - 1.0) > tol:
        raise ValueError(f"Consensus weights must sum to 1.0 (got sum = {total_w:.6f}).")


def validate_feature_values(
    h: Union[float, np.ndarray],
    g: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
    tol: float = 1e-6,
) -> None:
    """
    Validate feature values lie within their expected mathematical bounds.

    - 0.0 - tol <= H <= 1.0 + tol
    - 0.0 < G <= 1.0 + tol
    - 0.0 - tol <= P <= 1.0 + tol
    - 0.0 - tol <= D <= 1.0 + tol

    Raises:
        ValueError: If any feature value is invalid or out of bounds.
    """
    h_arr = np.asarray(h, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    d_arr = np.asarray(d, dtype=float)

    if np.isnan(h_arr).any() or np.isnan(g_arr).any() or np.isnan(p_arr).any() or np.isnan(d_arr).any():
        raise ValueError("Feature arrays contain NaN values.")

    if (h_arr < -tol).any() or (h_arr > 1.0 + tol).any():
        raise ValueError(f"Viewpoint entropy H contains values outside [0, 1]: min={h_arr.min()}, max={h_arr.max()}")

    if (g_arr <= 0.0).any() or (g_arr > 1.0 + tol).any():
        raise ValueError(f"Geometry score G contains values outside (0, 1]: min={g_arr.min()}, max={g_arr.max()}")

    if (p_arr < -tol).any() or (p_arr > 1.0 + tol).any():
        raise ValueError(f"Persistence score P contains values outside [0, 1]: min={p_arr.min()}, max={p_arr.max()}")

    if (d_arr < -tol).any() or (d_arr > 1.0 + tol).any():
        raise ValueError(f"Diversity score D contains values outside [0, 1]: min={d_arr.min()}, max={d_arr.max()}")


def compute_consensus_score(
    entropy_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    acceptance_threshold: float = 0.60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine multi-view evidence features into candidate consensus scores.

    Args:
        entropy_df: DataFrame containing 'candidate_id', 'viewpoint_entropy', etc.
        geometry_df: DataFrame containing 'candidate_id', 'geometry_score', etc.
        persistence_df: DataFrame containing 'candidate_id', 'persistence_score', etc.
        diversity_df: DataFrame containing 'candidate_id', 'viewpoint_diversity', etc.
        weights: Dictionary of feature weights (default: entropy=0.3, geometry=0.3, diversity=0.2, persistence=0.2).
        acceptance_threshold: Threshold for accepted flag (default: 0.60).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - consensus_df: Joined table with features, contributions, consensus score, accepted flag.
            - corr_df: Feature Pearson correlation matrix DataFrame.
            - ablation_df: Diagnostic leave-one-feature-out ablation summary DataFrame.
    """
    if weights is None:
        weights = {
            "entropy": 0.30,
            "geometry": 0.30,
            "diversity": 0.20,
            "persistence": 0.20,
        }

    validate_weights(weights)

    # 1. Validation & Merging
    dfs = [entropy_df, geometry_df, persistence_df, diversity_df]
    for idx, df in enumerate(dfs):
        if "candidate_id" not in df.columns:
            raise ValueError(f"Feature DataFrame at index {idx} missing 'candidate_id' column.")
        if df["candidate_id"].duplicated().any():
            dups = df["candidate_id"][df["candidate_id"].duplicated()].tolist()
            raise ValueError(f"Duplicate candidate IDs found in feature table {idx}: {dups}")

    merged = (
        entropy_df
        .merge(geometry_df, on="candidate_id", suffixes=("", "_geo"))
        .merge(persistence_df, on="candidate_id", suffixes=("", "_pers"))
        .merge(diversity_df, on="candidate_id", suffixes=("", "_div"))
    )

    expected_count = len(entropy_df)
    if len(merged) != expected_count:
        raise ValueError(
            f"Candidate count mismatch after joining feature tables: expected {expected_count}, got {len(merged)}"
        )

    # Extract features
    h = merged["viewpoint_entropy"].values
    g = merged["geometry_score"].values
    p = merged["persistence_score"].values
    d = merged["viewpoint_diversity"].values

    validate_feature_values(h, g, p, d)

    w_h = float(weights["entropy"])
    w_g = float(weights["geometry"])
    w_d = float(weights["diversity"])
    w_p = float(weights["persistence"])

    # Compute feature contributions
    entropy_contrib = w_h * h
    geometry_contrib = w_g * g
    diversity_contrib = w_d * d
    persistence_contrib = w_p * p

    consensus_score = entropy_contrib + geometry_contrib + diversity_contrib + persistence_contrib
    consensus_score = np.clip(consensus_score, 0.0, 1.0)
    accepted = consensus_score >= acceptance_threshold

    # Build output table
    res_df = pd.DataFrame({
        "candidate_id": merged["candidate_id"],
        "num_observations": merged.get("num_observations", merged.get("observation_count")),
        "unique_viewpoint_bins": merged.get("unique_viewpoint_bins", 1),
        "sigma_spatial_m": merged.get("sigma_spatial_m", 0.0),
        "duration_s": merged.get("duration_s", 0.0),
        "max_baseline_m": merged.get("max_baseline_m", 0.0),
        "viewpoint_entropy": h,
        "geometry_score": g,
        "persistence_score": p,
        "viewpoint_diversity": d,
        "entropy_contribution": entropy_contrib,
        "geometry_contribution": geometry_contrib,
        "diversity_contribution": diversity_contrib,
        "persistence_contribution": persistence_contrib,
        "consensus_score": consensus_score,
        "accepted": accepted,
    })

    # 2. Feature Correlation Matrix
    feature_df = res_df[["viewpoint_entropy", "geometry_score", "persistence_score", "viewpoint_diversity"]].rename(
        columns={
            "viewpoint_entropy": "entropy",
            "geometry_score": "geometry",
            "persistence_score": "persistence",
            "viewpoint_diversity": "diversity",
        }
    )
    corr_df = feature_df.corr(method="pearson")

    # 3. Ablation Diagnostics (Leave-one-out)
    ablation_results = []
    features_list = ["entropy", "geometry", "persistence", "diversity"]

    for feat in features_list:
        remaining_weights = {k: v for k, v in weights.items() if k != feat}
        sum_rem = sum(remaining_weights.values())
        norm_weights = {k: v / sum_rem for k, v in remaining_weights.items()}

        ablated_score = (
            norm_weights.get("entropy", 0.0) * h
            + norm_weights.get("geometry", 0.0) * g
            + norm_weights.get("diversity", 0.0) * d
            + norm_weights.get("persistence", 0.0) * p
        )
        ablated_score = np.clip(ablated_score, 0.0, 1.0)
        ablated_accepted = ablated_score >= acceptance_threshold

        ablation_results.append({
            "ablated_feature": feat,
            "remaining_features": ", ".join(list(remaining_weights.keys())),
            "mean_consensus_score": float(np.mean(ablated_score)),
            "median_consensus_score": float(np.median(ablated_score)),
            "min_consensus_score": float(np.min(ablated_score)),
            "max_consensus_score": float(np.max(ablated_score)),
            "accepted_count": int(np.sum(ablated_accepted)),
            "accepted_percentage": float(np.mean(ablated_accepted) * 100.0),
        })

    ablation_df = pd.DataFrame(ablation_results)

    return res_df, corr_df, ablation_df


def plot_consensus_distribution(
    consensus_df: pd.DataFrame,
    threshold: float,
    output_path: str,
) -> None:
    """
    Plot histogram of candidate consensus scores with diagnostic threshold marker.

    Args:
        consensus_df: DataFrame containing 'consensus_score'.
        threshold: Diagnostic acceptance threshold.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        consensus_df["consensus_score"],
        bins=20,
        range=(0.0, 1.0),
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.8,
    )

    ax.axvline(
        threshold,
        color="#d90429",
        linestyle="--",
        linewidth=2.0,
        label=f"Acceptance Threshold ({threshold:.2f})",
    )

    ax.set_title("EWMVC Multi-View Consensus Score Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Consensus Score [0, 1]", fontsize=12)
    ax.set_ylabel("Number of Candidate Components", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_consensus_vs_feature(
    consensus_df: pd.DataFrame,
    feature_col: str,
    feature_title: str,
    color: str,
    output_path: str,
) -> None:
    """
    Plot scatter plot of feature score vs consensus score.

    Args:
        consensus_df: DataFrame containing feature_col and 'consensus_score'.
        feature_col: Column name of feature.
        feature_title: Display name for title/label.
        color: Scatter marker hex color.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        consensus_df[feature_col],
        consensus_df["consensus_score"],
        c=color,
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title(f"EWMVC Consensus Score vs {feature_title}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(feature_title, fontsize=12)
    ax.set_ylabel("Consensus Score [0, 1]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot heatmap of feature Pearson correlation matrix.

    Args:
        corr_df: Correlation matrix DataFrame.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    cax = ax.matshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)

    cols = list(corr_df.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(cols, fontsize=10)

    # Annotate correlation values inside cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=11, fontweight="bold")

    ax.set_title("EWMVC Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
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

    con_cfg = config.get("ewmvc", {}).get("consensus", {})
    weights = con_cfg.get("weights", {
        "entropy": 0.30,
        "geometry": 0.30,
        "diversity": 0.20,
        "persistence": 0.20,
    })
    threshold = float(con_cfg.get("acceptance_threshold", 0.60))

    # Input CSV paths
    f_entropy = os.path.join(out_dir, "ewmvc_candidate_entropy.csv")
    f_geometry = os.path.join(out_dir, "ewmvc_candidate_geometry.csv")
    f_persistence = os.path.join(out_dir, "ewmvc_candidate_persistence.csv")
    f_diversity = os.path.join(out_dir, "ewmvc_candidate_diversity.csv")

    for f_path in [f_entropy, f_geometry, f_persistence, f_diversity]:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Required feature CSV '{f_path}' not found. Run feature steps 20C-20F first.")

    df_entropy = pd.read_csv(f_entropy)
    df_geometry = pd.read_csv(f_geometry)
    df_persistence = pd.read_csv(f_persistence)
    df_diversity = pd.read_csv(f_diversity)

    print("======================================================================")
    print("  EWMVC STEP 20G: MULTI-VIEW CONSENSUS SCORING")
    print("======================================================================")
    print(f"Loaded candidate entropy          : {len(df_entropy)} candidates")
    print(f"Loaded candidate geometry         : {len(df_geometry)} candidates")
    print(f"Loaded candidate persistence      : {len(df_persistence)} candidates")
    print(f"Loaded candidate diversity        : {len(df_diversity)} candidates")
    print(f"Configured Weights                : {weights}")
    print(f"Acceptance Threshold              : {threshold} (Diagnostic Parameter)")

    # 1. Compute Consensus Scores & Correlation Matrix
    consensus_df, corr_df, ablation_df = compute_consensus_score(
        df_entropy,
        df_geometry,
        df_persistence,
        df_diversity,
        weights=weights,
        acceptance_threshold=threshold,
    )

    # 2. Save Output CSVs
    out_consensus_csv = os.path.join(out_dir, "ewmvc_consensus.csv")
    consensus_df.to_csv(out_consensus_csv, index=False)
    print(f"Saved Consensus CSV               : '{out_consensus_csv}'")

    out_corr_csv = os.path.join(out_dir, "ewmvc_feature_correlation.csv")
    corr_df.to_csv(out_corr_csv)
    print(f"Saved Correlation CSV             : '{out_corr_csv}'")

    out_ablation_csv = os.path.join(out_dir, "ewmvc_consensus_ablation.csv")
    ablation_df.to_csv(out_ablation_csv, index=False)
    print(f"Saved Ablation CSV                : '{out_ablation_csv}'")

    # 3. Generate Diagnostic Plots
    out_hist_png = os.path.join(out_dir, "ewmvc_consensus_distribution.png")
    plot_consensus_distribution(consensus_df, threshold, out_hist_png)
    print(f"Saved Consensus Distribution PNG  : '{out_hist_png}'")

    plot_consensus_vs_feature(consensus_df, "viewpoint_entropy", "Viewpoint Entropy", "#2b5c8f", os.path.join(out_dir, "ewmvc_consensus_vs_entropy.png"))
    plot_consensus_vs_feature(consensus_df, "geometry_score", "Geometry Score", "#2a9d8f", os.path.join(out_dir, "ewmvc_consensus_vs_geometry.png"))
    plot_consensus_vs_feature(consensus_df, "persistence_score", "Persistence Score", "#3a86c8", os.path.join(out_dir, "ewmvc_consensus_vs_persistence.png"))
    plot_consensus_vs_feature(consensus_df, "viewpoint_diversity", "Viewpoint Diversity", "#8338ec", os.path.join(out_dir, "ewmvc_consensus_vs_diversity.png"))
    print("Saved Consensus vs Feature Scatter PNGs")

    out_corr_png = os.path.join(out_dir, "ewmvc_feature_correlation.png")
    plot_feature_correlation_heatmap(corr_df, out_corr_png)
    print(f"Saved Correlation Heatmap PNG     : '{out_corr_png}'")

    # Summary Statistics Report
    scores = consensus_df["consensus_score"]
    accepted_mask = consensus_df["accepted"]
    n_accepted = int(accepted_mask.sum())
    n_rejected = int((~accepted_mask).sum())
    pct_accepted = (n_accepted / len(consensus_df)) * 100.0
    pct_rejected = (n_rejected / len(consensus_df)) * 100.0

    print("\n--- Consensus Score Measured Results ---")
    print(f"Total Candidate Components          : {len(consensus_df)}")
    print(f"Minimum Consensus Score             : {scores.min():.6f}")
    print(f"Maximum Consensus Score             : {scores.max():.6f}")
    print(f"Mean Consensus Score                : {scores.mean():.6f}")
    print(f"Median Consensus Score              : {scores.median():.6f}")

    print(f"\n--- Diagnostic Acceptance (Threshold = {threshold:.2f}) ---")
    print(f"Accepted Candidates                 : {n_accepted} ({pct_accepted:.2f}%)")
    print(f"Rejected Candidates                 : {n_rejected} ({pct_rejected:.2f}%)")

    print("\n--- Pearson Feature Correlation Matrix ---")
    print(corr_df.to_string())

    print(f"\nCorrelation(Entropy, Diversity)    : {corr_df.loc['entropy', 'diversity']:.6f}")
    print(f"Correlation(Persistence, Diversity): {corr_df.loc['persistence', 'diversity']:.6f}")

    print("\n--- Ablation Diagnostics (Leave-One-Out) ---")
    print(ablation_df.to_string(index=False))
    print("======================================================================\n")


if __name__ == "__main__":
    main()
