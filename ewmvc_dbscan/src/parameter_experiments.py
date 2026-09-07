"""
DBSCAN Parameter Sensitivity Experiment Module.

Tests 24 parameter combinations (eps: 0.5 to 2.0, min_samples: 3 to 6)
on global cone observations to evaluate clustering behavior, noise fraction,
and ghost rejection diagnostics. Saves parameter experiment CSVs, heatmaps,
line plots, and baseline surviving ghost failure analysis data.
"""

import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates
from src.dbscan_mapper import run_dbscan, summarize_clusters
from src.evaluate import evaluate_dbscan


def run_parameter_sensitivity_experiment(
    transformed_df: pd.DataFrame,
    eps_values: List[float] = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    min_samples_values: List[int] = [3, 4, 5, 6],
    baseline_eps: float = 1.00,
    baseline_min_samples: int = 4,
) -> pd.DataFrame:
    """
    Run DBSCAN independently across all parameter combinations and collect diagnostics.

    Args:
        transformed_df (pd.DataFrame): Transformed global perception DataFrame.
        eps_values (List[float]): List of epsilon values to evaluate.
        min_samples_values (List[int]): List of min_samples values to evaluate.
        baseline_eps (float): Baseline epsilon value (1.0).
        baseline_min_samples (int): Baseline min_samples value (4).

    Returns:
        pd.DataFrame: Sorted DataFrame of 24 experiment configurations with diagnostics.
    """
    experiment_results = []

    for eps in sorted(eps_values):
        for min_samples in sorted(min_samples_values):
            # Run DBSCAN independently for this configuration
            clustered = run_dbscan(transformed_df, eps_m=eps, min_samples=min_samples)
            cone_map = summarize_clusters(clustered)
            metrics = evaluate_dbscan(clustered, cone_map)

            # Mark baseline configuration
            is_baseline = (abs(eps - baseline_eps) < 1e-5) and (min_samples == baseline_min_samples)

            row = {
                "eps": float(eps),
                "min_samples": int(min_samples),
                "is_baseline": bool(is_baseline),
                "total_observations": int(metrics["total_observations"]),
                "clustered_observations": int(metrics["clustered_observations"]),
                "noise_observations": int(metrics["noise_observations"]),
                "noise_fraction": float(metrics["noise_fraction"]),
                "total_ghost_observations": int(metrics["total_ghost_observations"]),
                "ghost_observations_inside_clusters": int(metrics["ghost_observations_inside_clusters"]),
                "ghost_observations_rejected_as_noise": int(metrics["ghost_observations_rejected_as_noise"]),
                "ghost_rejection_rate_by_noise": float(metrics["ghost_rejection_rate_by_noise"]),
                "number_of_clusters": int(metrics["number_of_clusters"]),
                "largest_cluster_size": int(metrics["largest_cluster_size"]),
                "average_cluster_size": float(metrics["average_cluster_size"]),
            }
            experiment_results.append(row)

    df_experiments = pd.DataFrame(experiment_results)
    df_experiments = df_experiments.sort_values(by=["eps", "min_samples"]).reset_index(drop=True)
    return df_experiments


def save_surviving_ghosts(
    transformed_df: pd.DataFrame,
    baseline_eps: float = 1.0,
    baseline_min_samples: int = 4,
    output_path: str = "outputs/baseline_surviving_ghosts.csv",
) -> pd.DataFrame:
    """
    Save ghost observations that survived inside clusters under the baseline DBSCAN setup.

    Args:
        transformed_df (pd.DataFrame): Transformed global perception DataFrame.
        baseline_eps (float): Baseline epsilon (1.0).
        baseline_min_samples (int): Baseline min_samples (4).
        output_path (str): File path to save baseline_surviving_ghosts.csv.

    Returns:
        pd.DataFrame: DataFrame of surviving ghost observations.
    """
    clustered = run_dbscan(transformed_df, eps_m=baseline_eps, min_samples=baseline_min_samples)
    
    # Filter surviving ghosts: label == 'ghost' and cluster_id != -1
    ghost_mask = (clustered["label"] == "ghost") & (clustered["cluster_id"] != -1)
    surviving_ghosts = clustered[ghost_mask].copy()

    columns_to_save = [
        "timestamp",
        "cone_type",
        "rel_x_sensor",
        "rel_y_sensor",
        "confidence",
        "global_x",
        "global_y",
        "cluster_id",
    ]

    surviving_ghosts_out = surviving_ghosts[columns_to_save]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    surviving_ghosts_out.to_csv(output_path, index=False)
    print(f"Saved baseline surviving ghosts ({len(surviving_ghosts_out)} rows) to: '{output_path}'")
    return surviving_ghosts_out


def generate_experiment_plots(df_exp: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Generate diagnostic line plots and heatmaps for DBSCAN parameter sensitivity.

    Generated files:
      - ghost_rejection_vs_eps.png
      - number_of_clusters_vs_eps.png
      - noise_fraction_vs_eps.png
      - dbscan_parameter_heatmap.png
    """
    os.makedirs(output_dir, exist_ok=True)

    min_samples_list = sorted(df_exp["min_samples"].unique())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D"]

    # 1. Ghost Rejection Rate vs. Eps
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, ms in enumerate(min_samples_list):
        sub = df_exp[df_exp["min_samples"] == ms]
        ax.plot(
            sub["eps"],
            sub["ghost_rejection_rate_by_noise"] * 100.0,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"min_samples = {ms}",
        )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (eps=1.0)")
    ax.set_title("Ghost Observation Rejection Rate vs. Epsilon", fontsize=13, pad=10)
    ax.set_xlabel("Epsilon / eps (m)", fontsize=11)
    ax.set_ylabel("Ghost Rejection Rate (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "ghost_rejection_vs_eps.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close(fig)

    # 2. Number of Clusters vs. Eps
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, ms in enumerate(min_samples_list):
        sub = df_exp[df_exp["min_samples"] == ms]
        ax.plot(
            sub["eps"],
            sub["number_of_clusters"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"min_samples = {ms}",
        )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (eps=1.0)")
    ax.set_title("Number of DBSCAN Clusters vs. Epsilon", fontsize=13, pad=10)
    ax.set_xlabel("Epsilon / eps (m)", fontsize=11)
    ax.set_ylabel("Number of Clusters", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "number_of_clusters_vs_eps.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig)

    # 3. Noise Fraction vs. Eps
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, ms in enumerate(min_samples_list):
        sub = df_exp[df_exp["min_samples"] == ms]
        ax.plot(
            sub["eps"],
            sub["noise_fraction"] * 100.0,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"min_samples = {ms}",
        )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (eps=1.0)")
    ax.set_title("Noise Observation Fraction vs. Epsilon", fontsize=13, pad=10)
    ax.set_xlabel("Epsilon / eps (m)", fontsize=11)
    ax.set_ylabel("Noise Fraction (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "noise_fraction_vs_eps.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close(fig)

    # 4. Parameter Heatmaps (Ghost Rejection Rate & Number of Clusters)
    eps_unique = sorted(df_exp["eps"].unique())
    ms_unique = sorted(df_exp["min_samples"].unique())

    grid_rejection = np.zeros((len(ms_unique), len(eps_unique)))
    grid_clusters = np.zeros((len(ms_unique), len(eps_unique)))

    for i, ms in enumerate(ms_unique):
        for j, ep in enumerate(eps_unique):
            sub = df_exp[(df_exp["min_samples"] == ms) & (np.abs(df_exp["eps"] - ep) < 1e-5)]
            if not sub.empty:
                grid_rejection[i, j] = sub["ghost_rejection_rate_by_noise"].values[0] * 100.0
                grid_clusters[i, j] = sub["number_of_clusters"].values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Heatmap 1: Ghost Rejection Rate (%)
    im1 = ax1.imshow(grid_rejection, cmap="Blues", aspect="auto")
    ax1.set_xticks(np.arange(len(eps_unique)))
    ax1.set_yticks(np.arange(len(ms_unique)))
    ax1.set_xticklabels([f"{e:.2f}" for e in eps_unique])
    ax1.set_yticklabels(ms_unique)
    ax1.set_xlabel("Epsilon / eps (m)", fontsize=11)
    ax1.set_ylabel("min_samples", fontsize=11)
    ax1.set_title("Ghost Rejection Rate (%)", fontsize=12, pad=10)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    for i in range(len(ms_unique)):
        for j in range(len(eps_unique)):
            val = grid_rejection[i, j]
            color = "white" if val > grid_rejection.max() * 0.7 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    # Heatmap 2: Number of Clusters
    im2 = ax2.imshow(grid_clusters, cmap="YlGnBu", aspect="auto")
    ax2.set_xticks(np.arange(len(eps_unique)))
    ax2.set_yticks(np.arange(len(ms_unique)))
    ax2.set_xticklabels([f"{e:.2f}" for e in eps_unique])
    ax2.set_yticklabels(ms_unique)
    ax2.set_xlabel("Epsilon / eps (m)", fontsize=11)
    ax2.set_ylabel("min_samples", fontsize=11)
    ax2.set_title("Number of Clusters", fontsize=12, pad=10)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    for i in range(len(ms_unique)):
        for j in range(len(eps_unique)):
            val = int(grid_clusters[i, j])
            color = "white" if val > grid_clusters.max() * 0.7 else "black"
            ax2.text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "dbscan_parameter_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close(fig)

    print(f"Generated sensitivity plots in '{output_dir}':")
    print(f"  - {plot1_path}")
    print(f"  - {plot2_path}")
    print(f"  - {plot3_path}")
    print(f"  - {heatmap_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perception_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    telemetry_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("======================================================================")
    print("  DBSCAN PARAMETER SENSITIVITY EXPERIMENT (24 CONFIGURATIONS)")
    print("======================================================================")

    # 1. Load and transform dataset
    perception, telemetry = load_logs(perception_path, telemetry_path)
    perception_interp = interpolate_telemetry(perception, telemetry)
    transformed_df = add_global_coordinates(perception_interp, rotation_deg=180.0)

    # 2. Run sensitivity grid (6 eps x 4 min_samples = 24 configs)
    df_exp = run_parameter_sensitivity_experiment(transformed_df)

    # 3. Save results to outputs/dbscan_parameter_experiments.csv
    exp_csv_path = os.path.join(output_dir, "dbscan_parameter_experiments.csv")
    df_exp.to_csv(exp_csv_path, index=False)
    print(f"\nSaved experiment results to: '{exp_csv_path}'")

    # 4. Save baseline surviving ghosts for failure analysis
    save_surviving_ghosts(transformed_df, baseline_eps=1.0, baseline_min_samples=4,
                          output_path=os.path.join(output_dir, "baseline_surviving_ghosts.csv"))

    # 5. Generate plots
    generate_experiment_plots(df_exp, output_dir=output_dir)

    # 6. Terminal Display: All 24 Configurations
    print("\n===== DBSCAN PARAMETER EXPERIMENT (ALL 24 CONFIGURATIONS) =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display_cols = [
        "eps",
        "min_samples",
        "is_baseline",
        "number_of_clusters",
        "noise_observations",
        "noise_fraction",
        "ghost_observations_inside_clusters",
        "ghost_observations_rejected_as_noise",
        "ghost_rejection_rate_by_noise",
        "average_cluster_size",
    ]
    print(df_exp[display_cols].to_string(index=False))

    # 7. Terminal Display: Current Baseline Configuration (eps=1.0, min_samples=4)
    baseline_row = df_exp[df_exp["is_baseline"]].iloc[0]
    print("\n===== CURRENT BASELINE =====")
    print("  eps = 1.0 m, min_samples = 4")
    print(f"  Total Observations                  : {baseline_row['total_observations']}")
    print(f"  Clustered Observations              : {baseline_row['clustered_observations']}")
    print(f"  Noise Observations                  : {baseline_row['noise_observations']}")
    print(f"  Noise Fraction                      : {baseline_row['noise_fraction']:.6f} ({baseline_row['noise_fraction']*100:.2f}%)")
    print(f"  Total Ghost Observations            : {baseline_row['total_ghost_observations']}")
    print(f"  Ghosts Inside Clusters              : {baseline_row['ghost_observations_inside_clusters']}")
    print(f"  Ghosts Rejected as Noise            : {baseline_row['ghost_observations_rejected_as_noise']}")
    print(f"  Ghost Rejection Rate                : {baseline_row['ghost_rejection_rate_by_noise']:.6f} ({baseline_row['ghost_rejection_rate_by_noise']*100:.2f}%)")
    print(f"  Number of Clusters                  : {baseline_row['number_of_clusters']}")
    print(f"  Largest Cluster Size                : {baseline_row['largest_cluster_size']}")
    print(f"  Average Cluster Size                : {baseline_row['average_cluster_size']:.2f}")

    # 8. Identify Extrema Configurations
    max_ghost_rej = df_exp.loc[df_exp["ghost_rejection_rate_by_noise"].idxmax()]
    min_ghost_rej = df_exp.loc[df_exp["ghost_rejection_rate_by_noise"].idxmin()]
    max_clusters = df_exp.loc[df_exp["number_of_clusters"].idxmax()]
    min_clusters = df_exp.loc[df_exp["number_of_clusters"].idxmin()]
    max_noise_frac = df_exp.loc[df_exp["noise_fraction"].idxmax()]
    min_noise_frac = df_exp.loc[df_exp["noise_fraction"].idxmin()]

    print("\n======================================================================")
    print("  PARAMETER EXTREMA ANALYSIS")
    print("======================================================================")
    print(f"Highest Ghost Rejection Rate : {max_ghost_rej['ghost_rejection_rate_by_noise']*100:.2f}% (eps={max_ghost_rej['eps']}, min_samples={max_ghost_rej['min_samples']})")
    print(f"Lowest Ghost Rejection Rate  : {min_ghost_rej['ghost_rejection_rate_by_noise']*100:.2f}% (eps={min_ghost_rej['eps']}, min_samples={min_ghost_rej['min_samples']})")
    print(f"Largest Number of Clusters   : {int(max_clusters['number_of_clusters'])} (eps={max_clusters['eps']}, min_samples={max_clusters['min_samples']})")
    print(f"Smallest Number of Clusters  : {int(min_clusters['number_of_clusters'])} (eps={min_clusters['eps']}, min_samples={min_clusters['min_samples']})")
    print(f"Highest Noise Fraction       : {max_noise_frac['noise_fraction']*100:.2f}% (eps={max_noise_frac['eps']}, min_samples={max_noise_frac['min_samples']})")
    print(f"Lowest Noise Fraction        : {min_noise_frac['noise_fraction']*100:.2f}% (eps={min_noise_frac['eps']}, min_samples={min_noise_frac['min_samples']})")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
