"""
EWMVC Geometric Consistency Module.

Calculates spatial dispersion and geometry-consistency scores for candidate landmark components.

Mathematical Formulations:
1. Provisional Component Center: (mu_x, mu_y) = (mean(global_x), mean(global_y)) for candidate k.
2. Distance from Center: r_i = sqrt((x_i - mu_x)^2 + (y_i - mu_y)^2).
3. RMS Spatial Dispersion: sigma_spatial = sqrt(mean(r_i^2)).
4. Diagnostic Radii: median_radius = median(r_i), max_radius = max(r_i), p95_radius = percentile(r_i, 95).
5. Geometry Consistency Score: w_geo = exp(-sigma_spatial^2 / (2 * sigma0^2)) in (0, 1].

NOTE ON DESIGN PARAMETER:
sigma0_m (default 0.75 m) is a PROPOSED CONFIGURATION PARAMETER, not an empirically tuned optimum.
Provisional center (mu_x, mu_y) is ONLY a reference centroid and NOT the final landmark position.
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_spatial_dispersion(
    global_x: Union[np.ndarray, pd.Series, List[float]],
    global_y: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute provisional centroid, RMS spatial dispersion, and diagnostic radii for observation positions.

    Args:
        global_x: Array of global X positions.
        global_y: Array of global Y positions.

    Returns:
        Tuple[float, float, float, float, float, float]:
            - mu_x: Provisional center X coordinate.
            - mu_y: Provisional center Y coordinate.
            - sigma_spatial: RMS spatial dispersion in meters.
            - median_radius: Median radius from provisional center in meters.
            - p95_radius: 95th percentile radius from provisional center in meters.
            - max_radius: Maximum radius from provisional center in meters.
    """
    gx = np.asarray(global_x, dtype=float)
    gy = np.asarray(global_y, dtype=float)

    if len(gx) == 0 or len(gy) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mu_x = float(np.mean(gx))
    mu_y = float(np.mean(gy))

    r_i = np.sqrt((gx - mu_x) ** 2 + (gy - mu_y) ** 2)

    # RMS spatial dispersion: sqrt(mean(r_i^2))
    sigma_spatial = float(np.sqrt(np.mean(r_i ** 2)))

    median_radius = float(np.median(r_i))
    max_radius = float(np.max(r_i))
    p95_radius = float(np.percentile(r_i, 95)) if len(r_i) > 1 else max_radius

    return mu_x, mu_y, sigma_spatial, median_radius, p95_radius, max_radius


def calculate_geometry_score(
    sigma_spatial: float,
    sigma0_m: float = 0.75,
) -> float:
    """
    Calculate geometry-consistency score w_geo = exp(-sigma_spatial^2 / (2 * sigma0^2)).

    Args:
        sigma_spatial: RMS spatial dispersion in meters.
        sigma0_m: Scale parameter in meters (default: 0.75).

    Returns:
        float: Geometry score in range (0, 1].
    """
    if sigma0_m <= 0:
        raise ValueError("Scale parameter sigma0_m must be strictly positive.")

    w_geo = float(np.exp(- (sigma_spatial ** 2) / (2.0 * (sigma0_m ** 2))))
    w_geo = float(np.clip(w_geo, 1e-15, 1.0))
    return w_geo


def compute_geometry_features(
    candidate_df: pd.DataFrame,
    sigma0_m: float = 0.75,
) -> pd.DataFrame:
    """
    Compute geometry features for all candidate components.

    Args:
        candidate_df: DataFrame containing 'candidate_id', 'global_x', 'global_y'.
        sigma0_m: Scale parameter in meters (default: 0.75).

    Returns:
        pd.DataFrame: DataFrame containing per-candidate geometry metrics:
            - candidate_id
            - num_observations
            - candidate_x
            - candidate_y
            - sigma_spatial_m
            - median_radius_m
            - p95_radius_m
            - max_radius_m
            - geometry_score
    """
    required = ["candidate_id", "global_x", "global_y"]
    missing = [c for c in required if c not in candidate_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    results = []
    grouped = candidate_df.groupby("candidate_id", sort=True)

    for cand_id, group in grouped:
        n_obs = len(group)
        gx = group["global_x"].values
        gy = group["global_y"].values

        mu_x, mu_y, sigma_sp, r_med, r_p95, r_max = compute_spatial_dispersion(gx, gy)
        score = calculate_geometry_score(sigma_sp, sigma0_m=sigma0_m)

        results.append({
            "candidate_id": cand_id,
            "num_observations": int(n_obs),
            "candidate_x": mu_x,
            "candidate_y": mu_y,
            "sigma_spatial_m": sigma_sp,
            "median_radius_m": r_med,
            "p95_radius_m": r_p95,
            "max_radius_m": r_max,
            "geometry_score": score,
        })

    res_df = pd.DataFrame(results)
    return res_df


def plot_geometry_score_distribution(
    geometry_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot histogram of candidate geometry consistency score distribution.

    Args:
        geometry_df: DataFrame containing 'geometry_score'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        geometry_df["geometry_score"],
        bins=20,
        range=(0.0, 1.0),
        color="#2a9d8f",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_title("EWMVC Geometry Score Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Geometry Consistency Score [0, 1]", fontsize=12)
    ax.set_ylabel("Number of Candidate Components", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_geometry_vs_component_size(
    geometry_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot scatter plot of geometry score vs component size (number of observations).

    Args:
        geometry_df: DataFrame containing 'num_observations' and 'geometry_score'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        geometry_df["num_observations"],
        geometry_df["geometry_score"],
        c="#e76f51",
        edgecolors="black",
        linewidths=0.5,
        s=35,
        alpha=0.8,
    )

    ax.set_title("EWMVC Geometry Score vs Component Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Component Size (Number of Observations)", fontsize=12)
    ax.set_ylabel("Geometry Consistency Score [0, 1]", fontsize=12)
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

    geo_cfg = config.get("ewmvc", {}).get("geometry", {})
    sigma0 = float(geo_cfg.get("sigma0_m", 0.75))

    in_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"Input candidate CSV file '{in_csv}' not found. Run Step 20B (src/graph_builder.py) first."
        )

    cand_df = pd.read_csv(in_csv)

    print("======================================================================")
    print("  EWMVC STEP 20D: GEOMETRIC CONSISTENCY CALCULATION")
    print("======================================================================")
    print(f"Loaded candidate observations     : {len(cand_df)}")
    print(f"Unique candidate component IDs    : {cand_df['candidate_id'].nunique()}")
    print(f"Geometry scale parameter (sigma0) : {sigma0} m (Configuration Parameter)")

    # 1. Calculate Geometry Features
    geometry_df = compute_geometry_features(cand_df, sigma0_m=sigma0)

    # 2. Save candidate geometry CSV
    out_geometry_csv = os.path.join(out_dir, "ewmvc_candidate_geometry.csv")
    geometry_df.to_csv(out_geometry_csv, index=False)
    print(f"Saved Candidate Geometry CSV     : '{out_geometry_csv}'")

    # 3. Generate diagnostic plots
    out_hist_png = os.path.join(out_dir, "ewmvc_geometry_score_distribution.png")
    plot_geometry_score_distribution(geometry_df, out_hist_png)
    print(f"Saved Geometry Distribution PNG  : '{out_hist_png}'")

    out_scatter_png = os.path.join(out_dir, "ewmvc_geometry_vs_component_size.png")
    plot_geometry_vs_component_size(geometry_df, out_scatter_png)
    print(f"Saved Geometry vs Size Plot PNG  : '{out_scatter_png}'")

    # Summary Statistics Report
    scores = geometry_df["geometry_score"]
    sigmas = geometry_df["sigma_spatial_m"]

    print("\n--- Geometry Score Measured Results ---")
    print(f"Total Candidate Components        : {len(geometry_df)}")
    print(f"Minimum Geometry Score            : {scores.min():.6f}")
    print(f"Maximum Geometry Score            : {scores.max():.6f}")
    print(f"Mean Geometry Score               : {scores.mean():.6f}")
    print(f"Median Geometry Score             : {scores.median():.6f}")

    print("\n--- Spatial Dispersion (sigma_spatial) Measured Results ---")
    print(f"Minimum sigma_spatial             : {sigmas.min():.6f} m")
    print(f"Maximum sigma_spatial             : {sigmas.max():.6f} m")
    print(f"Mean sigma_spatial                : {sigmas.mean():.6f} m")
    print(f"Median sigma_spatial              : {sigmas.median():.6f} m")

    print("\n--- Threshold Extrema Counts ---")
    print(f"Candidates with Geometry Score > 0.5 : {int((scores > 0.5).sum())}")
    print(f"Candidates with Geometry Score > 0.8 : {int((scores > 0.8).sum())}")
    print(f"Candidates with Geometry Score > 0.9 : {int((scores > 0.9).sum())}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
