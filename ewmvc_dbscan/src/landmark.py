"""
EWMVC Inverse-Covariance Landmark Estimation Module.

Fuses 2D perception observations of consensus-supported candidate components using
inverse-covariance (precision-weighted) estimation:

    p_hat = (sum Sigma_i^-1)^-1 (sum Sigma_i^-1 p_i)
    Sigma_landmark = (sum Sigma_i^-1)^-1

Only candidate components with accepted == True from the consensus stage are estimated.
"""

import os
from typing import Dict, Tuple, Any, List, Optional, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def fuse_observations_inverse_covariance(
    global_x: Union[np.ndarray, pd.Series, List[float]],
    global_y: Union[np.ndarray, pd.Series, List[float]],
    cov_xx: Union[np.ndarray, pd.Series, List[float]],
    cov_xy: Union[np.ndarray, pd.Series, List[float]],
    cov_yy: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Perform inverse-covariance precision-weighted fusion for a set of 2D spatial observations.

    Args:
        global_x: Array of observation global X coordinates.
        global_y: Array of observation global Y coordinates.
        cov_xx: Array of cov_xx elements.
        cov_xy: Array of cov_xy elements.
        cov_yy: Array of cov_yy elements.

    Returns:
        Tuple[float, float, float, float, float, float, float, float]:
            - landmark_x: Fused 2D position X coordinate.
            - landmark_y: Fused 2D position Y coordinate.
            - landmark_cov_xx: Fused 2D covariance XX element.
            - landmark_cov_xy: Fused 2D covariance XY element.
            - landmark_cov_yy: Fused 2D covariance YY element.
            - landmark_sigma_x: Standard deviation in X (sqrt(landmark_cov_xx)).
            - landmark_sigma_y: Standard deviation in Y (sqrt(landmark_cov_yy)).
            - weighted_shift_m: Euclidean shift between fused position and arithmetic mean.
    """
    gx = np.asarray(global_x, dtype=float)
    gy = np.asarray(global_y, dtype=float)
    c_xx = np.asarray(cov_xx, dtype=float)
    c_xy = np.asarray(cov_xy, dtype=float)
    c_yy = np.asarray(cov_yy, dtype=float)

    n_obs = len(gx)
    if n_obs == 0:
        raise ValueError("Cannot fuse empty observation set.")

    mean_x = float(np.mean(gx))
    mean_y = float(np.mean(gy))

    W_sum = np.zeros((2, 2), dtype=float)
    b_sum = np.zeros(2, dtype=float)

    for i in range(n_obs):
        p_i = np.array([gx[i], gy[i]], dtype=float)
        sigma_i = np.array([[c_xx[i], c_xy[i]], [c_xy[i], c_yy[i]]], dtype=float)

        # Check positive-definiteness & determinant
        det_i = sigma_i[0, 0] * sigma_i[1, 1] - sigma_i[0, 1] ** 2
        if not np.isfinite(det_i) or det_i <= 1e-12 or sigma_i[0, 0] <= 0 or sigma_i[1, 1] <= 0:
            # Fallback pseudo-inverse
            try:
                w_i = np.linalg.pinv(sigma_i)
            except Exception:
                continue
        else:
            w_i = np.array([
                [sigma_i[1, 1], -sigma_i[0, 1]],
                [-sigma_i[0, 1], sigma_i[0, 0]]
            ], dtype=float) / det_i

        W_sum += w_i
        b_sum += w_i @ p_i

    det_W = W_sum[0, 0] * W_sum[1, 1] - W_sum[0, 1] ** 2

    if not np.isfinite(det_W) or det_W <= 1e-12:
        sigma_landmark = np.linalg.pinv(W_sum)
        p_hat = sigma_landmark @ b_sum
    else:
        sigma_landmark = np.array([
            [W_sum[1, 1], -W_sum[0, 1]],
            [-W_sum[0, 1], W_sum[0, 0]]
        ], dtype=float) / det_W
        p_hat = sigma_landmark @ b_sum

    landmark_x = float(p_hat[0])
    landmark_y = float(p_hat[1])
    landmark_cov_xx = float(sigma_landmark[0, 0])
    landmark_cov_xy = float(sigma_landmark[0, 1])
    landmark_cov_yy = float(sigma_landmark[1, 1])

    landmark_sigma_x = float(np.sqrt(max(0.0, landmark_cov_xx)))
    landmark_sigma_y = float(np.sqrt(max(0.0, landmark_cov_yy)))

    weighted_shift_m = float(np.sqrt((landmark_x - mean_x) ** 2 + (landmark_y - mean_y) ** 2))

    return (
        landmark_x,
        landmark_y,
        landmark_cov_xx,
        landmark_cov_xy,
        landmark_cov_yy,
        landmark_sigma_x,
        landmark_sigma_y,
        weighted_shift_m,
    )


def compute_landmarks(
    candidate_obs_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate representative 2D landmark locations for all consensus-supported candidate components.

    Args:
        candidate_obs_df: Observations DataFrame with 'candidate_id', 'global_x', 'global_y', 'cov_xx', 'cov_xy', 'cov_yy'.
        consensus_df: Consensus DataFrame with 'candidate_id', 'accepted', 'consensus_score', etc.

    Returns:
        pd.DataFrame: Landmarks DataFrame for accepted candidates (accepted == True).
    """
    required_obs = ["candidate_id", "global_x", "global_y", "cov_xx", "cov_xy", "cov_yy"]
    missing_obs = [c for c in required_obs if c not in candidate_obs_df.columns]
    if missing_obs:
        raise ValueError(f"Input candidate observations DataFrame missing columns: {missing_obs}")

    required_con = ["candidate_id", "accepted", "consensus_score"]
    missing_con = [c for c in required_con if c not in consensus_df.columns]
    if missing_con:
        raise ValueError(f"Input consensus DataFrame missing columns: {missing_con}")

    # Select accepted candidate component IDs
    accepted_df = consensus_df[consensus_df["accepted"] == True].copy()
    accepted_ids = accepted_df["candidate_id"].unique()

    results = []

    for cand_id in accepted_ids:
        obs_group = candidate_obs_df[candidate_obs_df["candidate_id"] == cand_id]
        if len(obs_group) == 0:
            continue

        con_info = accepted_df[accepted_df["candidate_id"] == cand_id].iloc[0]

        gx = obs_group["global_x"].values
        gy = obs_group["global_y"].values
        c_xx = obs_group["cov_xx"].values
        c_xy = obs_group["cov_xy"].values
        c_yy = obs_group["cov_yy"].values

        lx, ly, lcov_xx, lcov_xy, lcov_yy, lsig_x, lsig_y, w_shift = fuse_observations_inverse_covariance(
            gx, gy, c_xx, c_xy, c_yy
        )

        results.append({
            "candidate_id": cand_id,
            "landmark_x": lx,
            "landmark_y": ly,
            "landmark_cov_xx": lcov_xx,
            "landmark_cov_xy": lcov_xy,
            "landmark_cov_yy": lcov_yy,
            "landmark_sigma_x": lsig_x,
            "landmark_sigma_y": lsig_y,
            "consensus_score": float(con_info["consensus_score"]),
            "observation_count": int(con_info.get("num_observations", len(obs_group))),
            "unique_viewpoint_bins": int(con_info.get("unique_viewpoint_bins", 1)),
            "sigma_spatial_m": float(con_info.get("sigma_spatial_m", 0.0)),
            "duration_s": float(con_info.get("duration_s", 0.0)),
            "max_baseline_m": float(con_info.get("max_baseline_m", 0.0)),
            "viewpoint_entropy": float(con_info.get("viewpoint_entropy", 0.0)),
            "geometry_score": float(con_info.get("geometry_score", 0.0)),
            "persistence_score": float(con_info.get("persistence_score", 0.0)),
            "viewpoint_diversity": float(con_info.get("viewpoint_diversity", 0.0)),
            "weighted_shift_m": w_shift,
        })

    landmarks_df = pd.DataFrame(results)
    return landmarks_df


def compute_inter_landmark_distances(landmarks_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute pairwise Euclidean distances between estimated landmark positions.

    Args:
        landmarks_df: Landmarks DataFrame with 'landmark_x', 'landmark_y'.

    Returns:
        Tuple[float, float, float]: (min_dist, median_dist, max_dist).
    """
    if len(landmarks_df) <= 1:
        return 0.0, 0.0, 0.0

    pos = landmarks_df[["landmark_x", "landmark_y"]].values
    dx = pos[:, 0, None] - pos[:, 0]
    dy = pos[:, 1, None] - pos[:, 1]
    dist_matrix = np.sqrt(dx ** 2 + dy ** 2)

    triu_indices = np.triu_indices(len(landmarks_df), k=1)
    pairwise_dists = dist_matrix[triu_indices]

    if len(pairwise_dists) == 0:
        return 0.0, 0.0, 0.0

    min_d = float(np.min(pairwise_dists))
    med_d = float(np.median(pairwise_dists))
    max_d = float(np.max(pairwise_dists))

    return min_d, med_d, max_d


def plot_landmark_map(
    landmarks_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot consensus-supported landmark estimates with vehicle trajectory and observation cloud.

    Args:
        landmarks_df: Landmarks DataFrame containing 'landmark_x', 'landmark_y'.
        obs_df: Observation DataFrame containing 'global_x', 'global_y', 'vehicle_x', 'vehicle_y'.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot vehicle trajectory if available
    if "vehicle_x" in obs_df.columns and "vehicle_y" in obs_df.columns:
        traj = obs_df[["vehicle_x", "vehicle_y"]].drop_duplicates()
        ax.plot(traj["vehicle_x"], traj["vehicle_y"], "k--", alpha=0.5, linewidth=1.5, label="Vehicle Trajectory")

    # Plot low-alpha observation cloud
    ax.scatter(
        obs_df["global_x"],
        obs_df["global_y"],
        c="lightblue",
        s=10,
        alpha=0.25,
        label=f"Perception Observations (N={len(obs_df)})",
    )

    # Plot estimated landmarks
    ax.scatter(
        landmarks_df["landmark_x"],
        landmarks_df["landmark_y"],
        c="#d90429",
        s=45,
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        label=f"Consensus-Supported Landmarks (N={len(landmarks_df)})",
    )

    ax.set_title("EWMVC Consensus-Supported Landmark Map", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Global X [m]", fontsize=12)
    ax.set_ylabel("Global Y [m]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axis("equal")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_landmark_uncertainty(
    landmarks_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot estimated landmark positions with 1-sigma uncertainty ellipses.

    Args:
        landmarks_df: Landmarks DataFrame.
        output_path: File path to save PNG figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        landmarks_df["landmark_x"],
        landmarks_df["landmark_y"],
        c="#1d3557",
        s=40,
        marker="o",
        label=f"Landmark Positions (N={len(landmarks_df)})",
    )

    # Draw 1-sigma covariance ellipses
    for _, row in landmarks_df.iterrows():
        lx = row["landmark_x"]
        ly = row["landmark_y"]
        c_xx = row["landmark_cov_xx"]
        c_xy = row["landmark_cov_xy"]
        c_yy = row["landmark_cov_yy"]

        cov_matrix = np.array([[c_xx, c_xy], [c_xy, c_yy]])
        vals, vecs = np.linalg.eigh(cov_matrix)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2.0 * np.sqrt(np.maximum(0.0, vals))  # 1-sigma radii -> diameter

        ell = Ellipse(
            xy=(lx, ly),
            width=width,
            height=height,
            angle=angle,
            edgecolor="#e63946",
            facecolor="none",
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
        )
        ax.add_patch(ell)

    ax.set_title("EWMVC Landmark Spatial Position Uncertainty (1-Sigma Ellipses)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Global X [m]", fontsize=12)
    ax.set_ylabel("Global Y [m]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axis("equal")
    ax.legend(loc="upper right", fontsize=10)

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

    in_obs_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    in_con_csv = os.path.join(out_dir, "ewmvc_consensus.csv")

    if not os.path.exists(in_obs_csv):
        raise FileNotFoundError(f"Input file '{in_obs_csv}' not found. Run Step 20B first.")
    if not os.path.exists(in_con_csv):
        raise FileNotFoundError(f"Input file '{in_con_csv}' not found. Run Step 20G first.")

    obs_df = pd.read_csv(in_obs_csv)
    con_df = pd.read_csv(in_con_csv)

    print("======================================================================")
    print("  EWMVC STEP 20H: INVERSE-COVARIANCE LANDMARK ESTIMATION")
    print("======================================================================")
    print(f"Loaded observations                 : {len(obs_df)}")
    print(f"Loaded consensus components         : {len(con_df)}")

    # 1. Compute Landmarks for accepted candidates
    landmarks_df = compute_landmarks(obs_df, con_df)

    # 2. Save Landmark CSV
    out_landmarks_csv = os.path.join(out_dir, "ewmvc_landmarks.csv")
    landmarks_df.to_csv(out_landmarks_csv, index=False)
    print(f"Saved Landmarks CSV                 : '{out_landmarks_csv}'")

    # 3. Generate Visualizations
    out_map_png = os.path.join(out_dir, "ewmvc_landmark_map.png")
    plot_landmark_map(landmarks_df, obs_df, out_map_png)
    print(f"Saved Landmark Map PNG              : '{out_map_png}'")

    out_unc_png = os.path.join(out_dir, "ewmvc_landmark_uncertainty.png")
    plot_landmark_uncertainty(landmarks_df, out_unc_png)
    print(f"Saved Landmark Uncertainty PNG      : '{out_unc_png}'")

    # Diagnostics & Summary Statistics
    sig_x = landmarks_df["landmark_sigma_x"]
    sig_y = landmarks_df["landmark_sigma_y"]
    shifts = landmarks_df["weighted_shift_m"]
    min_d, med_d, max_d = compute_inter_landmark_distances(landmarks_df)

    print("\n--- Landmark Estimation Summary ---")
    print(f"Consensus-Supported Landmarks       : {len(landmarks_df)}")
    print(f"Landmark X Bounds [m]              : min={landmarks_df['landmark_x'].min():.4f}, max={landmarks_df['landmark_x'].max():.4f}")
    print(f"Landmark Y Bounds [m]              : min={landmarks_df['landmark_y'].min():.4f}, max={landmarks_df['landmark_y'].max():.4f}")

    print("\n--- Landmark Position Uncertainty (Standard Deviations) ---")
    print(f"landmark_sigma_x [m]                : mean={sig_x.mean():.6f}, median={sig_x.median():.6f}, max={sig_x.max():.6f}")
    print(f"landmark_sigma_y [m]                : mean={sig_y.mean():.6f}, median={sig_y.median():.6f}, max={sig_y.max():.6f}")

    print("\n--- Weighted Centroid Shift (vs. Ordinary Mean) ---")
    print(f"weighted_shift_m [m]                : min={shifts.min():.6f}, mean={shifts.mean():.6f}, median={shifts.median():.6f}, max={shifts.max():.6f}")

    print("\n--- Spatial Sanity Check (Inter-Landmark Distances) ---")
    print(f"Minimum Inter-Landmark Distance     : {min_d:.4f} m")
    print(f"Median Inter-Landmark Distance      : {med_d:.4f} m")
    print(f"Maximum Inter-Landmark Distance     : {max_d:.4f} m")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
