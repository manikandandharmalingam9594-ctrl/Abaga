"""
EWMVC Mahalanobis Compatibility Graph & Candidate Landmark Component Extraction.

Constructs an observation compatibility graph based on pairwise 2D Mahalanobis statistical distance
using propagated spatial covariance matrices:

    d²_M(i,j) = (p_i - p_j)^T (Σ_i + Σ_j)^(-1) (p_i - p_j)

Edges are created when d²_M(i,j) <= chi_square_threshold (default 5.991, 95% chi-square threshold for 2-DOF).
Connected components extracted from the graph form candidate landmark components.

COMPUTATIONAL OPTIMIZATION vs EWMVC DECISION RULE:
- candidate_radius_m (default 1.5 m) is ONLY a spatial pre-filter optimization using cKDTree to prevent O(N²)
  pairwise Mahalanobis evaluations over 9,774 observations.
- The final compatibility decision rule is strictly based on Mahalanobis distance d²_M(i,j) <= chi_square_threshold.
"""

import os
from typing import Dict, Tuple, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_mahalanobis_distance_sq(
    p1: np.ndarray,
    cov1: np.ndarray,
    p2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """
    Compute the squared Mahalanobis distance d²_M between two 2D observations i and j.

    d²_M(i,j) = (p_i - p_j)^T (Σ_i + Σ_j)^(-1) (p_i - p_j)

    Args:
        p1 (np.ndarray): 2D global position vector [x_i, y_i].
        cov1 (np.ndarray): 2x2 covariance matrix for observation i.
        p2 (np.ndarray): 2D global position vector [x_j, y_j].
        cov2 (np.ndarray): 2x2 covariance matrix for observation j.

    Returns:
        float: Squared Mahalanobis distance d²_M(i, j). Returns float('inf') if matrices
               are non-finite, non-symmetric, or numerically singular.
    """
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)

    if p1.shape != (2,) or p2.shape != (2,):
        raise ValueError("Positions p1 and p2 must be 2D vectors.")
    if cov1.shape != (2, 2) or cov2.shape != (2, 2):
        raise ValueError("Covariances cov1 and cov2 must be 2x2 matrices.")

    # Check finite values
    if not (np.isfinite(p1).all() and np.isfinite(p2).all() and np.isfinite(cov1).all() and np.isfinite(cov2).all()):
        return float("inf")

    # Check symmetry
    if abs(cov1[0, 1] - cov1[1, 0]) > 1e-8 or abs(cov2[0, 1] - cov2[1, 0]) > 1e-8:
        return float("inf")

    cov_sum = cov1 + cov2
    a = cov_sum[0, 0]
    b = cov_sum[0, 1]
    c = cov_sum[1, 1]
    det = a * c - b * b

    # Check positive-definiteness & numerical singularity
    if not np.isfinite(det) or det <= 1e-12 or a <= 0 or c <= 0:
        # Robust fallback using pseudo-inverse
        try:
            inv_cov = np.linalg.pinv(cov_sum)
            dp = p1 - p2
            d_sq = float(dp.T @ inv_cov @ dp)
            return d_sq if np.isfinite(d_sq) and d_sq >= 0 else float("inf")
        except Exception:
            return float("inf")

    dp = p1 - p2
    d_sq = (c * dp[0] ** 2 - 2.0 * b * dp[0] * dp[1] + a * dp[1] ** 2) / det
    return float(max(0.0, d_sq))


def build_mahalanobis_graph(
    data: pd.DataFrame,
    chi_square_threshold: float = 5.991,
    candidate_radius_m: float = 1.5,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Construct an observation compatibility graph using Mahalanobis distance.

    Computational Strategy:
    1. Uses cKDTree with candidate_radius_m ONLY as a spatial pre-filter to find candidate pairs.
    2. Computes exact squared Mahalanobis distance d²_M(i,j) for candidate pairs.
    3. Adds edge (i,j) if d²_M(i,j) <= chi_square_threshold.
    4. Graph construction strictly avoids using perception labels or DBSCAN cluster IDs.

    Args:
        data (pd.DataFrame): DataFrame containing global_x, global_y, cov_xx, cov_xy, cov_yy.
        chi_square_threshold (float): Theoretical 95% chi-square threshold for 2-DOF (default: 5.991).
        candidate_radius_m (float): Computational spatial pre-filter radius in meters (default: 1.5).

    Returns:
        Tuple[csr_matrix, np.ndarray, np.ndarray, int]:
            - adj_matrix: Sparse SciPy CSR adjacency matrix (N x N).
            - candidate_ids: Array of connected component candidate IDs for each observation.
            - degrees: Array of graph node degrees for each observation.
            - num_components: Total number of connected components.
    """
    required_cols = ["global_x", "global_y", "cov_xx", "cov_xy", "cov_yy"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns for graph building: {missing}")

    n_obs = len(data)
    pos = data[["global_x", "global_y"]].values
    cov_xx = data["cov_xx"].values
    cov_xy = data["cov_xy"].values
    cov_yy = data["cov_yy"].values

    # 1. Spatial pre-filtering via cKDTree (Computational optimization)
    tree = cKDTree(pos)
    candidate_pairs = list(tree.query_pairs(r=candidate_radius_m))

    row_indices = []
    col_indices = []

    # 2. Evaluate exact Mahalanobis compatibility on candidate pairs
    for i, j in candidate_pairs:
        p1 = pos[i]
        p2 = pos[j]
        c1 = np.array([[cov_xx[i], cov_xy[i]], [cov_xy[i], cov_yy[i]]])
        c2 = np.array([[cov_xx[j], cov_xy[j]], [cov_xy[j], cov_yy[j]]])

        d_sq = compute_mahalanobis_distance_sq(p1, c1, p2, c2)

        if d_sq <= chi_square_threshold:
            row_indices.extend([i, j])
            col_indices.extend([j, i])

    # 3. Construct sparse symmetric adjacency matrix
    if row_indices:
        data_ones = np.ones(len(row_indices), dtype=int)
        adj_matrix = csr_matrix((data_ones, (row_indices, col_indices)), shape=(n_obs, n_obs))
    else:
        adj_matrix = csr_matrix((n_obs, n_obs), dtype=int)

    # 4. Connected components extraction
    n_components, candidate_ids = connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True
    )

    # 5. Graph node degrees
    degrees = np.asarray(adj_matrix.sum(axis=1)).ravel()

    return adj_matrix, candidate_ids, degrees, n_components


def generate_graph_summary(
    data: pd.DataFrame,
    adj_matrix: csr_matrix,
    candidate_ids: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    """
    Generate graph summary metrics table.

    Args:
        data (pd.DataFrame): Observation DataFrame.
        adj_matrix (csr_matrix): Sparse adjacency matrix.
        candidate_ids (np.ndarray): Component candidate IDs.
        n_components (int): Total number of components.

    Returns:
        pd.DataFrame: Summary statistics DataFrame with single row.
    """
    n_obs = len(data)
    total_edges = int(adj_matrix.nnz // 2)

    comp_series = pd.Series(candidate_ids)
    comp_counts = comp_series.value_counts()

    singleton_components = int((comp_counts == 1).sum())
    largest_component_size = int(comp_counts.max()) if len(comp_counts) > 0 else 0
    average_component_size = float(comp_counts.mean()) if len(comp_counts) > 0 else 0.0

    summary_df = pd.DataFrame([{
        "total_observations": n_obs,
        "total_edges": total_edges,
        "number_of_components": n_components,
        "singleton_components": singleton_components,
        "largest_component_size": largest_component_size,
        "average_component_size": round(average_component_size, 4),
    }])

    return summary_df


def plot_candidate_components(
    data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Visualize EWMVC candidate landmark components.

    Args:
        data (pd.DataFrame): DataFrame containing global_x, global_y, candidate_id, vehicle_x, vehicle_y.
        output_path (str): File path to save the generated PNG figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot vehicle trajectory if available
    if "vehicle_x" in data.columns and "vehicle_y" in data.columns:
        traj = data[["vehicle_x", "vehicle_y"]].drop_duplicates()
        ax.plot(traj["vehicle_x"], traj["vehicle_y"], "k--", alpha=0.5, linewidth=1.5, label="Vehicle Trajectory")

    # Singletons vs Multi-observation components
    comp_counts = data["candidate_id"].value_counts()
    singletons = comp_counts[comp_counts == 1].index
    multi_comps = comp_counts[comp_counts > 1].index

    df_singles = data[data["candidate_id"].isin(singletons)]
    df_multi = data[data["candidate_id"].isin(multi_comps)]

    # Scatter singletons
    if len(df_singles) > 0:
        ax.scatter(
            df_singles["global_x"],
            df_singles["global_y"],
            c="gray",
            s=12,
            alpha=0.4,
            label=f"Singleton Components (n={len(singletons)})",
        )

    # Scatter multi-observation candidate components
    if len(df_multi) > 0:
        scatter = ax.scatter(
            df_multi["global_x"],
            df_multi["global_y"],
            c=df_multi["candidate_id"],
            cmap="tab20",
            s=25,
            alpha=0.85,
            label=f"Multi-Observation Candidates (n={len(multi_comps)})",
        )

    ax.set_title("EWMVC Candidate Landmark Components", fontsize=14, fontweight="bold", pad=12)
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

    graph_cfg = config.get("ewmvc", {}).get("graph", {})
    chi2_thresh = float(graph_cfg.get("chi_square_threshold", 5.991))
    cand_radius = float(graph_cfg.get("candidate_radius_m", 1.5))

    in_csv = os.path.join(out_dir, "ewmvc_observations_with_covariance.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"Input file '{in_csv}' not found. Run Step 20A (src/uncertainty.py) first."
        )

    df_obs = pd.read_csv(in_csv)

    print("======================================================================")
    print("  EWMVC STEP 20B: MAHALANOBIS COMPATIBILITY GRAPH & COMPONENTS")
    print("======================================================================")
    print(f"Loaded observations               : {len(df_obs)}")
    print(f"Chi-square 2-DOF threshold        : {chi2_thresh} (Theoretical 95%)")
    print(f"Candidate spatial pre-filter      : {cand_radius} m (Computational optimization)")

    # 1. Build Mahalanobis graph & extract components
    adj_matrix, candidate_ids, degrees, n_components = build_mahalanobis_graph(
        df_obs,
        chi_square_threshold=chi2_thresh,
        candidate_radius_m=cand_radius,
    )

    df_obs["candidate_id"] = candidate_ids
    df_obs["mahalanobis_graph_degree"] = degrees

    # 2. Save detailed observations with candidate component IDs
    out_components_csv = os.path.join(out_dir, "ewmvc_candidate_components.csv")
    df_obs.to_csv(out_components_csv, index=False)
    print(f"Saved Candidate Components CSV    : '{out_components_csv}'")

    # 3. Generate & save graph summary CSV
    summary_df = generate_graph_summary(df_obs, adj_matrix, candidate_ids, n_components)
    out_summary_csv = os.path.join(out_dir, "ewmvc_graph_summary.csv")
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Saved Graph Summary CSV           : '{out_summary_csv}'")

    # 4. Generate candidate component visualization
    out_png = os.path.join(out_dir, "ewmvc_candidate_components.png")
    plot_candidate_components(df_obs, out_png)
    print(f"Saved Components Plot PNG         : '{out_png}'")

    # Print Summary Report
    s = summary_df.iloc[0]
    print("\n--- Graph Builder Summary Report ---")
    print(f"Total Observations                : {s['total_observations']}")
    print(f"Total Graph Edges                 : {s['total_edges']}")
    print(f"Number of Candidate Components   : {s['number_of_components']}")
    print(f"Singleton Components              : {s['singleton_components']}")
    print(f"Largest Component Size            : {s['largest_component_size']}")
    print(f"Average Component Size            : {s['average_component_size']:.4f}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
