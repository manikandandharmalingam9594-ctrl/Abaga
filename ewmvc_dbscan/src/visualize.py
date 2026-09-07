"""
Visualization module for cone observations, vehicle trajectory, and DBSCAN clustering.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_global_observations(data: pd.DataFrame, output_path: str) -> None:
    """
    Plot raw global cone observations along with the vehicle trajectory.

    Args:
        data (pd.DataFrame): DataFrame containing 'global_x', 'global_y', 'vehicle_x', 'vehicle_y'.
        output_path (str): File path to save the generated plot image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot vehicle trajectory
    if "vehicle_x" in data.columns and "vehicle_y" in data.columns:
        vehicle_poses = data[["vehicle_x", "vehicle_y"]].drop_duplicates()
        ax.plot(
            vehicle_poses["vehicle_x"],
            vehicle_poses["vehicle_y"],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Vehicle Trajectory",
        )

    # Plot global cone observations
    ax.scatter(
        data["global_x"],
        data["global_y"],
        c="tab:blue",
        s=12,
        alpha=0.6,
        edgecolors="none",
        label="Raw Cone Observations",
    )

    ax.set_title("Global Cone Observations & Vehicle Trajectory", fontsize=14, pad=12)
    ax.set_xlabel("Global X (m)", fontsize=12)
    ax.set_ylabel("Global Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved successfully to '{output_path}'.")


def plot_dbscan_map(
    clustered: pd.DataFrame, cone_map: pd.DataFrame, output_path: str
) -> None:
    """
    Plot DBSCAN clustering results: noise points, clustered observations,
    weighted cluster centroids with IDs, and vehicle trajectory.

    Args:
        clustered (pd.DataFrame): Clustered observations DataFrame containing 'cluster_id',
                                 'global_x', 'global_y', 'vehicle_x', 'vehicle_y'.
        cone_map (pd.DataFrame): Summarized cone map DataFrame containing 'cluster_id',
                                 'global_x', 'global_y'.
        output_path (str): File path to save the generated DBSCAN map plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Plot Vehicle Trajectory
    if "vehicle_x" in clustered.columns and "vehicle_y" in clustered.columns:
        vehicle_poses = clustered[["vehicle_x", "vehicle_y"]].drop_duplicates()
        ax.plot(
            vehicle_poses["vehicle_x"],
            vehicle_poses["vehicle_y"],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Vehicle Trajectory",
            zorder=1,
        )

    # 2. Separate Clustered Observations and Noise Points
    noise_points = clustered[clustered["cluster_id"] == -1]
    valid_points = clustered[clustered["cluster_id"] != -1]

    # Plot Noise Points
    if not noise_points.empty:
        ax.scatter(
            noise_points["global_x"],
            noise_points["global_y"],
            c="gray",
            marker="x",
            s=25,
            alpha=0.6,
            linewidths=1,
            label="DBSCAN Noise (-1)",
            zorder=2,
        )

    # Plot Clustered Observations (colored by cluster_id)
    if not valid_points.empty:
        ax.scatter(
            valid_points["global_x"],
            valid_points["global_y"],
            c=valid_points["cluster_id"],
            cmap="tab20",
            s=12,
            alpha=0.5,
            edgecolors="none",
            label="Clustered Observations",
            zorder=3,
        )

    # 3. Plot Cluster Centroids & Annotate Cluster IDs
    if not cone_map.empty:
        ax.scatter(
            cone_map["global_x"],
            cone_map["global_y"],
            c="red",
            marker="o",
            s=40,
            edgecolors="black",
            linewidths=1,
            label="Cone Centroids",
            zorder=4,
        )

        for _, row in cone_map.iterrows():
            cid = int(row["cluster_id"])
            cx, cy = row["global_x"], row["global_y"]
            ax.annotate(
                str(cid),
                (cx, cy),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="darkred",
                zorder=5,
            )

    ax.set_title("Fixed-Epsilon DBSCAN Cone Map Baseline", fontsize=14, pad=12)
    ax.set_xlabel("Global X (m)", fontsize=12)
    ax.set_ylabel("Global Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"DBSCAN map plot saved successfully to '{output_path}'.")
