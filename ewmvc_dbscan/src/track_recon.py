"""
EWMVC Track Boundary Reconstruction Module (Step 20I)

Reconstructs left and right track boundary curves from consensus-supported EWMVC landmark estimates
and vehicle telemetry state logs.

Calculates trajectory-relative longitudinal (s) and signed lateral (l) coordinates:
    s = (x_l - x_v) * cos(yaw) + (y_l - y_v) * sin(yaw)
    l = -(x_l - x_v) * sin(yaw) + (y_l - y_v) * cos(yaw)

Where:
    s > 0: landmark is ahead of vehicle
    s < 0: landmark is behind vehicle
    l > 0: landmark is on left side
    l < 0: landmark is on right side

Fits parametric cubic splines x(s) and y(s) for left and right boundary estimates without
hallucinating missing cones or creating synthetic landmark detections.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def compute_trajectory_relative_coordinates(
    landmark_x: float,
    landmark_y: float,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_yaw: float
) -> Tuple[float, float]:
    """
    Calculates longitudinal (s) and signed lateral (l) coordinates of a landmark relative to a vehicle pose.

    Args:
        landmark_x: Global X position of landmark (m).
        landmark_y: Global Y position of landmark (m).
        vehicle_x: Global X position of vehicle (m).
        vehicle_y: Global Y position of vehicle (m).
        vehicle_yaw: Vehicle heading yaw (rad).

    Returns:
        (s, l): Longitudinal coordinate s (m) and signed lateral coordinate l (m).
    """
    rx = landmark_x - vehicle_x
    ry = landmark_y - vehicle_y
    cos_yaw = np.cos(vehicle_yaw)
    sin_yaw = np.sin(vehicle_yaw)

    s = rx * cos_yaw + ry * sin_yaw
    l = -rx * sin_yaw + ry * cos_yaw

    return float(s), float(l)


def find_closest_trajectory_pose(
    landmark_x: float,
    landmark_y: float,
    telemetry_df: pd.DataFrame
) -> Tuple[float, float, float, float, int]:
    """
    Finds the closest trajectory point on the vehicle telemetry path for a given landmark position.

    Args:
        landmark_x: Global X position of landmark (m).
        landmark_y: Global Y position of landmark (m).
        telemetry_df: Telemetry DataFrame containing 'x', 'y', 'yaw_rad', 'timestamp'.

    Returns:
        (x_v, y_v, yaw_rad, timestamp, idx) of the closest trajectory pose.
    """
    vx = telemetry_df['x'].values
    vy = telemetry_df['y'].values

    dists_sq = (vx - landmark_x) ** 2 + (vy - landmark_y) ** 2
    closest_idx = int(np.argmin(dists_sq))

    row = telemetry_df.iloc[closest_idx]
    return (
        float(row['x']),
        float(row['y']),
        float(row['yaw_rad']),
        float(row['timestamp']),
        closest_idx
    )


def classify_landmark_side(l: float, min_lateral_distance_m: float = 0.5) -> str:
    """
    Classifies landmark side based on signed lateral coordinate l.

    Args:
        l: Signed lateral coordinate (m).
        min_lateral_distance_m: Minimum lateral distance threshold (m).

    Returns:
        'left', 'right', or 'ambiguous'.
    """
    if abs(l) < min_lateral_distance_m:
        return 'ambiguous'
    elif l > 0:
        return 'left'
    else:
        return 'right'


def fit_boundary_curve(
    landmarks_df: pd.DataFrame,
    side_name: str,
    spline_degree: int = 3,
    spline_samples: int = 200
) -> pd.DataFrame:
    """
    Fits a smooth parametric boundary curve x(s) and y(s) for a given side (left/right).

    Minimum landmark requirements:
    - 0 landmarks: returns empty DataFrame with columns ['s_m', 'x_m', 'y_m']
    - 1 landmark: returns single point DataFrame ['s_m', 'x_m', 'y_m']
    - 2 landmarks: fits linear interpolation between s_min and s_max
    - 3+ landmarks: fits parametric cubic spline without extrapolation beyond [s_min, s_max]

    Args:
        landmarks_df: Filtered landmark DataFrame containing 's_m' (or 's_track'), 'landmark_x', 'landmark_y'.
        side_name: 'left' or 'right'.
        spline_degree: Degree of spline (default 3 for cubic).
        spline_samples: Number of sample points along the fitted curve.

    Returns:
        DataFrame containing ['s_m', 'x_m', 'y_m'].
    """
    cols = ['s_m', 'x_m', 'y_m']
    if landmarks_df.empty:
        return pd.DataFrame(columns=cols)

    num_pts = len(landmarks_df)

    # Use longitudinal ordering coordinate for parametric representation
    s_col = 's_track' if 's_track' in landmarks_df.columns else 's_m'
    df_sorted = landmarks_df.sort_values(s_col).copy()

    s_arr = df_sorted[s_col].values
    x_arr = df_sorted['landmark_x'].values
    y_arr = df_sorted['landmark_y'].values

    if num_pts == 1:
        return pd.DataFrame([{
            's_m': float(s_arr[0]),
            'x_m': float(x_arr[0]),
            'y_m': float(y_arr[0])
        }])

    if num_pts == 2:
        s0, s1 = float(s_arr[0]), float(s_arr[1])
        x0, x1 = float(x_arr[0]), float(x_arr[1])
        y0, y1 = float(y_arr[0]), float(y_arr[1])

        if abs(s1 - s0) < 1e-9:
            s_grid = np.full(spline_samples, s0)
            x_grid = np.linspace(x0, x1, spline_samples)
            y_grid = np.linspace(y0, y1, spline_samples)
        else:
            s_grid = np.linspace(s0, s1, spline_samples)
            x_grid = x0 + (s_grid - s0) / (s1 - s0) * (x1 - x0)
            y_grid = y0 + (s_grid - s0) / (s1 - s0) * (y1 - y0)

        return pd.DataFrame({
            's_m': s_grid,
            'x_m': x_grid,
            'y_m': y_grid
        })

    # 3+ landmarks: Parametric cubic spline fitting
    # Ensure s_arr is strictly increasing by adding tiny numerical epsilon if identical
    s_arr_unique = s_arr.copy()
    for i in range(1, len(s_arr_unique)):
        if s_arr_unique[i] <= s_arr_unique[i - 1]:
            s_arr_unique[i] = s_arr_unique[i - 1] + 1e-6

    s_min, s_max = float(s_arr_unique[0]), float(s_arr_unique[-1])
    s_grid = np.linspace(s_min, s_max, spline_samples)

    try:
        cs_x = CubicSpline(s_arr_unique, x_arr, bc_type='natural')
        cs_y = CubicSpline(s_arr_unique, y_arr, bc_type='natural')
        x_grid = cs_x(s_grid)
        y_grid = cs_y(s_grid)
    except Exception:
        # Fallback to linear interpolation if cubic spline fitting encounters numerical singularity
        x_grid = np.interp(s_grid, s_arr_unique, x_arr)
        y_grid = np.interp(s_grid, s_arr_unique, y_arr)

    return pd.DataFrame({
        's_m': s_grid,
        'x_m': x_grid,
        'y_m': y_grid
    })


def compute_centerline(
    left_boundary_df: pd.DataFrame,
    right_boundary_df: pd.DataFrame,
    spline_samples: int = 200
) -> Optional[pd.DataFrame]:
    """
    Calculates the track centerline as the average of left and right boundary curves
    over their common longitudinal range.

    Args:
        left_boundary_df: Fitted left boundary DataFrame ['s_m', 'x_m', 'y_m'].
        right_boundary_df: Fitted right boundary DataFrame ['s_m', 'x_m', 'y_m'].
        spline_samples: Number of sample evaluation points.

    Returns:
        DataFrame containing ['s_m', 'x_m', 'y_m'] or None if common range is unsupported.
    """
    if len(left_boundary_df) < 2 or len(right_boundary_df) < 2:
        return None

    s_left_min, s_left_max = left_boundary_df['s_m'].min(), left_boundary_df['s_m'].max()
    s_right_min, s_right_max = right_boundary_df['s_m'].min(), right_boundary_df['s_m'].max()

    s_comm_min = max(s_left_min, s_right_min)
    s_comm_max = min(s_left_max, s_right_max)

    if s_comm_max <= s_comm_min:
        return None

    s_grid = np.linspace(s_comm_min, s_comm_max, spline_samples)

    lx_interp = np.interp(s_grid, left_boundary_df['s_m'].values, left_boundary_df['x_m'].values)
    ly_interp = np.interp(s_grid, left_boundary_df['s_m'].values, left_boundary_df['y_m'].values)

    rx_interp = np.interp(s_grid, right_boundary_df['s_m'].values, right_boundary_df['x_m'].values)
    ry_interp = np.interp(s_grid, right_boundary_df['s_m'].values, right_boundary_df['y_m'].values)

    cx = (lx_interp + rx_interp) / 2.0
    cy = (ly_interp + ry_interp) / 2.0

    return pd.DataFrame({
        's_m': s_grid,
        'x_m': cx,
        'y_m': cy
    })


def reconstruct_track(
    landmarks_path: str = "outputs/ewmvc_landmarks.csv",
    telemetry_path: str = "data/raw/telemetry_log.csv",
    config_path: str = "configs/config.yaml",
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Main execution pipeline for EWMVC Step 20I Track Boundary Reconstruction.

    Loads consensus-supported landmarks and vehicle telemetry, computes trajectory-relative
    longitudinal and lateral coordinates, classifies sides, fits boundary curves, and generates
    CSV artifacts and visualizations.

    Returns:
        Summary statistics dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        recon_cfg = cfg.get('ewmvc', {}).get('track_reconstruction', {})
    else:
        recon_cfg = {}

    min_forward_dist = recon_cfg.get('min_forward_distance_m', 0.0)
    max_forward_dist = recon_cfg.get('max_forward_distance_m', 60.0)
    min_lateral_dist = recon_cfg.get('min_lateral_distance_m', 0.5)
    max_lateral_dist = recon_cfg.get('max_lateral_distance_m', 15.0)
    spline_samples = recon_cfg.get('spline_samples', 200)

    # 2. Load Data
    landmarks_df = pd.read_csv(landmarks_path)
    telemetry_df = pd.read_csv(telemetry_path)

    # Compute cumulative distance along telemetry trajectory
    dx = np.diff(telemetry_df['x'].values, prepend=telemetry_df['x'].values[0])
    dy = np.diff(telemetry_df['y'].values, prepend=telemetry_df['y'].values[0])
    s_traj = np.cumsum(np.sqrt(dx**2 + dy**2))

    # 3. Calculate trajectory-relative coordinates for every landmark
    track_landmarks_rows = []
    for idx_row, row in landmarks_df.iterrows():
        cid = row['candidate_id']
        lx = float(row['landmark_x'])
        ly = float(row['landmark_y'])
        score = float(row['consensus_score'])

        xv, yv, yaw, ts, pose_idx = find_closest_trajectory_pose(lx, ly, telemetry_df)
        s_rel, l_rel = compute_trajectory_relative_coordinates(lx, ly, xv, yv, yaw)
        side = classify_landmark_side(l_rel, min_lateral_dist)
        s_track = float(s_traj[pose_idx] + s_rel)

        track_landmarks_rows.append({
            'candidate_id': cid,
            'landmark_x': lx,
            'landmark_y': ly,
            'consensus_score': score,
            's_m': s_rel,
            'l_m': l_rel,
            'side': side,
            'trajectory_timestamp': ts,
            's_track': s_track
        })

    df_track_landmarks = pd.DataFrame(track_landmarks_rows)

    # Save outputs/ewmvc_track_landmarks.csv (excluding internal helper column s_track)
    export_cols = ['candidate_id', 'landmark_x', 'landmark_y', 'consensus_score', 's_m', 'l_m', 'side', 'trajectory_timestamp']
    track_landmarks_file = output_path / "ewmvc_track_landmarks.csv"
    df_track_landmarks[export_cols].to_csv(track_landmarks_file, index=False)

    # 4. Filter landmarks for track boundary reconstruction
    # Keep landmarks on left/right side within valid lateral bounds
    df_reconstruction_candidates = df_track_landmarks[
        (df_track_landmarks['side'].isin(['left', 'right'])) &
        (df_track_landmarks['l_m'].abs() <= max_lateral_dist)
    ].copy()

    df_left_pts = df_reconstruction_candidates[df_reconstruction_candidates['side'] == 'left'].copy()
    df_right_pts = df_reconstruction_candidates[df_reconstruction_candidates['side'] == 'right'].copy()

    # 5. Fit boundary curves
    df_left_boundary = fit_boundary_curve(df_left_pts, side_name='left', spline_samples=spline_samples)
    df_right_boundary = fit_boundary_curve(df_right_pts, side_name='right', spline_samples=spline_samples)

    left_boundary_file = output_path / "ewmvc_left_boundary.csv"
    right_boundary_file = output_path / "ewmvc_right_boundary.csv"

    df_left_boundary.to_csv(left_boundary_file, index=False)
    df_right_boundary.to_csv(right_boundary_file, index=False)

    # 6. Centerline calculation if supported
    df_centerline = compute_centerline(df_left_boundary, df_right_boundary, spline_samples=spline_samples)
    centerline_file = output_path / "ewmvc_centerline.csv"
    if df_centerline is not None:
        df_centerline.to_csv(centerline_file, index=False)

    # 7. Visualization 1: ewmvc_track_reconstruction.png
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot vehicle trajectory
    ax.plot(telemetry_df['x'], telemetry_df['y'], color='gray', linestyle='--', linewidth=1.5, label='Vehicle Trajectory', alpha=0.7)

    # Plot left landmarks (blue dots) & fitted left boundary (blue line)
    if not df_left_pts.empty:
        ax.scatter(df_left_pts['landmark_x'], df_left_pts['landmark_y'], color='blue', s=40, zorder=5, label='Left Landmarks (Blue)')
    if not df_left_boundary.empty:
        ax.plot(df_left_boundary['x_m'], df_left_boundary['y_m'], color='royalblue', linewidth=2.0, label='Left Boundary Curve')

    # Plot right landmarks (yellow/orange dots) & fitted right boundary (orange line)
    if not df_right_pts.empty:
        ax.scatter(df_right_pts['landmark_x'], df_right_pts['landmark_y'], color='gold', edgecolor='darkorange', s=40, zorder=5, label='Right Landmarks (Yellow)')
    if not df_right_boundary.empty:
        ax.plot(df_right_boundary['x_m'], df_right_boundary['y_m'], color='darkorange', linewidth=2.0, label='Right Boundary Curve')

    # Plot centerline if available
    if df_centerline is not None:
        ax.plot(df_centerline['x_m'], df_centerline['y_m'], color='green', linestyle=':', linewidth=1.8, label='Track Centerline')

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Global X Position [m]')
    ax.set_ylabel('Global Y Position [m]')
    ax.set_title("EWMVC Track Boundary Reconstruction", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    recon_plot_file = output_path / "ewmvc_track_reconstruction.png"
    plt.savefig(recon_plot_file, dpi=300)
    plt.close()

    # 8. Visualization 2: ewmvc_track_coordinates.png
    fig, ax = plt.subplots(figsize=(10, 6))

    left_mask = df_track_landmarks['side'] == 'left'
    right_mask = df_track_landmarks['side'] == 'right'
    ambig_mask = df_track_landmarks['side'] == 'ambiguous'

    if left_mask.any():
        ax.scatter(df_track_landmarks.loc[left_mask, 's_m'], df_track_landmarks.loc[left_mask, 'l_m'], color='blue', s=35, label='Left Landmarks (l > 0)')
    if right_mask.any():
        ax.scatter(df_track_landmarks.loc[right_mask, 's_m'], df_track_landmarks.loc[right_mask, 'l_m'], color='gold', edgecolor='darkorange', s=35, label='Right Landmarks (l < 0)')
    if ambig_mask.any():
        ax.scatter(df_track_landmarks.loc[ambig_mask, 's_m'], df_track_landmarks.loc[ambig_mask, 'l_m'], color='red', marker='x', s=50, label='Ambiguous Landmarks')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.7, label='Centerline Reference (l = 0)')
    ax.axhline(min_lateral_dist, color='blue', linestyle=':', alpha=0.5, label=f'+{min_lateral_dist}m Threshold')
    ax.axhline(-min_lateral_dist, color='orange', linestyle=':', alpha=0.5, label=f'-{min_lateral_dist}m Threshold')

    ax.set_xlabel('Relative Longitudinal Coordinate s [m]')
    ax.set_ylabel('Signed Lateral Coordinate l [m]')
    ax.set_title("EWMVC Track Coordinates (s vs l)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    coord_plot_file = output_path / "ewmvc_track_coordinates.png"
    plt.savefig(coord_plot_file, dpi=300)
    plt.close()

    # 9. Build Summary Statistics
    total_landmarks = len(landmarks_df)
    landmarks_used = len(df_reconstruction_candidates)
    left_count = len(df_left_pts)
    right_count = len(df_right_pts)
    ambig_count = int(ambig_mask.sum())

    left_s_min = float(df_left_boundary['s_m'].min()) if not df_left_boundary.empty else None
    left_s_max = float(df_left_boundary['s_m'].max()) if not df_left_boundary.empty else None
    right_s_min = float(df_right_boundary['s_m'].min()) if not df_right_boundary.empty else None
    right_s_max = float(df_right_boundary['s_m'].max()) if not df_right_boundary.empty else None

    common_s_range = None
    if df_centerline is not None and not df_centerline.empty:
        common_s_range = (float(df_centerline['s_m'].min()), float(df_centerline['s_m'].max()))

    summary = {
        'total_landmarks': total_landmarks,
        'landmarks_used': landmarks_used,
        'left_count': left_count,
        'right_count': right_count,
        'ambiguous_count': ambig_count,
        'left_s_min': left_s_min,
        'left_s_max': left_s_max,
        'left_landmark_count': left_count,
        'right_s_min': right_s_min,
        'right_s_max': right_s_max,
        'right_landmark_count': right_count,
        'common_centerline_range': common_s_range,
        'spline_samples': spline_samples
    }

    # Print summary CLI report
    print("=" * 70)
    print("  EWMVC STEP 20I: TRACK BOUNDARY RECONSTRUCTION")
    print("=" * 70)
    print(f"Total EWMVC Landmarks Loaded         : {total_landmarks}")
    print(f"Landmarks Used for Reconstruction    : {landmarks_used}")
    print(f"  - Left Landmarks (l > 0)          : {left_count}")
    print(f"  - Right Landmarks (l < 0)         : {right_count}")
    print(f"  - Ambiguous Landmarks (|l| < 0.5m): {ambig_count}")
    print("\n--- Fitted Boundary Curve Statistics ---")
    if left_s_min is not None:
        print(f"Left Boundary  : s in [{left_s_min:.4f} m, {left_s_max:.4f} m], {left_count} landmarks, {len(df_left_boundary)} samples")
    else:
        print("Left Boundary  : No curve fitted")
    if right_s_min is not None:
        print(f"Right Boundary : s in [{right_s_min:.4f} m, {right_s_max:.4f} m], {right_count} landmarks, {len(df_right_boundary)} samples")
    else:
        print("Right Boundary : No curve fitted")
    if common_s_range is not None:
        print(f"Centerline     : Common s range [{common_s_range[0]:.4f} m, {common_s_range[1]:.4f} m]")
    else:
        print("Centerline     : Unsupported / No common longitudinal range")

    print("\n--- Output Artifacts Generated ---")
    print(f"Saved Track Landmarks CSV            : '{track_landmarks_file}'")
    print(f"Saved Left Boundary CSV              : '{left_boundary_file}'")
    print(f"Saved Right Boundary CSV             : '{right_boundary_file}'")
    if df_centerline is not None:
        print(f"Saved Centerline CSV                 : '{centerline_file}'")
    print(f"Saved Track Reconstruction PNG       : '{recon_plot_file}'")
    print(f"Saved Track Coordinates PNG          : '{coord_plot_file}'")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    reconstruct_track()
