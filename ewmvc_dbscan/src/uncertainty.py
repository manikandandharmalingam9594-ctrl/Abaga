"""
EWMVC Uncertainty Propagation Module.

Propagates 2D perception sensor observation noise and vehicle pose uncertainty
into a 2D global spatial covariance matrix (cov_xx, cov_xy, cov_yy) for each observation.

NOTE ON DATASET UNCERTAINTY PARAMETERS:
The dataset does NOT provide measured sensor covariance or vehicle pose covariance matrices.
Therefore, all uncertainty parameters (sigma_sensor_x, sigma_sensor_y, sigma_pose_x,
sigma_pose_y, sigma_pose_yaw) are CONFIGURABLE DESIGN ASSUMPTIONS rather than measured dataset values.
"""

import os
from typing import Dict, Tuple, Union
import yaml
import numpy as np
import pandas as pd

from src.data_loader import load_logs, interpolate_telemetry
from src.transforms import add_global_coordinates


def compute_observation_covariance(
    rel_x_vehicle: Union[float, np.ndarray, pd.Series],
    rel_y_vehicle: Union[float, np.ndarray, pd.Series],
    vehicle_yaw: Union[float, np.ndarray, pd.Series],
    sigma_sensor_x: float = 0.10,
    sigma_sensor_y: float = 0.10,
    sigma_pose_x: float = 0.05,
    sigma_pose_y: float = 0.05,
    sigma_pose_yaw: float = 0.0174533,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D global spatial covariance matrix components (cov_xx, cov_xy, cov_yy).

    Kinematic Transformation:
        p_global = p_vehicle_pose + R(yaw) * p_vehicle_frame

    Jacobian Formulation:
        J_sensor = [[cos(yaw), -sin(yaw)],
                    [sin(yaw),  cos(yaw)]]

        J_pose   = [[1, 0, -sin(yaw)*x_v - cos(yaw)*y_v],
                    [0, 1,  cos(yaw)*x_v - sin(yaw)*y_v]]

        Sigma_global = J_pose * Sigma_pose * J_pose^T + J_sensor * Sigma_sensor * J_sensor^T

    Args:
        rel_x_vehicle: Relative X coordinate in vehicle frame (m).
        rel_y_vehicle: Relative Y coordinate in vehicle frame (m).
        vehicle_yaw: Vehicle heading yaw angle in radians.
        sigma_sensor_x: Sensor relative X standard deviation assumption (m).
        sigma_sensor_y: Sensor relative Y standard deviation assumption (m).
        sigma_pose_x: Vehicle pose X standard deviation assumption (m).
        sigma_pose_y: Vehicle pose Y standard deviation assumption (m).
        sigma_pose_yaw: Vehicle heading yaw standard deviation assumption (rad).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (cov_xx, cov_xy, cov_yy) arrays.

    Raises:
        ValueError: If any sigma parameter is negative.
    """
    if any(s < 0 for s in [sigma_sensor_x, sigma_sensor_y, sigma_pose_x, sigma_pose_y, sigma_pose_yaw]):
        raise ValueError("Uncertainty standard deviation parameters must be non-negative.")

    x_v = np.asarray(rel_x_vehicle)
    y_v = np.asarray(rel_y_vehicle)
    yaw = np.asarray(vehicle_yaw)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Pose Jacobian 3rd column components: a = dx/dyaw, b = dy/dyaw
    a = -sin_yaw * x_v - cos_yaw * y_v
    b = cos_yaw * x_v - sin_yaw * y_v

    var_sx = sigma_sensor_x ** 2
    var_sy = sigma_sensor_y ** 2
    var_px = sigma_pose_x ** 2
    var_py = sigma_pose_y ** 2
    var_pyaw = sigma_pose_yaw ** 2

    # J_sensor * Sigma_sensor * J_sensor^T
    cov_sensor_xx = (cos_yaw ** 2) * var_sx + (sin_yaw ** 2) * var_sy
    cov_sensor_yy = (sin_yaw ** 2) * var_sx + (cos_yaw ** 2) * var_sy
    cov_sensor_xy = sin_yaw * cos_yaw * (var_sx - var_sy)

    # J_pose * Sigma_pose * J_pose^T
    cov_pose_xx = var_px + (a ** 2) * var_pyaw
    cov_pose_yy = var_py + (b ** 2) * var_pyaw
    cov_pose_xy = a * b * var_pyaw

    # Total propagated global 2D spatial covariance matrix elements
    cov_xx = cov_pose_xx + cov_sensor_xx
    cov_yy = cov_pose_yy + cov_sensor_yy
    cov_xy = cov_pose_xy + cov_sensor_xy

    return cov_xx, cov_xy, cov_yy


def add_uncertainty_propagation(
    data: pd.DataFrame,
    sigma_sensor_x: float = 0.10,
    sigma_sensor_y: float = 0.10,
    sigma_pose_x: float = 0.05,
    sigma_pose_y: float = 0.05,
    sigma_pose_yaw: float = 0.0174533,
) -> pd.DataFrame:
    """
    Augment perception DataFrame with propagated 2D spatial covariance columns.

    Added Columns:
        - cov_xx
        - cov_xy
        - cov_yy
        - sigma_x
        - sigma_y

    Args:
        data (pd.DataFrame): Transformed perception DataFrame containing
                             ('rel_x_vehicle', 'rel_y_vehicle', 'vehicle_yaw').
        sigma_sensor_x (float): Sensor X standard deviation design assumption (m).
        sigma_sensor_y (float): Sensor Y standard deviation design assumption (m).
        sigma_pose_x (float): Vehicle pose X standard deviation design assumption (m).
        sigma_pose_y (float): Vehicle pose Y standard deviation design assumption (m).
        sigma_pose_yaw (float): Vehicle yaw standard deviation design assumption (rad).

    Returns:
        pd.DataFrame: Copy of input DataFrame with covariance columns appended.
    """
    required = ["rel_x_vehicle", "rel_y_vehicle", "vehicle_yaw"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required vehicle columns: {missing}")

    df = data.copy()

    cov_xx, cov_xy, cov_yy = compute_observation_covariance(
        df["rel_x_vehicle"].values,
        df["rel_y_vehicle"].values,
        df["vehicle_yaw"].values,
        sigma_sensor_x=sigma_sensor_x,
        sigma_sensor_y=sigma_sensor_y,
        sigma_pose_x=sigma_pose_x,
        sigma_pose_y=sigma_pose_y,
        sigma_pose_yaw=sigma_pose_yaw,
    )

    df["cov_xx"] = cov_xx
    df["cov_xy"] = cov_xy
    df["cov_yy"] = cov_yy
    df["sigma_x"] = np.sqrt(cov_xx)
    df["sigma_y"] = np.sqrt(cov_yy)

    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    p_rel = config["data"]["perception_file"]
    t_rel = config["data"]["telemetry_file"]
    rot_deg = float(config["sensor"]["rotation_deg"])
    out_dir_rel = config["output"]["directory"]

    unc_cfg = config["ewmvc"]["uncertainty"]
    s_sx = float(unc_cfg["sigma_sensor_x"])
    s_sy = float(unc_cfg["sigma_sensor_y"])
    s_px = float(unc_cfg["sigma_pose_x"])
    s_py = float(unc_cfg["sigma_pose_y"])
    s_pyaw = float(unc_cfg["sigma_pose_yaw"])

    p_path = os.path.join(base_dir, p_rel)
    t_path = os.path.join(base_dir, t_rel)
    out_dir = os.path.join(base_dir, out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)

    print("======================================================================")
    print("  EWMVC STEP 20A: UNCERTAINTY PROPAGATION PREPARATION")
    print("======================================================================")

    # 1. Load & Synchronize
    perception, telemetry = load_logs(p_path, t_path)
    perc_interp = interpolate_telemetry(perception, telemetry)

    # 2. Transform coordinates
    df_global = add_global_coordinates(perc_interp, rotation_deg=rot_deg)

    # 3. Propagate 2D spatial covariance
    df_unc = add_uncertainty_propagation(
        df_global,
        sigma_sensor_x=s_sx,
        sigma_sensor_y=s_sy,
        sigma_pose_x=s_px,
        sigma_pose_y=s_py,
        sigma_pose_yaw=s_pyaw,
    )

    # 4. Save output
    out_csv = os.path.join(out_dir, "ewmvc_observations_with_covariance.csv")
    df_unc.to_csv(out_csv, index=False)

    print(f"Loaded observations             : {len(df_unc)}")
    print(f"Propagated Covariance File Saved : '{out_csv}'")

    # Diagnostics
    nan_cov_count = df_unc[["cov_xx", "cov_xy", "cov_yy"]].isna().sum().sum()
    pos_diag_valid = (df_unc["cov_xx"] > 0).all() and (df_unc["cov_yy"] > 0).all()

    print(f"\n--- Validation Checks ---")
    print(f"NaN Covariance Values Count     : {nan_cov_count}")
    print(f"Positive Diagonal Covariance    : {pos_diag_valid}")
    print(f"Mean sigma_x [m]                : {df_unc['sigma_x'].mean():.4f}")
    print(f"Mean sigma_y [m]                : {df_unc['sigma_y'].mean():.4f}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
