"""
Coordinate transformation module for fixed-epsilon DBSCAN cone-mapping baseline and EWMVC.

Coordinate Frames:
1. Sensor Frame (rel_x_sensor, rel_y_sensor):
   - Relative coordinates measured by perception sensor mounted on the vehicle.
   - Rotated by configurable mounting angle (default 180 deg) to align with vehicle frame.

2. Vehicle Frame (rel_x_vehicle, rel_y_vehicle):
   - Local vehicle coordinate frame (X forward, Y left).

3. Global World Frame (global_x, global_y):
   - Fixed map coordinate frame.
   - Transformed from vehicle frame using SE(2) rigid body kinematics:
       global_x = vehicle_x + cos(yaw) * x_vehicle - sin(yaw) * y_vehicle
       global_y = vehicle_y + sin(yaw) * x_vehicle + cos(yaw) * y_vehicle
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def rotate_points(
    points_xy: np.ndarray, angle_rad: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Rotate 2D points (x, y) around origin (0, 0) by angle_rad radians.

    Args:
        points_xy (np.ndarray): Array of shape (N, 2) or (2,) containing (x, y) coordinates.
        angle_rad (Union[float, np.ndarray]): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated 2D points of shape (N, 2) or (2,).
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if points_xy.ndim == 1:
        x, y = points_xy[0], points_xy[1]
        x_rot = cos_a * x - sin_a * y
        y_rot = sin_a * x + cos_a * y
        return np.array([x_rot, y_rot])

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return np.column_stack((x_rot, y_rot))


def sensor_to_vehicle(
    rel_x: Union[np.ndarray, pd.Series],
    rel_y: Union[np.ndarray, pd.Series],
    rotation_deg: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform sensor-frame relative coordinates to vehicle-frame coordinates.

    Args:
        rel_x (Union[np.ndarray, pd.Series]): Relative x coordinates in sensor frame.
        rel_y (Union[np.ndarray, pd.Series]): Relative y coordinates in sensor frame.
        rotation_deg (float): Sensor mounting rotation angle in degrees (default 180.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (rel_x_vehicle, rel_y_vehicle) in vehicle frame.
    """
    angle_rad = np.radians(rotation_deg)
    points_sensor = np.column_stack((np.asarray(rel_x), np.asarray(rel_y)))
    points_vehicle = rotate_points(points_sensor, angle_rad)
    return points_vehicle[:, 0], points_vehicle[:, 1]


def vehicle_to_global(
    vehicle_points: np.ndarray,
    vehicle_x: Union[float, np.ndarray, pd.Series],
    vehicle_y: Union[float, np.ndarray, pd.Series],
    vehicle_yaw: Union[float, np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform vehicle-frame coordinates to global world map coordinates using SE(2).

    SE(2) Kinematic Equations:
        global_x = vehicle_x + cos(yaw) * x_vehicle - sin(yaw) * y_vehicle
        global_y = vehicle_y + sin(yaw) * x_vehicle + cos(yaw) * y_vehicle

    Args:
        vehicle_points (np.ndarray): (N, 2) array of (x_vehicle, y_vehicle) points.
        vehicle_x (Union[float, np.ndarray, pd.Series]): Vehicle global X position.
        vehicle_y (Union[float, np.ndarray, pd.Series]): Vehicle global Y position.
        vehicle_yaw (Union[float, np.ndarray, pd.Series]): Vehicle heading yaw angle in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (global_x, global_y) coordinates in global frame.
    """
    x_veh = vehicle_points[:, 0]
    y_veh = vehicle_points[:, 1]

    cos_yaw = np.cos(vehicle_yaw)
    sin_yaw = np.sin(vehicle_yaw)

    global_x = vehicle_x + cos_yaw * x_veh - sin_yaw * y_veh
    global_y = vehicle_y + sin_yaw * x_veh + cos_yaw * y_veh

    return np.asarray(global_x), np.asarray(global_y)


def add_global_coordinates(
    data: pd.DataFrame, rotation_deg: float = 180.0
) -> pd.DataFrame:
    """
    Augment perception DataFrame with vehicle-frame and global world coordinates.

    Applies sensor-to-vehicle rotation and vehicle-to-global SE(2) transformation.

    Added Columns:
        - rel_x_vehicle
        - rel_y_vehicle
        - global_x
        - global_y

    Args:
        data (pd.DataFrame): Perception DataFrame containing ('rel_x_sensor', 'rel_y_sensor',
                             'vehicle_x', 'vehicle_y', 'vehicle_yaw').
        rotation_deg (float): Sensor mounting rotation in degrees (default 180.0).

    Returns:
        pd.DataFrame: Augmented DataFrame copy containing vehicle and global coordinates.
    """
    required = ["rel_x_sensor", "rel_y_sensor", "vehicle_x", "vehicle_y", "vehicle_yaw"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns for transformation: {missing}")

    df = data.copy()

    # 1. Sensor -> Vehicle frame transformation
    x_veh, y_veh = sensor_to_vehicle(
        df["rel_x_sensor"].values, df["rel_y_sensor"].values, rotation_deg
    )
    df["rel_x_vehicle"] = x_veh
    df["rel_y_vehicle"] = y_veh

    # 2. Vehicle -> Global frame SE(2) transformation
    veh_points = np.column_stack((x_veh, y_veh))
    gx, gy = vehicle_to_global(
        veh_points,
        df["vehicle_x"].values,
        df["vehicle_y"].values,
        df["vehicle_yaw"].values,
    )
    df["global_x"] = gx
    df["global_y"] = gy

    return df
