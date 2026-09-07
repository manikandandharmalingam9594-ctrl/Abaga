"""
Data loader module for fixed-epsilon DBSCAN cone mapping baseline and EWMVC.

Handles loading CSV logs, column schema validation, sorting, and linear interpolation
of vehicle telemetry (position and unwrapped yaw) onto perception timestamps.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd


# Required column schemas
REQUIRED_PERCEPTION_COLS = [
    "timestamp",
    "cone_type",
    "rel_x_sensor",
    "rel_y_sensor",
    "confidence",
    "label",
]

REQUIRED_TELEMETRY_COLS = [
    "timestamp",
    "x",
    "y",
    "yaw_rad",
    "speed_mps",
    "steering_rad",
    "yaw_rate_rps",
]


def load_logs(perception_path: str, telemetry_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load perception and telemetry CSV logs, validate schemas, sort by timestamp,
    and reset DataFrame indexes.

    Args:
        perception_path (str): File path to perception CSV log.
        telemetry_path (str): File path to telemetry CSV log.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Validated and sorted (perception, telemetry) DataFrames.

    Raises:
        FileNotFoundError: If perception or telemetry file paths do not exist.
        ValueError: If required columns are missing in either dataset or files are empty.
    """
    if not os.path.exists(perception_path):
        raise FileNotFoundError(f"Perception log file not found at: '{perception_path}'")
    if not os.path.exists(telemetry_path):
        raise FileNotFoundError(f"Telemetry log file not found at: '{telemetry_path}'")

    perception = pd.read_csv(perception_path)
    telemetry = pd.read_csv(telemetry_path)

    if perception.empty:
        raise ValueError(f"Perception log at '{perception_path}' is empty.")
    if telemetry.empty:
        raise ValueError(f"Telemetry log at '{telemetry_path}' is empty.")

    # Validate perception log schema
    missing_perception = [col for col in REQUIRED_PERCEPTION_COLS if col not in perception.columns]
    if missing_perception:
        raise ValueError(
            f"Perception log at '{perception_path}' is missing required columns: {missing_perception}"
        )

    # Validate telemetry log schema
    missing_telemetry = [col for col in REQUIRED_TELEMETRY_COLS if col not in telemetry.columns]
    if missing_telemetry:
        raise ValueError(
            f"Telemetry log at '{telemetry_path}' is missing required columns: {missing_telemetry}"
        )

    # Sort datasets by timestamp and reset indexes
    perception = perception.sort_values(by="timestamp").reset_index(drop=True)
    telemetry = telemetry.sort_values(by="timestamp").reset_index(drop=True)

    return perception, telemetry


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angle in radians to the interval [-pi, pi].

    Args:
        angle (np.ndarray): Angle in radians.

    Returns:
        np.ndarray: Wrapped angle in range [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def interpolate_telemetry(perception: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate vehicle telemetry (x, y, yaw) onto exact perception timestamps.

    Yaw is unwrapped prior to linear interpolation to prevent artificial discontinuities 
    at the -pi/pi boundary, and wrapped back to [-pi, pi] post-interpolation.

    Multiple perception observations at the same timestamp are preserved.

    Args:
        perception (pd.DataFrame): Perception DataFrame sorted by timestamp.
        telemetry (pd.DataFrame): Telemetry DataFrame sorted by timestamp.

    Returns:
        pd.DataFrame: Augmented perception DataFrame copy containing:
                      'vehicle_x', 'vehicle_y', 'vehicle_yaw'.
    """
    perc_df = perception.copy()

    target_ts = perc_df["timestamp"].values
    source_ts = telemetry["timestamp"].values

    # Linear interpolation for 2D position coordinates
    perc_df["vehicle_x"] = np.interp(target_ts, source_ts, telemetry["x"].values)
    perc_df["vehicle_y"] = np.interp(target_ts, source_ts, telemetry["y"].values)

    # Unwrap yaw angles to avoid artificial jumps (e.g. -pi to +pi wrap-around) during interpolation
    unwrapped_yaw = np.unwrap(telemetry["yaw_rad"].values)
    interp_unwrapped_yaw = np.interp(target_ts, source_ts, unwrapped_yaw)

    # Wrap interpolated yaw values back to [-pi, pi]
    perc_df["vehicle_yaw"] = wrap_angle(interp_unwrapped_yaw)

    return perc_df
