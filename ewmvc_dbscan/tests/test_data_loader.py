"""
Unit tests for src/data_loader.py module.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_logs, interpolate_telemetry, wrap_angle


def test_load_logs_success():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perception_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    telemetry_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")

    perception, telemetry = load_logs(perception_path, telemetry_path)

    assert len(perception) == 9774, "Perception row count must equal 9774"
    assert len(telemetry) == 1200, "Telemetry row count must equal 1200"

    # Verify timestamps are sorted
    assert perception["timestamp"].is_monotonic_increasing, "Perception timestamps must be sorted"
    assert telemetry["timestamp"].is_monotonic_increasing, "Telemetry timestamps must be sorted"


def test_load_logs_missing_file():
    with pytest.raises(FileNotFoundError):
        load_logs("invalid/path/perception.csv", "invalid/path/telemetry.csv")


def test_load_logs_missing_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_file = os.path.join(tmpdir, "perc.csv")
        t_file = os.path.join(tmpdir, "telem.csv")

        # Invalid schema (missing 'confidence')
        pd.DataFrame({"timestamp": [0.0], "cone_type": ["left"]}).to_csv(p_file, index=False)
        pd.DataFrame({"timestamp": [0.0], "x": [0.0], "y": [0.0], "yaw_rad": [0.0],
                      "speed_mps": [0.0], "steering_rad": [0.0], "yaw_rate_rps": [0.0]}).to_csv(t_file, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_logs(p_file, t_file)


def test_load_logs_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_file = os.path.join(tmpdir, "perc_empty.csv")
        t_file = os.path.join(tmpdir, "telem_empty.csv")

        pd.DataFrame().to_csv(p_file, index=False)
        pd.DataFrame().to_csv(t_file, index=False)

        with pytest.raises(ValueError):
            load_logs(p_file, t_file)


def test_wrap_angle():
    angles = np.array([3.5, -3.5, 0.0, 2 * np.pi])
    wrapped = wrap_angle(angles)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped <= np.pi)
    np.testing.assert_allclose(wrapped[2], 0.0, atol=1e-6)
    np.testing.assert_allclose(wrapped[3], 0.0, atol=1e-6)


def test_interpolate_telemetry():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perception_path = os.path.join(base_dir, "data", "raw", "perception_log.csv")
    telemetry_path = os.path.join(base_dir, "data", "raw", "telemetry_log.csv")

    perception, telemetry = load_logs(perception_path, telemetry_path)
    perc_interp = interpolate_telemetry(perception, telemetry)

    # Verify rows preserved exactly (including duplicate timestamps)
    assert len(perc_interp) == 9774, "Interpolated perception row count must be 9774"
    assert "vehicle_x" in perc_interp.columns
    assert "vehicle_y" in perc_interp.columns
    assert "vehicle_yaw" in perc_interp.columns

    # Check vehicle_yaw is wrapped inside [-pi, pi]
    assert (perc_interp["vehicle_yaw"] >= -np.pi - 1e-6).all()
    assert (perc_interp["vehicle_yaw"] <= np.pi + 1e-6).all()
