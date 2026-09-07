"""
Unit tests for EWMVC Track Boundary Reconstruction Module (src/track_recon.py).
"""

import math
import pytest
import numpy as np
import pandas as pd

from src.track_recon import (
    compute_trajectory_relative_coordinates,
    find_closest_trajectory_pose,
    classify_landmark_side,
    fit_boundary_curve,
    compute_centerline,
    reconstruct_track
)


def test_1_straight_trajectory_both_sides():
    """
    TEST 1: Straight trajectory with landmarks on both sides.
    Verify left landmarks have l > 0 and right landmarks have l < 0.
    """
    vx, vy, yaw = 0.0, 0.0, 0.0

    # Left landmarks
    s_left, l_left1 = compute_trajectory_relative_coordinates(5.0, 3.0, vx, vy, yaw)
    s_left2, l_left2 = compute_trajectory_relative_coordinates(15.0, 2.5, vx, vy, yaw)

    # Right landmarks
    s_right1, l_right1 = compute_trajectory_relative_coordinates(5.0, -3.0, vx, vy, yaw)
    s_right2, l_right2 = compute_trajectory_relative_coordinates(15.0, -2.5, vx, vy, yaw)

    assert l_left1 > 0.0
    assert l_left2 > 0.0
    assert l_right1 < 0.0
    assert l_right2 < 0.0

    assert classify_landmark_side(l_left1, min_lateral_distance_m=0.5) == 'left'
    assert classify_landmark_side(l_right1, min_lateral_distance_m=0.5) == 'right'


def test_2_known_coordinate_transformation():
    """
    TEST 2: Known coordinate transformation.
    For vehicle = (0,0), yaw = 0, landmark = (10, 3): Expected s = 10, l = 3.
    """
    s, l = compute_trajectory_relative_coordinates(10.0, 3.0, 0.0, 0.0, 0.0)
    assert math.isclose(s, 10.0, abs_tol=1e-6)
    assert math.isclose(l, 3.0, abs_tol=1e-6)


def test_3_yaw_ninety_degrees():
    """
    TEST 3: Yaw = pi / 2.
    For vehicle = (0,0), yaw = pi/2, landmark = (10, 3):
    r = (10, 3)
    s = 10*cos(pi/2) + 3*sin(pi/2) = 3
    l = -10*sin(pi/2) + 3*cos(pi/2) = -10
    """
    s, l = compute_trajectory_relative_coordinates(10.0, 3.0, 0.0, 0.0, math.pi / 2.0)
    assert math.isclose(s, 3.0, abs_tol=1e-6)
    assert math.isclose(l, -10.0, abs_tol=1e-6)


def test_4_landmark_behind_vehicle():
    """
    TEST 4: Landmark behind vehicle.
    For vehicle = (5, 0), yaw = 0, landmark = (2, 1): Expected s < 0.
    """
    s, l = compute_trajectory_relative_coordinates(2.0, 1.0, 5.0, 0.0, 0.0)
    assert s < 0.0
    assert math.isclose(s, -3.0, abs_tol=1e-6)
    assert math.isclose(l, 1.0, abs_tol=1e-6)


def test_5_ambiguous_landmark_near_centerline():
    """
    TEST 5: Ambiguous landmark near centerline.
    If |l| < min_lateral_distance_m (e.g., l = 0.2 < 0.5), expected side = 'ambiguous'.
    """
    side = classify_landmark_side(0.2, min_lateral_distance_m=0.5)
    assert side == 'ambiguous'

    side_neg = classify_landmark_side(-0.3, min_lateral_distance_m=0.5)
    assert side_neg == 'ambiguous'


def test_6_zero_landmarks_on_a_side():
    """
    TEST 6: Zero landmarks on a side.
    Expected: no spline is generated (empty DataFrame returned).
    """
    empty_df = pd.DataFrame(columns=['s_m', 'landmark_x', 'landmark_y'])
    curve_df = fit_boundary_curve(empty_df, side_name='left', spline_samples=200)

    assert curve_df.empty
    assert list(curve_df.columns) == ['s_m', 'x_m', 'y_m']


def test_7_two_landmarks_linear_interpolation():
    """
    TEST 7: Two landmarks.
    Expected: linear interpolation curve generated with specified spline_samples.
    """
    df_two = pd.DataFrame([
        {'s_m': 0.0, 'landmark_x': 0.0, 'landmark_y': 3.0},
        {'s_m': 10.0, 'landmark_x': 10.0, 'landmark_y': 3.0}
    ])

    curve_df = fit_boundary_curve(df_two, side_name='left', spline_samples=100)
    assert len(curve_df) == 100
    assert math.isclose(curve_df['s_m'].min(), 0.0, abs_tol=1e-6)
    assert math.isclose(curve_df['s_m'].max(), 10.0, abs_tol=1e-6)
    assert math.isclose(curve_df['y_m'].iloc[50], 3.0, abs_tol=1e-6)


def test_8_three_or_more_landmarks_spline():
    """
    TEST 8: Three or more landmarks.
    Verify cubic spline is generated and remains within the supported s range.
    """
    df_three = pd.DataFrame([
        {'s_m': 0.0, 'landmark_x': 0.0, 'landmark_y': 3.0},
        {'s_m': 5.0, 'landmark_x': 5.0, 'landmark_y': 3.5},
        {'s_m': 10.0, 'landmark_x': 10.0, 'landmark_y': 3.0}
    ])

    curve_df = fit_boundary_curve(df_three, side_name='left', spline_samples=200)
    assert len(curve_df) == 200
    assert not curve_df.isna().any().any()
    assert curve_df['s_m'].min() >= 0.0
    assert curve_df['s_m'].max() <= 10.0


def test_9_no_extrapolation():
    """
    TEST 9: No extrapolation.
    Verify generated boundary s values do not extend beyond the min and max landmark s values.
    """
    s_min_val, s_max_val = 2.5, 45.8
    df_pts = pd.DataFrame([
        {'s_m': s_min_val, 'landmark_x': 2.5, 'landmark_y': 3.0},
        {'s_m': 15.0, 'landmark_x': 15.0, 'landmark_y': 3.2},
        {'s_m': 30.0, 'landmark_x': 30.0, 'landmark_y': 2.9},
        {'s_m': s_max_val, 'landmark_x': s_max_val, 'landmark_y': 3.1}
    ])

    curve_df = fit_boundary_curve(df_pts, side_name='left', spline_samples=200)
    assert math.isclose(curve_df['s_m'].min(), s_min_val, abs_tol=1e-6)
    assert math.isclose(curve_df['s_m'].max(), s_max_val, abs_tol=1e-6)
    assert curve_df['s_m'].min() >= s_min_val
    assert curve_df['s_m'].max() <= s_max_val


def test_10_actual_dataset():
    """
    TEST 10: Actual dataset validation.
    Verify:
    - Every output landmark has a valid side ('left', 'right', or 'ambiguous')
    - No NaN s_m / l_m
    - Left/right outputs contain only actual EWMVC landmark candidate_ids
    - No new landmark IDs or synthetic cone coordinates created
    - No ground-truth label column used
    """
    summary = reconstruct_track(
        landmarks_path="outputs/ewmvc_landmarks.csv",
        telemetry_path="data/raw/telemetry_log.csv",
        config_path="configs/config.yaml",
        output_dir="outputs"
    )

    df_landmarks = pd.read_csv("outputs/ewmvc_landmarks.csv")
    df_track_landmarks = pd.read_csv("outputs/ewmvc_track_landmarks.csv")
    df_left = pd.read_csv("outputs/ewmvc_left_boundary.csv")
    df_right = pd.read_csv("outputs/ewmvc_right_boundary.csv")

    assert summary['total_landmarks'] == len(df_landmarks)
    assert len(df_track_landmarks) == len(df_landmarks)

    # Check side validity
    valid_sides = {'left', 'right', 'ambiguous'}
    assert set(df_track_landmarks['side']).issubset(valid_sides)

    # Check no NaN values
    assert not df_track_landmarks.isna().any().any()
    assert not df_left.isna().any().any()
    assert not df_right.isna().any().any()

    # Check candidate IDs match strictly without extra/new IDs
    assert set(df_track_landmarks['candidate_id']) == set(df_landmarks['candidate_id'])

    # Check no perception label column present
    assert 'label' not in df_track_landmarks.columns
    assert 'real' not in df_track_landmarks.columns
    assert 'ghost' not in df_track_landmarks.columns
