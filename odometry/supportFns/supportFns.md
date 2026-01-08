# Support Functions Directory Documentation

This directory contains utility functions and classes used throughout the odometry module, particularly for coordinate management and geometric transformations.

## Files

### `coordinate_systems.py`
*   **`cartesian_to_polar`**: Converts a set of (x, y) cartesian points to (range, theta) polar coordinates.
*   **`polar_to_cartesian`**: Converts a set of (range, theta) polar coordinates to (x, y) cartesian points.

### `rotation_functions.py`
Provides a suite of functions for handling 2D rotations and rigid body transformations.
*   **`get_rot_matrix`**: Returns a 2x2 rotation matrix for a given angle.
*   **`get_angle_from_rot_matrix`**: Extracts the angle from a 2x2 rotation matrix.
*   **`apply_rot_trans`**: Applies a single rotation and translation to a point cloud.
*   **`apply_multiple_rot_trans`**: Applies N different rotations and translations to a single point cloud (broadcasting).
*   **`apply_unique_rot_trans_to_multiple_points`**: Applies N unique rotations/translations to N unique points (one-to-one mapping).
*   **`wrap_heading`**: Utility to wrap heading angles to be within [-pi, pi].

## Usage in Odometry Module

### `rotation_functions.py`
This module is ubiquitous for handling reference frames:
*   **Test Benches**:
    *   `odometry/test_benches/naive_radar_tb.py`
    *   `odometry/test_benches/odom_only_tb.py`
    *   `odometry/test_benches/radar_model_ekf_tb.py`
    *   `odometry/test_benches/radnav_stacked_pc_tb.py`
    *   `odometry/test_benches/radnav_vehicle_vel_estimator_tb.py`
    *   `odometry/test_benches/_test_bench.py`
*   **Localization**:
    *   Used heavily in `icp2D.py` and `particle_filter.py` (though not via script imports directly).
*   **Point Cloud Processing**:
    *   Used in `pc_stacker.py` and `temporal_pc_stacker.py`.

### `coordinate_systems.py`
*   **Test Benches**:
    *   `odometry/test_benches/naive_radar_tb.py`: Uses `cartesian_to_polar`.
    *   `odometry/test_benches/radar_model_ekf_tb.py`: Uses `cartesian_to_polar` and `polar_to_cartesian`.
    *   `odometry/test_benches/radnav_stacked_pc_tb.py`: Uses `cartesian_to_polar`.
