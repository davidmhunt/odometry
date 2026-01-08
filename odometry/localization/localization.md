# Localization Directory Documentation

This directory contains classes and functions for localizing the agent within a map, primarily using ICP (Iterative Closest Point) and Particle Filter (Monte Carlo Localization) techniques.

## Files

### `_localizer.py`
*   **`_Localizer`**: An abstract base class defining the interface for localization strategies. It handles loading maps from images, setting map limits, and managing the current odometry state.

### `icp2D.py`
*   **`icp2D`**: Implements the core 2D Iterative Closest Point algorithm.
    *   **`icp`**: The main method that iteratively aligns a source point cloud to a reference map to find the optimal rotation and translation.
    *   **`get_nearerest_points_percentile`**: Finds nearest neighbors between point clouds.
    *   **`compute_optimal_rot_trans`**: Solves for the rigid body transformation that minimizes the distance between matched points.

### `icp2D_localization.py`
*   **`icp2DLocalization`**: A comprehensive localization class that wraps `icp2D`.
    *   **`update_odometry`**: The primary interface for updating the agent's pose. It predicts the new pose (using odometry or kinematics) and then refines it using the ICP result against a global map.

### `particle_filter.py`
*   **`particleFilter`**: Implements 2D Monte Carlo Localization (MCL).
    *   Manages a set of particles representing the posterior distribution of the agent's pose.
    *   **`update_odometry`**: Propagates particles using a motion model.
    *   **`run_MCL_alg`**: Weights particles using a measurement model (likelihood field) and resamples them.

## Usage in Odometry Module

### `icp2D.py` / `icp2D_localization.py`
These files are the backbone of the current localization system and are used extensively:

*   **Test Benches (`odometry/test_benches/`)**:
    *   `lidar_icp_localization_tb.py`
    *   `naive_radar_tb.py`
    *   `odom_only_tb.py`
    *   `radar_model_ekf_tb.py`
    *   `radnav_stacked_pc_tb.py`
    *   `radnav_vehicle_vel_estimator_tb.py`
    *   `point_cloud_integrator_tb.py`
    *   `gnn_point_cloud_integrator_tb.py`
    *   `_test_bench.py`

*   **Scripts (`scripts/`)**:
    *   `radnav_analysis.py` / `radnav_analysis_navigation.py`
    *   `pc_integrator_analysis.py` / `pc_integrator_gnn_analysis.py`
    *   `pc_integrator_gnn_dataset_gen.py` / `pc_integrator_gnn_dataset_gen_uav.py`
    *   `naive_radar_analysis.py`
    *   `radarHD_analysis.py`
    *   `radcloud_analysis.py`

### `_localizer.py`
*   **Test Benches**:
    *   `odometry/test_benches/_test_bench.py`: Used as a base for setting up localization test environments.

### `particle_filter.py`
*   **Test Benches**:
    *   (No direct usage found in current active test benches)
*   **Scripts**:
    *   (No direct usage found in current active scripts)
