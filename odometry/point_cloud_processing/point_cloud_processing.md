# Point Cloud Processing Directory Documentation

This directory contains advanced processing algorithms for point clouds, including stacking, multipath removal, velocity estimation, and integration with grids.

## Files

### `_point_cloud_integrator.py`
*   **`_PointCloudIntegrator`**: Wrapper class that integrates new points into `ProbabilisticPCGrid` and optionally a `HistoricalPCGrid`. It handles detection filtering and transforming points based on pose updates.

### `multipath.py`
*   **`MultiPath`**: Algorithms to identify and remove multipath reflections (ghost detections).
    *   Uses ray tracing logic and DBSCAN clustering.
    *   **`remove_multipath`**: Main method to filter a point cloud by removing clusters that appear "behind" other valid clusters along a ray.

### `pc_range_filter.py`
*   **`pcRangeFilter`**: Simple utility to filter points based on minimum and maximum distance from the sensor (e.g., removing self-reflections or far noise).

### `pc_stacker.py`
*   **`pcStacker`**: A simpler stacking class that accumulates points into a grid (`point_cloud_grid`) over time.
    *   Quantizes the space into bins.
    *   Handles recentering of the grid as the vehicle moves.
    *   **`add_points`**: Transforms points to initial frame and adds to grid.

### `temporal_pc_stacker.py`
*   **`temporalPcStacker`**: A comprehensive class for temporal integration of point clouds.
    *   Combines `InertialIntegrator`, `MultiPath`, and `VelFiltering`.
    *   Maintains static and dynamic point cloud grids with history (`pc_grid_static`, `pc_grid_dynamic`).
    *   **`refresh`**: Recenters grids, removes dynamic objects, removes multipath, and updates the latest usable point cloud.
    *   **`predict`**: Updates internal state using inertial data between point cloud frames.

### `vehicle_vel_estimator.py`
*   **`VehicleVelEstimator`**: Estimates the ego-velocity of the vehicle using static background points.
    *   **`estimate_ego_vel`**: Uses RANSAC and least squares (`lsq_fit_2D`) to find the velocity vector that best explains the Doppler/velocity measurements of static points.

### `vel_filtering.py`
*   **`VelFiltering`**: Logic to separate static and dynamic objects.
    *   **`get_dynamic_detections`** / **`get_static_detections`**: classifies points based on consistency with ego-velocity.
    *   **`remove_dynamic_clusters_from_static_detections_knn`**: Refines static classification by removing points close to dynamic clusters (using DBSCAN + KNN).

## Usage in Odometry Module

### `temporal_pc_stacker.py`
*   **Test Benches**:
    *   `odometry/test_benches/radar_model_ekf_tb.py`
    *   `odometry/test_benches/radnav_stacked_pc_tb.py`
    *   `odometry/test_benches/radnav_vehicle_vel_estimator_tb.py`
    *   `odometry/test_benches/_test_bench.py`
*   **Scripts**:
    *   `scripts/radnav_analysis.py`
    *   `scripts/radnav_analysis_navigation.py`

### `_point_cloud_integrator.py`
*   **Test Benches**:
    *   `odometry/test_benches/gnn_point_cloud_integrator_tb.py`
    *   `odometry/test_benches/point_cloud_integrator_tb.py`
*   **Scripts**:
    *   `scripts/pc_integrator_analysis.py`
    *   `scripts/pc_integrator_gnn_analysis.py`
    *   `scripts/pc_integrator_gnn_dataset_gen.py`
    *   `scripts/pc_integrator_gnn_dataset_gen_uav.py`

### `vehicle_vel_estimator.py`
*   **Test Benches**:
    *   `odometry/test_benches/radnav_vehicle_vel_estimator_tb.py`

### `multipath.py`
*   **Test Benches**:
    *   `odometry/test_benches/radar_model_ekf_tb.py`
    *   `odometry/test_benches/radnav_stacked_pc_tb.py`
    *   `odometry/test_benches/_test_bench.py`

### `vel_filtering.py`
*   Used internally by `temporal_pc_stacker.py` and `pc_integrator_*` scripts (indirectly).
