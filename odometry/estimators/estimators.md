# Estimators Directory Documentation

This directory contains classes and functions for state estimation, including Kalman Filters and Motion Models.

## Files

### `estimators.py`
This file defines the core estimation logic using Kalman Filters and Inertial Integration.

*   **`Inertial`**: A data class to hold inertial measurements (gyro, accel, wheel encoder velocity).
*   **`_ExtendedKalmanFilter`**: Base class for an Extended Kalman Filter (EKF), handling the predict and update steps.
*   **`_KalmanXYPhiSpeed`**: An intermediate class defining the measurement function `h_func` and `H` matrix for a state vector `[x, y, phi, speed]`.
*   **`KalmanXYPhiSpeedKinematic`**: A concrete EKF implementation using a kinematic motion model for state propagation.
*   **`KalmanXYPhiSpeedGyroEncoder`**: A concrete EKF implementation using gyro and wheel encoder data for propagation. State vector: `[x, y, phi, speed, gyro_bias, encoder_bias]`.
*   **`InertialIntegrator`**: A class to perform state prediction (dead reckoning) using inertial data without any measurement updates.

### `motion_models.py`
This file defines motion models, primarily for use in Particle Filters or other sampling-based estimators.

*   **`MotionModel`**: Base abstract class for motion models.
*   **`InertialIntegratorMM`**: A motion model wrapper around inertial integration principles, maintaining covariance `P`.
*   **`GyroEncoderIntegratorMM`**: A specific motion model using Gyro and Encoder data (similar to `KalmanXYPhiSpeedGyroEncoder` logic but as a `MotionModel`).
*   **`OdometryMM`**: A probabilistic odometry motion model (sample_motion_model_odometry) often used in particle filters.

## Usage in Odometry Module

### `estimators.py`
*   **Test Benches**:
    *   `odometry/test_benches/naive_radar_tb.py`
    *   `odometry/test_benches/odom_only_tb.py`
    *   `odometry/test_benches/radar_model_ekf_tb.py`
    *   `odometry/test_benches/radnav_stacked_pc_tb.py`
    *   `odometry/test_benches/radnav_vehicle_vel_estimator_tb.py`
    *   `odometry/test_benches/_test_bench.py`
*   **Scripts**:
    *   (No direct usage found in top-level scripts, primarily used via `point_cloud_processing` or `localization` modules)
*   **Other Modules**:
    *   `odometry/point_cloud_processing/temporal_pc_stacker.py`
    *   `odometry/localization/particle_filter.py`

### `motion_models.py`
*   **Test Benches**:
    *   (No direct usage found in test benches)
*   **Scripts**:
    *   (No direct usage found in scripts)
*   **Other Modules**:
    *   `odometry/localization/particle_filter.py` (Imported as `MotionModel`)
