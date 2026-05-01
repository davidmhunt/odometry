import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from tqdm import tqdm
import torch

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from scipy.spatial import cKDTree

def load_eval_module(file_path):
    spec = importlib.util.spec_from_file_location("eval_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    # Import dependencies needed for setup
    from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator import TemporalDensityPCIntegrator
    from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
    from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
    from odometry.test_benches._test_bench import PredictionSource, GroundTruthSource, OdomCoordinateFrame
    from odometry.localization.icp2D_localization import icp2DLocalization

    # Platform Toggle
    PLATFORM = "ugv" # "uav" or "ugv"
    
    # Parameters for evaluation
    if PLATFORM == "uav":
        DATASET_PATH = "/data/IcaRAus/datasets/UAV/Radar_datasets"
        MAP_DIRECTORY = "/data/IcaRAus/maps"
        folder_name = "vicon_box"
        file_name = "vicon_box_1"
        map_file = "north_vicon_1.yaml"
        vicon_folder = "vicon_x500_8"
        odom_frame = OdomCoordinateFrame.NED
        gt_source = GroundTruthSource.MOTION_CAPTURE
        best_points_percentile = 85
        radar_clustering_eps = 0.35
        gt_subsample_percentage = 0.75
        gt_distance_threshold_m = 0.4
        gt_angle_res_rad = 0.051

    else: # ugv
        DATASET_PATH = "/data/IcaRAus/datasets/UGV"
        MAP_DIRECTORY = "/data/IcaRAus/maps"
        folder_name = "WILK"
        file_name = "IcaRAus_ugv_wilk_1_5m"
        map_file = "wilk_map.yaml"
        vicon_folder = None
        odom_frame = OdomCoordinateFrame.FLU
        gt_source = GroundTruthSource.LIDAR
        best_points_percentile = 85
        radar_clustering_eps = 0.25    
        gt_subsample_percentage = 0.5
        gt_distance_threshold_m = 0.4
        gt_angle_res_rad = 0.034
    
    # Initialize Dataset
    ds_params = {
        "dataset_path": os.path.join(DATASET_PATH, folder_name, file_name),
        "radar_pc_folder": "radar_combined_pc",
        "lidar_folder": "lidar",
        "camera_folder": "camera",
        "vehicle_odom_folder": "vehicle_odom",
        "imu_orientation_folder": "imu_data",
        "imu_full_folder": "imu_data_full",
        "vehicle_vel_folder": "vehicle_vel"
    }
    if vicon_folder:
        ds_params["vicon_folder"] = vicon_folder
        
    dataset = CpslDS(**ds_params)
    
    # Initialize Map
    map_handler = MapHandler(
        maps_folder=MAP_DIRECTORY,
        map_file=map_file
    )
    
    # Initialize Localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=best_points_percentile,
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=7,
        icp_max_iterations=5,
        self_detection_radius_m=0
    )

    lidar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.1,
        icp_best_points_percentile=50,
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=10,
        icp_max_iterations=5,
        self_detection_radius_m=1.0
    )
    
    # Integrator Setup
    gt_occlusion_aware_clustering = OcclusionAwareClustering(
            clustering_eps=0.5,
            clustering_min_samples=12,
            angle_res_rad=gt_angle_res_rad, #was 0.017
            occlusion_threshold=0.7,
            subsample_percentage=gt_subsample_percentage,
            remove_occluded=True,
            filter_method='ray_trace'
        )
    
    occlusion_aware_clustering = OcclusionAwareClustering(
            clustering_eps=radar_clustering_eps,
            clustering_min_samples=10,
            angle_res_rad=0.017,
            occlusion_threshold=0.9,
            subsample_percentage=0.20,
            remove_occluded=False,
            filter_method='ray_trace'
        )

    point_cloud_integrator = TemporalDensityPCIntegrator(
        gt_distance_threshold_m=gt_distance_threshold_m,
        num_frames_history_gt=1,
        valid_fovs_deg=[(-70,70),(110,-110)],
        num_frames_history=50,
        min_detection_radius=1.0,
        max_detection_radius=8.0,
        grid_resolution_m=0.1,
        gt_point_labeling_strategy=GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
        gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
        occlusion_aware_clustering=occlusion_aware_clustering
    )

    # Test Bench Setup
    test_bench = TemporalDensityPCIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        prediction_source=PredictionSource.VEHICLE_ODOM,
        gt_source=gt_source,
        odom_frame=odom_frame
    )

    # Initialize localization
    start_heading = np.deg2rad(0)
    start_pose = np.array([0.0, 0.0])
    test_bench.init_localization(est_start_heading_rad=start_heading, est_start_pose_m=start_pose)
    test_bench.init_filter(
        est_start_heading_rad=start_heading,
        est_start_position_m=start_pose,
        start_time_s=test_bench.get_dataset_start_time(idx=0)
    )

    # Select 5 frames
    frame_indices = np.linspace(0, dataset.num_frames - 1, 5, dtype=int)
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    current_frame = 0
    for i, target_idx in enumerate(frame_indices):
        print(f"Processing to frame {target_idx}...")
        test_bench.run(start_frame=current_frame, max_frame=target_idx + 1, gt_enabled=True)
        current_frame = target_idx + 1

        # 1. Accumulated Radar history
        radar_pc = point_cloud_integrator.get_points()[:, 0:2]
        
        # Overlay Map on Row 1
        map_points = map_handler.map_points
        pose_m = test_bench.history_position_m_gt[target_idx]
        heading_rad = np.deg2rad(test_bench.history_heading_deg_gt[target_idx])
        cos_h, sin_h = np.cos(-heading_rad), np.sin(-heading_rad)
        R_inv = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        map_pc_sensor = (map_points[:, 0:2] - pose_m) @ R_inv.T
        
        # Filter map points for visualization range
        map_mask = np.linalg.norm(map_pc_sensor, axis=1) < 15.0
        axes[0, i].scatter(map_pc_sensor[map_mask, 0], map_pc_sensor[map_mask, 1], s=1, c='gray', alpha=0.2)
        
        axes[0, i].scatter(radar_pc[:, 0], radar_pc[:, 1], s=2, c='blue', alpha=0.5)
        axes[0, i].set_title(f"F{target_idx}: Radar History")

        # 2. Raw GT (transformed to sensor frame)
        if gt_source == GroundTruthSource.LIDAR:
            gt_pc_raw = dataset.get_lidar_point_cloud_raw(target_idx)
            valid_points = gt_pc_raw[:, 2] > -0.1
            valid_points = valid_points & (gt_pc_raw[:, 2] < 0.5)
            gt_pc_sensor = gt_pc_raw[valid_points, :2]
        else:
            if dataset.lidar_enabled:
                gt_pc = dataset.get_lidar_point_cloud(target_idx)
            else:
                map_points = map_handler.map_points
                pose_m = test_bench.history_position_m_gt[target_idx]
                dist = np.linalg.norm(map_points[:, 0:2] - pose_m, axis=1)
                gt_pc = map_points[dist < 15.0]
            
            pose_m = test_bench.history_position_m_gt[target_idx]
            heading_rad = np.deg2rad(test_bench.history_heading_deg_gt[target_idx])
            cos_h, sin_h = np.cos(-heading_rad), np.sin(-heading_rad)
            R_inv = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            gt_pc_sensor = (gt_pc[:, 0:2] - pose_m) @ R_inv.T

        axes[1, i].scatter(gt_pc_sensor[:, 0], gt_pc_sensor[:, 1], s=1, c='gray', alpha=0.3)
        axes[1, i].set_title(f"F{target_idx}: Raw GT (Sensor Frame)")

        # 3. Ray-traced GT
        gt_pc_raytraced, _, _ = gt_occlusion_aware_clustering.process(gt_pc_sensor)

        axes[2, i].scatter(gt_pc_raytraced[:, 0], gt_pc_raytraced[:, 1], s=2, c='green')
        axes[2, i].set_title(f"F{target_idx}: Ray-traced GT")

        # 4. Labeled Radar Points (True Positives vs False Positives/Clutter)
        threshold = gt_distance_threshold_m
        
        # Overlay Map on Row 4
        axes[3, i].scatter(map_pc_sensor[map_mask, 0], map_pc_sensor[map_mask, 1], s=1, c='gray', alpha=0.2)

        if radar_pc.shape[0] > 0:
            if gt_pc_raytraced.shape[0] > 0:
                tree = cKDTree(gt_pc_raytraced[:, 0:2])
                dists, _ = tree.query(radar_pc[:, 0:2], distance_upper_bound=threshold)
                labels = dists <= threshold
                
                axes[3, i].scatter(radar_pc[labels, 0], radar_pc[labels, 1], s=4, c='blue', label='True Positive')
                axes[3, i].scatter(radar_pc[~labels, 0], radar_pc[~labels, 1], s=2, c='red', alpha=0.2, label='False Positive')
                if i == 0:
                    axes[3, i].legend()
            
            # Calculate and apply plot limits based on radar points (Symmetric viewport)
            x_min, x_max = np.min(radar_pc[:, 0]), np.max(radar_pc[:, 0])
            y_min, y_max = np.min(radar_pc[:, 1]), np.max(radar_pc[:, 1])
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            max_range = max(x_max - x_min, y_max - y_min) / 2
            
            # Add padding
            pad = 0.5
            xlim = (center_x - max_range - pad, center_x + max_range + pad)
            ylim = (center_y - max_range - pad, center_y + max_range + pad)
            
            for r in range(4):
                axes[r, i].set_xlim(xlim)
                axes[r, i].set_ylim(ylim)
                axes[r, i].set_aspect('equal', adjustable='box')
        else:
            axes[3, i].text(0.5, 0.5, "No radar/gt points", ha='center')
        
        axes[3, i].set_title(f"F{target_idx}: Labeled Radar (dist={threshold}m)")

    # Save output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_path = os.path.join(results_dir, f"gt_tuning_plot_{PLATFORM}.png")
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
