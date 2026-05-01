import os
import sys
import numpy as np
import importlib.util
from tqdm import tqdm

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy

def main():
    # Import dependencies
    from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator import TemporalDensityPCIntegrator
    from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
    from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
    from odometry.test_benches._test_bench import PredictionSource, GroundTruthSource, OdomCoordinateFrame
    from odometry.localization.icp2D_localization import icp2DLocalization

    # Dataset Parameters (UGV)
    DATASET_PATH = "/data/IcaRAus/datasets/UGV"
    MAP_DIRECTORY = "/data/IcaRAus/maps"
    folder_name = "WILK"
    file_name = "IcaRAus_ugv_wilk_1_5m"
    map_file = "wilk_map.yaml"
    odom_frame = OdomCoordinateFrame.FLU
    gt_source = GroundTruthSource.LIDAR

    # Initialize Dataset
    ds_params = {
        "dataset_path": os.path.join(DATASET_PATH, folder_name, file_name),
        "radar_pc_folder": "radar_combined_pc",
        "lidar_folder": "lidar",
        "vehicle_odom_folder": "vehicle_odom",
        "imu_orientation_folder": "imu_data",
        "imu_full_folder": "imu_data_full",
        "vehicle_vel_folder": "vehicle_vel"
    }
    dataset = CpslDS(**ds_params)
    
    # Initialize Map
    map_handler = MapHandler(maps_folder=MAP_DIRECTORY, map_file=map_file)
    
    # Initialize Localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=80,
        icp_point_pairs_threshold=7,
        icp_max_iterations=5
    )
    lidar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.1,
        icp_best_points_percentile=50,
        icp_point_pairs_threshold=10,
        icp_max_iterations=5
    )
    
    # Integrator Setup
    gt_occlusion_aware_clustering = OcclusionAwareClustering(
        clustering_eps=0.5,
        clustering_min_samples=12,
        angle_res_rad=0.017,
        occlusion_threshold=0.7,
        subsample_percentage=0.2,
        remove_occluded=True,
        filter_method='ray_trace',
        enable_timing=True
    )
    
    occlusion_aware_clustering = OcclusionAwareClustering(
        clustering_eps=0.35,
        clustering_min_samples=10,
        angle_res_rad=0.017,
        occlusion_threshold=0.9,
        subsample_percentage=0.20,
        remove_occluded=False,
        filter_method='ray_trace',
        enable_timing=True
    )

    point_cloud_integrator = TemporalDensityPCIntegrator(
        gt_distance_threshold_m=1.0,
        num_frames_history_gt=1,
        valid_fovs_deg=[(-70,70),(110,-110)],
        num_frames_history=50,
        min_detection_radius=1.0,
        max_detection_radius=8.0,
        grid_resolution_m=0.1,
        gt_point_labeling_strategy=GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
        gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
        occlusion_aware_clustering=occlusion_aware_clustering,
        enable_timing=True
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
        odom_frame=odom_frame,
        enable_timing=True
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

    # Run for a subset of frames to get timing data
    num_frames = min(100, dataset.num_frames)
    print(f"Running timing analysis for {num_frames} frames...")
    
    test_bench.run(start_frame=0, max_frame=num_frames, gt_enabled=True)

    # Report Timing Stats
    print("\n" + "="*40)
    print("TEST BENCH TIMING STATS (Cumulative)")
    print("="*40)
    for key, val in test_bench.timing_stats.items():
        print(f"{key:20}: {val:10.4f} s ({(val/num_frames)*1000:8.2f} ms/frame)")
    
    print("\n" + "="*40)
    print("PC ACCUMULATOR TIMING STATS (Cumulative)")
    print("="*40)
    acc = point_cloud_integrator.raw_point_history
    for key, val in acc.timing_stats.items():
        print(f"{key:20}: {val:10.4f} s ({(val/num_frames)*1000:8.2f} ms/frame)")

    print("\n" + "="*40)
    print("GT CLUSTERING TIMING STATS (Cumulative)")
    print("="*40)
    gt_clust = point_cloud_integrator.raw_point_history.gt_occlusion_aware_clustering
    if gt_clust:
        for key, val in gt_clust.timing_stats.items():
            print(f"{key:20}: {val:10.4f} s ({(val/num_frames)*1000:8.2f} ms/frame)")
    
    print("\n" + "="*40)
    print("RADAR CLUSTERING TIMING STATS (Cumulative)")
    print("="*40)
    radar_clust = point_cloud_integrator.raw_point_history.occlusion_aware_clustering
    if radar_clust:
        for key, val in radar_clust.timing_stats.items():
            print(f"{key:20}: {val:10.4f} s ({(val/num_frames)*1000:8.2f} ms/frame)")
    print("="*40)

if __name__ == "__main__":
    main()
