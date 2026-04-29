import os
import numpy as np
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator import TemporalDensityPCIntegrator
from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from odometry.test_benches._test_bench import PredictionSource, OdomCoordinateFrame, GroundTruthSource


def get_test_bench(dataset, map_handler, model_info=None, num_frames_history=50, **kwargs):
    """Initializes and returns a TemporalDensityPCIntegratorTB for IcaRAus Naive UAV evaluation.

    Args:
        dataset (CpslDS): The dataset object to be processed.
        map_handler (MapHandler): The map handler for localization.
        model_info (dict, optional): Unused for naive evaluation.
        num_frames_history (int, optional): Number of frames for point cloud accumulation. 
            Defaults to 50.
        **kwargs: Additional arguments (ignored by this factory).

    Returns:
        TemporalDensityPCIntegratorTB: A fully initialized test bench for naive evaluation.
    """
    
    # initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=85,
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

    point_cloud_integrator = TemporalDensityPCIntegrator(
            gt_distance_threshold_m=1.0,
            num_frames_history_gt=1,
            valid_fovs_deg=[(-70,70),(110,-110)],
            num_frames_history=num_frames_history,
            min_detection_radius=1.0,
            max_detection_radius=8.0,
            grid_resolution_m=0.1,
            subsample_percentage=1.0,
            gt_point_labeling_strategy=GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            occlusion_aware_clustering=OcclusionAwareClustering(
                clustering_eps=0.35,
                clustering_min_samples=10,
                angle_res_rad=0.017,
                occlusion_threshold=0.9,
                subsample_percentage=0.20,
                remove_occluded=False,
                filter_method='ray_trace'
            )
        )
    
    return TemporalDensityPCIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        dynamic_point_cloud_integrator=None,
        model_dataset_generator=None,
        use_filters=True,
        prediction_source=PredictionSource.VEHICLE_ODOM,
        gt_source=GroundTruthSource.MOTION_CAPTURE,
        odom_frame=OdomCoordinateFrame.NED
    )
