import torch
import os
import numpy as np
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator_gnn import TemporalDensityPCIntegratorGNN
from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from mmwave_model_integrator.config import Config
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from odometry.test_benches._test_bench import PredictionSource, OdomCoordinateFrame, GroundTruthSource


def get_test_bench(dataset, map_handler, model_info=None, num_frames_history=50, normalize_frames=True):
    """Initializes and returns a TemporalDensityPCIntegratorTB for RaGNNarok GNN on IcaRAus UGV evaluation.
    """
    
    if model_info is None:
        raise ValueError("model_info is required for ragnnarok_gnn_icaraus_ugv evaluation")

    model_config_path = model_info['model_config_path']
    model_state_dict_path = model_info['model_state_dict_path']

    # initialize the dataset encoders
    input_encoder = _NodeEncoder()

    # initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=90,
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

    # instantiate the model
    config = Config(model_config_path)
    model_cfg = config.model
    if 'type' in model_cfg:
        model_cfg.pop('type')
    model = SageGNNClassifier(**model_cfg)

    dataset_cfg = config.trainer["dataset"]
    enable_downsampling = dataset_cfg.get("enable_downsampling", False)

    runner = GNNRunner(
        model=model,
        state_dict_path=model_state_dict_path,
        cuda_device="cuda:0" if torch.cuda.is_available() else "cpu",
        edge_radius=10.0,
        enable_downsampling=enable_downsampling,
        use_sigmoid=False,
        print_stats=False
    )

    point_cloud_integrator = TemporalDensityPCIntegratorGNN(
            gnn_runner=runner,
            input_encoder=input_encoder,
            normalize_frames=normalize_frames,
            gt_distance_threshold_m=0.4,
            num_frames_history_gt=1,
            valid_fovs_deg=[(-70,70),(110,-110)],
            num_frames_history=num_frames_history,
            min_detection_radius=1.0,
            max_detection_radius=8.0,
            grid_resolution_m=0.1,
            subsample_percentage=1.0,
            gt_point_labeling_strategy=GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            gt_occlusion_aware_clustering=OcclusionAwareClustering(
                clustering_eps=0.5,
                clustering_min_samples=12,
                angle_res_rad=0.017,
                occlusion_threshold=0.7,
                subsample_percentage=1.0,
                remove_occluded=True,
                filter_method='ray_trace'
            ),
            occlusion_aware_clustering=OcclusionAwareClustering(
                clustering_eps=0.25,
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
        gt_source=GroundTruthSource.LIDAR,
        odom_frame=OdomCoordinateFrame.FLU
    )
