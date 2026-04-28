import torch
import os
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.point_cloud_processing.accumulation.integrators.ragnnarok_pc_integrator_gnn import RagnnarokPointCloudIntegratorGNN
from odometry.test_benches.ragnnarok_point_cloud_integrator_tb import RaGNNPointCloudIntegratorTB
from mmwave_model_integrator.config import Config
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier
from odometry.test_benches._test_bench import PredictionSource, OdomCoordinateFrame


def get_test_bench(dataset, map_handler, model_info=None, **kwargs):
    """Initializes and returns a RaGNNPointCloudIntegratorTB for RaGNNarok GNN evaluation.

    Args:
        dataset (CpslDS): The dataset object to be evaluated.
        map_handler (MapHandler): The map handler for localization.
        model_info (dict, optional): Dictionary containing 'model_config_path' and 
            'model_state_dict_path'. Required for GNN-based evaluation.

    Returns:
        RaGNNPointCloudIntegratorTB: A fully initialized test bench for evaluation.
    """
    
    if model_info is None:
        raise ValueError("model_info is required for ragnnarok_gnn evaluation")

    model_config_path = model_info['model_config_path']
    model_state_dict_path = model_info['model_state_dict_path']

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
        icp_max_iterations=20,
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

    point_cloud_integrator = RagnnarokPointCloudIntegratorGNN(
        gnn_runner=runner,
        grid_resolution_m_prob=0.1,
        grid_max_distance_m_prob=5.0,
        num_frames_history_prob=20,
        grid_resolution_m_hist=0.2,
        grid_max_distance_m_hist=5.0,
        num_frames_history_hist=10,
        min_detection_radius=1.0,
        max_detection_radius=5.0,
        gt_distance_threshold_m_prob=0.2
    )
    
    return RaGNNPointCloudIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        model_dataset_generator=None,
        use_filters=True,
        prediction_source=PredictionSource.IMU_AND_VEL,
        gt_source="lidar",
        odom_frame=OdomCoordinateFrame.FLU
    )
