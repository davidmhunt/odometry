from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.test_benches.naive_radar_baseline_tb import NaiveRadarBaselineTB
from odometry.test_benches._test_bench import PredictionSource, OdomCoordinateFrame, GroundTruthSource

def get_test_bench(dataset, map_handler, model_info=None, **kwargs):
    """Initializes and returns a NaiveRadarBaselineTB for IcaRAus Naive Radar UGV evaluation.
    """
    
    # initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=85,
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=30, #less due to sparsity
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

    return NaiveRadarBaselineTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        use_filters=True,
        prediction_source=PredictionSource.VEHICLE_ODOM,
        gt_source=GroundTruthSource.LIDAR,
        odom_frame=OdomCoordinateFrame.FLU
    )
