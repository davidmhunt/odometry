import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator, GtPointLabelingStrategy

from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner

from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering

class _PointCloudIntegratorGnnRunner(_PointCloudIntegrator):
    """
    Base class for point cloud integration and history management.

    Maintains a history of raw points accumulated over time, handling pose
    transformations and range filtering. It serves as a foundation for more
    complex integration strategies.

    Attributes:
        previous_pose (Pose): The pose of the vehicle at the previous timestamp.
        current_transformation (Transformation): The transformation computed from
            the previous pose to the current pose.
        current_static_points (np.ndarray): The filtered static points from the
            current frame.
        min_detection_radius (float): Minimum radius for accepting points.
        max_detection_radius (float): Maximum radius for accepting points.
        pc_range_filter (pcRangeFilter): Utility for filtering points by range.
        raw_point_history (PcAccumulator): Accumulator for the raw point history.
        classify_gt_on_current_frame (bool, optional): If true, all points are
            re-classified as gt points or not based on the current frame's gt
            points. On false, points are classified as gt points only using
            that frame's gt points. Defaults to False.
        use_occlusion_aware_detector (bool, optional): If True, the occlusion aware
            detector is used to determine gt points. If False, the standard
            detector is used. Defaults to False.
    """

    def __init__(
            self,
            gnn_runner:GNNRunner,
            normalize_frames:bool = False,
            gt_distance_threshold_m: float = 0.1,
            num_frames_history: int = 20,
            num_frames_history_gt: int = 1,
            min_detection_radius: float = 0.25,
            max_detection_radius: float = 20.0,
            grid_resolution_m: float = 0.1,
            gt_point_labeling_strategy: GtPointLabelingStrategy = GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            gt_occlusion_aware_clustering:OcclusionAwareClustering=None,
            occlusion_aware_clustering:OcclusionAwareClustering=None,
    ) -> None:
        """
        Initialize the _PointCloudIntegrator.

        Args:
            gnn_runner (GNNRunner): The GNN runner to use for processing.
            normalize_frames (bool, optional): If True, normalizes the frames
                to be from 0 to 1. Defaults to False.
            gt_distance_threshold_m (float, optional): Distance threshold for
                associating ground truth points. Defaults to 0.1.
            num_frames_history (int, optional): Number of frames to keep in history.
                Defaults to 20.
            num_frames_history_gt (int, optional): Number of frames to keep gt points 
                in history. Defaults to 1.
            min_detection_radius (float, optional): Minimum distance from origin to
                keep points. Defaults to 0.25.
            max_detection_radius (float, optional): Maximum distance from origin to
                keep points. Defaults to 20.0.
            grid_resolution_m (float, optional): Resolution of a quantized grid
                of the point clouds. Defaults to 0.1.
            gt_point_labeling_strategy (GtPointLabelingStrategy, optional): Strategy for 
                labeling the ground truth points.
            gt_occlusion_aware_clustering (OcclusionAwareClustering): If not None, the occlusion aware
                clustering is used to determine gt points. If None, the standard
                clustering is used.
            occlusion_aware_clustering (OcclusionAwareClustering): If not None, the occlusion aware
                clustering is used to determine pre-filter radar detections. If None, the standard
                clustering is used.
        """
        
        super().__init__(
            gt_distance_threshold_m=gt_distance_threshold_m,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            min_detection_radius=min_detection_radius,
            max_detection_radius=max_detection_radius,
            grid_resolution_m=grid_resolution_m,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
            occlusion_aware_clustering=occlusion_aware_clustering
        )

        self.normalize_frames:bool = normalize_frames
        self.gnn_runner:GNNRunner = gnn_runner

    def get_points(self) -> np.ndarray:
        """
        Retrieve the integrated points.

        Returns:
            np.ndarray: Array of integrated points. Defaults to returning the
            raw point history if not overridden.
        """
        #should be overridden by child classes
        if not self.check_valid_num_frames():
            return np.empty(shape=(0,3))
        
        nodes,labels = self.get_nodes(self.normalize_frames)
        
        pred = self.gnn_runner.make_prediction(
            nodes=nodes
        )
        
        return pred[:,0:3]