import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator

from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner

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
    """

    def __init__(
            self,
            gnn_runner:GNNRunner,
            normalize_frames:bool = False,
            gt_distance_threshold_m: float = 0.1,
            num_frames_history: int = 20,
            min_detection_radius: float = 0.25,
            max_detection_radius: float = 20.0,
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
            min_detection_radius (float, optional): Minimum distance from origin to
                keep points. Defaults to 0.25.
            max_detection_radius (float, optional): Maximum distance from origin to
                keep points. Defaults to 20.0.
        """
        
        super().__init__(
            gt_distance_threshold_m=gt_distance_threshold_m,
            num_frames_history=num_frames_history,
            min_detection_radius=min_detection_radius,
            max_detection_radius=max_detection_radius
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