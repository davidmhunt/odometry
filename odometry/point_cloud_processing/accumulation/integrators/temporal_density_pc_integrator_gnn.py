import numpy as np

from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder

from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering

from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator import TemporalDensityPCIntegrator

class TemporalDensityPCIntegratorGNN(TemporalDensityPCIntegrator):
    """
    Integrates point clouds using a TemporalDensityPCGrid and processes them with a GNN.

    Inherits from TemporalDensityPCIntegrator and extends it by using a GNN model
    to process the integrated points. It uses a GNNRunner to make predictions
    based on the nodes retrieved from the temporal density grid.
    """

    def __init__(
            self,
            gnn_runner: GNNRunner,
            input_encoder: _NodeEncoder = None,
            normalize_frames: bool = False,
            gt_distance_threshold_m: float = 0.1,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_history: int = 20,
            num_frames_history_gt: int = 1,
            subsample_percentage: float = 1.0,
            min_detection_radius: float = 0.25,
            max_detection_radius: float = 20.0,
            grid_resolution_m: float = 0.1,
            gt_point_labeling_strategy: GtPointLabelingStrategy = GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            gt_occlusion_aware_clustering: OcclusionAwareClustering = None,
            occlusion_aware_clustering: OcclusionAwareClustering = None,
            **kwargs
    ) -> None:
        """
        Initialize the TemporalDensityPCIntegratorGNN.

        Args:
            gnn_runner (GNNRunner): The GNN runner instance used for point classification.
            input_encoder (_NodeEncoder, optional): The input encoder to use for processing.
                Defaults to a base _NodeEncoder if None.
            normalize_frames (bool, optional): If True, normalizes the frames to be from 0 to 1.
                Defaults to False.
            gt_distance_threshold_m (float, optional): Distance threshold for associating
                ground truth points. Defaults to 0.1.
            valid_fovs_deg (list[tuple[float, float]], optional): A list of valid FOVs in degrees.
                Defaults to [(-180, 180)].
            num_frames_history (int, optional): Number of frames to keep in history.
                Defaults to 20.
            num_frames_history_gt (int, optional): Number of frames to keep gt points in history.
                Defaults to 1.
            subsample_percentage (float, optional): Percentage of new points to keep.
                Should be between 0.0 and 1.0. Defaults to 1.0.
            min_detection_radius (float, optional): Minimum distance from origin to keep points.
                Defaults to 0.25.
            max_detection_radius (float, optional): Maximum distance from origin to keep points.
                Defaults to 20.0.
            grid_resolution_m (float, optional): Resolution of the cells in the grid.
                Defaults to 0.1.
            gt_point_labeling_strategy (GtPointLabelingStrategy, optional): Strategy for labeling
                the ground truth points. Defaults to USE_VALID_POINTS_FOR_GT_CLASSIFICATION.
            gt_occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None,
                used to determine gt points. Defaults to None.
            occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None,
                used to pre-filter radar detections. Defaults to None.
        """
        
        super().__init__(
            gt_distance_threshold_m=gt_distance_threshold_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            subsample_percentage=subsample_percentage,
            min_detection_radius=min_detection_radius,
            max_detection_radius=max_detection_radius,
            grid_resolution_m=grid_resolution_m,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
            occlusion_aware_clustering=occlusion_aware_clustering,
            **kwargs
        )

        self.gnn_runner: GNNRunner = gnn_runner
        self.normalize_frames: bool = normalize_frames

        if input_encoder is None:
            self.input_encoder: _NodeEncoder = _NodeEncoder()
        else:
            self.input_encoder: _NodeEncoder = input_encoder

    def get_points(self) -> np.ndarray:
        """
        Retrieve the integrated points processed by the GNN.

        Returns:
            np.ndarray: Array of integrated points after GNN processing.
            Returns an empty array if not enough frames have been captured.
        """
        if not self.check_valid_num_frames():
            return np.empty(shape=(0, 3))
        
        nodes, labels = self.get_nodes(self.normalize_frames)
        
        # Encode the nodes
        nodes = self.input_encoder.encode(nodes)

        pred = self.gnn_runner.make_prediction(
            nodes=nodes
        )
        
        return pred[:, 0:3]
