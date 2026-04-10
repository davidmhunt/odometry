import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering

from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.point_cloud_processing.accumulation.grids.temporal_density_pc_grid import TemporalDensityPCGrid

class TemporalDensityPCIntegrator(_PointCloudIntegrator):
    """
    Integrates point clouds using a TemporalDensityPCGrid for grid-based accumulation.

    Maintains a history of points using a TemporalDensityPCGrid, which records
    log-normalized density and temporal information (most recent frame) for each cell.
    This integrator is particularly useful for generating 5-channel grid nodes
    [x, y, z=0, density, temporal] for graph-based models.
    """

    def __init__(
            self,
            gt_distance_threshold_m: float = 0.1,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_history: int = 20,
            num_frames_history_gt: int = 1,
            num_frames_valid_point_history: int = 0,
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
        Initialize the TemporalDensityPCIntegrator.

        Args:
            gt_distance_threshold_m (float, optional): Distance threshold for associates.
            valid_fovs_deg (list[tuple[float, float]], optional): List of valid FOVs.
            num_frames_history (int, optional): History length for detections.
            num_frames_history_gt (int, optional): History length for gt.
            num_frames_valid_point_history (int, optional): History length for valid points.
            subsample_percentage (float, optional): Percentage of new points to keep.
                Should be between 0.0 and 1.0. Defaults to 1.0 (no subsampling).
            min_detection_radius (float, optional): Min point distance.
            max_detection_radius (float, optional): Max point distance.
            grid_resolution_m (float, optional): Resolution of the cells.
            gt_point_labeling_strategy (GtPointLabelingStrategy, optional): Labeling method.
            gt_occlusion_aware_clustering (OcclusionAwareClustering, optional): For GT processing.
            occlusion_aware_clustering (OcclusionAwareClustering, optional): For detection processing.
        """
        
        super().__init__(
            gt_distance_threshold_m=gt_distance_threshold_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            num_frames_valid_point_history=num_frames_valid_point_history,
            subsample_percentage=subsample_percentage,
            min_detection_radius=min_detection_radius,
            max_detection_radius=max_detection_radius,
            grid_resolution_m=grid_resolution_m,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
            occlusion_aware_clustering=occlusion_aware_clustering,
            **kwargs
        )

        self.valid_point_history: TemporalDensityPCGrid = TemporalDensityPCGrid(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=max_detection_radius,
            valid_fovs_deg=[(-180, 180)],
            num_frames_history=num_frames_valid_point_history,
            num_frames_history_gt=0,
            subsample_percentage=1.0,
            efficient=True,
            gt_distance_threshold_m=gt_distance_threshold_m,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=None,
            # occlusion_aware_clustering=OcclusionAwareClustering(
            #     clustering_eps=0.25,
            #     clustering_min_samples=10,
            #     angle_res_rad=0.017,
            #     occlusion_threshold=0.9,
            #     subsample_percentage=1.0,
            #     remove_occluded=False,
            #     filter_method='overlap' #ray_trace or overlap
            # ),
            **kwargs
        )

        # Replace the base raw_point_history with TemporalDensityPCGrid
        self.raw_point_history: TemporalDensityPCGrid = TemporalDensityPCGrid(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=max_detection_radius,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            subsample_percentage=subsample_percentage,
            gt_distance_threshold_m=gt_distance_threshold_m,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
            occlusion_aware_clustering=occlusion_aware_clustering,
            **kwargs
        )
        
