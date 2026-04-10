import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator, GtPointLabelingStrategy
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering

class _PointCloudIntegrator:
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
        grid_resolution_m (float): Resolution of a quantized grid
            of the point clouds.
        classify_gt_on_current_frame (bool): If true, all points are
            re-classified as gt points or not based on the current frame's gt
            points. On false, points are classified as gt points only using
            that frame's gt points.
        gt_occlusion_aware_clustering (OcclusionAwareClustering): If not None, the occlusion aware
            clustering is used to determine gt points. If None, the standard
            clustering is used.
        occlusion_aware_clustering (OcclusionAwareClustering): If not None, the occlusion aware
            clustering is used to determine pre-filter radar detections. If None, the standard
            clustering is used.
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
            gt_occlusion_aware_clustering:OcclusionAwareClustering=None,
            occlusion_aware_clustering:OcclusionAwareClustering=None,
    ) -> None:
        """
        Initialize the _PointCloudIntegrator.

        Args:
            gt_distance_threshold_m (float, optional): Distance threshold for
                associating ground truth points. Defaults to 0.1.
            valid_fovs_deg (list[tuple[float, float]], optional): A list of valid FOVs in degrees, e.g. [(-60, 60)].
                0 degrees is the +x axis, +90 degrees is the +y axis. Defaults to [(-180, 180)].
            num_frames_history (int, optional): Number of frames to keep in history.
                Defaults to 20.
            num_frames_history_gt (int, optional): Number of frames to keep gt points
                in history. Defaults to 1.
            num_frames_valid_point_history (int, optional): Number of frames to keep valid points 
                in history. Defaults to 0.
            subsample_percentage (float, optional): Percentage of new points to keep.
                Should be between 0.0 and 1.0. Defaults to 1.0 (no subsampling).
            min_detection_radius (float, optional): Minimum distance from origin to
                keep points. Defaults to 0.25.
            max_detection_radius (float, optional): Maximum distance from origin to
                keep points. Defaults to 20.0.
            grid_resolution_m (float, optional): Resolution of a quantized grid
                of the point clouds. Defaults to 0.1.
            gt_point_labeling_strategy (GtPointLabelingStrategy, optional): Strategy for 
                labeling the ground truth points.
            gt_occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None, the occlusion aware
                clustering is used to determine gt points. If None, the standard
                clustering is used. Defaults to None.
            occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None, the occlusion aware
                clustering is used to determine pre-filter radar detections. If None, the standard
                clustering is used. Defaults to None.
        """
        
        #pose tracking
        self.previous_pose:Pose = None
        self.current_transformation:Transformation = None
        self.current_static_points:np.ndarray = None

        #detection filtering
        self.min_detection_radius:float = min_detection_radius
        self.max_detection_radius:float = max_detection_radius
        self.pc_range_filter:pcRangeFilter = pcRangeFilter(
            min_detection_radius_m=self.min_detection_radius,
            max_detection_radius_m=self.max_detection_radius
        )
        
        #raw point cloud history/accumulation
        self.raw_point_history:PcAccumulator = PcAccumulator(
            gt_distance_threshold_m=gt_distance_threshold_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            subsample_percentage=subsample_percentage,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=gt_occlusion_aware_clustering,
            occlusion_aware_clustering=occlusion_aware_clustering,
            grid_resolution_m=grid_resolution_m,
            max_detection_range=max_detection_radius
        )

        self.num_frames_valid_point_history: int = num_frames_valid_point_history
        self.valid_point_history: PcAccumulator = PcAccumulator(
            gt_distance_threshold_m=gt_distance_threshold_m,
            valid_fovs_deg=[(-180, 180)],
            num_frames_history=num_frames_valid_point_history,
            num_frames_history_gt=0,
            subsample_percentage=1.0,
            efficient=True,
            gt_point_labeling_strategy=gt_point_labeling_strategy,
            gt_occlusion_aware_clustering=None,
            occlusion_aware_clustering=None,
            grid_resolution_m=grid_resolution_m,
            max_detection_range=max_detection_radius
        )
        self.valid_points: np.ndarray = None

        self.num_frames_history = num_frames_history
        self.num_frames_captured = 0

    def reset(self): 
        """
        Reset the integrator state.

        Clears the point history and resets internal state variables.
        """

        self.raw_point_history.reset()
        self.valid_point_history.reset()
        self.num_frames_captured = 0
    
    def check_valid_num_frames(self) -> bool:
        """
        Check if the number of frames captured is valid.

        Returns:
            bool: True if the number of frames captured is valid, False otherwise.
        """
        return self.num_frames_captured >= self.num_frames_history  


    def add_points(
            self,
            static_points: np.ndarray,
            current_pose: Pose,
            gt_points: np.ndarray = np.empty(shape=(0, 3))
    ):
        """
        Add new points to the integrator, updating state based on vehicle motion.

        Filters points by range, computes the transformation from the previous pose,
        transforms existing history, and adds the new points.

        Args:
            static_points (np.ndarray): Nx3 (or Nx4) array of static points from
                the current frame.
            current_pose (Pose): The current pose of the vehicle.
            gt_points (np.ndarray, optional): Nx3 array of ground truth points.
                Defaults to empty.
        """
        #should be extended by child class
        
        #filter points to the desired range
        self.current_static_points = self.pc_range_filter.get_points_in_detection_range(
            points=static_points
        )

        #compute the transformation to go from the previous
        if self.previous_pose:
            
            self.current_transformation = Transformation.from_orig_to_new(
                original_pose=self.previous_pose,
                new_pose=current_pose
            )
            #add points to the raw history
            self.raw_point_history.apply_transformation(self.current_transformation)
            self.raw_point_history.add_points(
                new_points=self.current_static_points[:,0:3],
                new_gt_points=gt_points
            )
        
            self._update_valid_points()
        
        #save the previous pose
        self.previous_pose = current_pose

        self.num_frames_captured += 1

    def _update_valid_points(self) -> None:
        """
        Update the valid point history.

        This method synchronizes the valid point history with the current vehicle
        pose and appends the newly computed valid points retrieved from the raw accumulator.
        Ground truth points are purposefully excluded.
        """
        if self.current_transformation:
            self.valid_point_history.apply_transformation(self.current_transformation)
            
        self.valid_points = self.raw_point_history.get_points()
        
        self.valid_point_history.add_points(
            new_points=self.valid_points[:, 0:3],
            new_gt_points=np.empty(shape=(0, 3))
        )

    def get_points(self) -> np.ndarray:
        """
        Retrieve the integrated points.

        Returns:
            np.ndarray: Array of integrated points.
        """
        if not self.check_valid_num_frames():
            return np.empty(shape=(0,3))
        return self.valid_point_history.get_points()
    
    def get_gt_points(self) -> np.ndarray:
        """
        Retrieve the ground truth points.

        Returns:
            np.ndarray: Array of ground truth points.
        """
        if not self.check_valid_num_frames():
            return np.empty(shape=(0,3))
        return self.raw_point_history.get_gt_points()
    
    def get_raw_point_history(self) -> np.ndarray:
        """
        Retrieve the raw accumulated point history.

        Returns:
            np.ndarray: Nx3 (or Nx4) array of raw accumulated points.
        """
        
        return self.raw_point_history.get_points(raw=True)
        
    
    def get_nodes(self, normalize_frames: bool = False, raw: bool = False) -> tuple:
        """Retrieve nodes and labels for graph-based processing.

        Args:
            normalize_frames (bool, optional): If True, normalizes the frames
                remaining by the total number of frames. Defaults to False.
            raw (bool, optional): If True, returns nodes and labels derived from
                the raw point history. If False, returns nodes and labels derived
                from the processed (clustered/filtered) point history.
                Defaults to False.

        Returns:
            tuple: (nodes, labels). behavior depends on implementation point history.
        """
        return self.raw_point_history.get_nodes(normalize_frames=normalize_frames, raw=raw)
    
    def get_grid(self, density:bool = False, filter_for_gt_regions:bool = False) -> np.ndarray:
        """
        Get the current grid.

        Args:
            density (bool, optional): If True, return the density grid. Defaults to False.
            filter_for_gt_regions (bool, optional): If True, filter the points for only points near
                gt points (used for generating grid with GT points nearby). Defaults to False.

        Returns:
            np.ndarray: MxM grid where 1 indicates occupancy or log-normalized point densities.
        """
        return self.raw_point_history.get_grid(density=density, filter_for_gt_regions=filter_for_gt_regions)
    
    def get_gt_grid(self, density:bool = False) -> np.ndarray:
        """
        Get the current ground truth grid.

        Args:
            density (bool, optional): If True, return the density grid. Defaults to False.

        Returns:
            np.ndarray: MxM grid where 1 indicates occupancy or log-normalized point densities.
        """
        return self.raw_point_history.get_gt_grid(density=density)