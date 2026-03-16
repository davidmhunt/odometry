import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator

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
    """

    def __init__(
            self,
            gt_distance_threshold_m: float = 0.1,
            num_frames_history: int = 20,
            min_detection_radius: float = 0.25,
            max_detection_radius: float = 20.0,
    ) -> None:
        """
        Initialize the _PointCloudIntegrator.

        Args:
            gt_distance_threshold_m (float, optional): Distance threshold for
                associating ground truth points. Defaults to 0.1.
            num_frames_history (int, optional): Number of frames to keep in history.
                Defaults to 20.
            min_detection_radius (float, optional): Minimum distance from origin to
                keep points. Defaults to 0.25.
            max_detection_radius (float, optional): Maximum distance from origin to
                keep points. Defaults to 20.0.
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
            num_frames_history=num_frames_history
        )

        self.num_frames_history = num_frames_history
        self.num_frames_captured = 0

    def reset(self): 
        """
        Reset the integrator state.

        Clears the point history and resets internal state variables.
        """

        self.raw_point_history.reset()
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
        
        #save the previous pose
        self.previous_pose = current_pose

        self.num_frames_captured += 1

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
        return self.raw_point_history.get_points()
    
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
        
        return self.raw_point_history.get_points()
        
    
    def get_nodes(self, normalize_frames:bool = False) -> tuple:
        """
        Retrieve nodes and labels for graph-based processing.

        Args:
            normalize_frames (bool, optional): If True, normalizes the frames
                remaining by the total number of frames. Defaults to False.

        Returns:
            tuple: (nodes, labels). behavior depends on implementation point history.
        """
        #should be overridden by child classes
        return self.raw_point_history.get_nodes(normalize_frames=normalize_frames)