import numpy as np
from enum import Enum
from geometries.transforms.transformation import Transformation
from scipy.spatial import cKDTree
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter

class GtPointLabelingStrategy(Enum):
    USE_VALID_POINTS_FOR_GT_CLASSIFICATION = 0
    USE_GT_POINTS_FOR_GT_CLASSIFICATION = 1

class PcAccumulator:
    """
    Accumulates point cloud data over multiple frames, handling persistence and decay.

    This class maintains a collection of 3D points and optional ground truth points,
    managing their lifecycle through a 'time-to-live' mechanism. It supports
    rigid body transformations and querying of valid points.

    Attributes:
        gt_distance_threshold_m (float): Distance threshold in meters for associating
            detections with ground truth points.
        num_frames_history (int): Number of frames a point persists before being removed.
        num_frames_history_gt (int): Number of frames a gt point persists before being removed.
        points_raw (np.ndarray): Nx4 array storing [x, y, z, frames_remaining] for detections.
        gt_points_raw (np.ndarray): Nx4 array storing [x, y, z, frames_remaining] for ground truth.
        gt_point_labeling_strategy (GtPointLabelingStrategy): The strategy to use for labeling gt points.
    """
    def __init__(
            self,
            gt_distance_threshold_m: float = 0.05,
            num_frames_history:int = 30,
            num_frames_history_gt:int = 1,
            gt_point_labeling_strategy:GtPointLabelingStrategy = GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            gt_occlusion_aware_clustering:OcclusionAwareClustering=None,
            occlusion_aware_clustering:OcclusionAwareClustering=None,
            grid_resolution_m: float = 0.1,
            max_detection_range: float = 20.0
    ):
        """
        Initialize the PcAccumulator.

        Args:
            gt_distance_threshold_m (float, optional): Maximum distance in meters to
                associate a detection with a ground truth point. Defaults to 0.05.
            num_frames_history (int, optional): The number of frames a point should
                persist in the accumulator before expiring. Defaults to 30.
            num_frames_history_gt (int, optional): The number of frames a gt point should
                persist in the accumulator before expiring. Defaults to 1.
            gt_point_labeling_strategy (GtPointLabelingStrategy, optional): The strategy to use for labeling gt points.
                Defaults to GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION.
            gt_occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None, the occlusion aware
                clustering is used to determine gt points. If None, the standard
                clustering is used. Defaults to None.
            occlusion_aware_clustering (OcclusionAwareClustering, optional): If not None, the occlusion aware
                clustering is used to determine pre-filter radar detections. If None, the standard
                clustering is used. Defaults to None.
            grid_resolution_m (float, optional): Resolution of the grid. Defaults to 0.1.
            max_detection_range (float, optional): Maximum distance from center for the grid. Defaults to 20.0.
        """

        self.gt_distance_threshold_m:float = gt_distance_threshold_m
        self.num_frames_history:int = num_frames_history
        self.num_frames_history_gt:int = num_frames_history_gt

        #gt point labeling strategy
        self.gt_point_labeling_strategy:GtPointLabelingStrategy = gt_point_labeling_strategy

        #clustering
        self.gt_occlusion_aware_clustering:OcclusionAwareClustering = gt_occlusion_aware_clustering
        self.occlusion_aware_clustering:OcclusionAwareClustering = occlusion_aware_clustering

        self.grid_resolution_m: float = grid_resolution_m
        self.grid_max_distance_m: float = max_detection_range
        self.grid_bins: np.ndarray = np.arange(
            start=-1 * self.grid_max_distance_m,
            stop=self.grid_max_distance_m + self.grid_resolution_m,
            step=self.grid_resolution_m
        )

        self.pc_range_filter:pcRangeFilter = pcRangeFilter(
            min_detection_radius_m=0.0,
            max_detection_radius_m=max_detection_range
        )

        # Collection of currently available points
        self.points_raw:np.ndarray = None
        self.gt_points_raw:np.ndarray = None

        #collection of post processed points
        self.points:np.ndarray = None
        self.gt_points:np.ndarray = None
        
        self.reset()
    
    def reset(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points:np.ndarray = np.empty(shape=(0,3))):
        """
        Reset the accumulator and optionally seed it with new points.

        Clears existing points and optionally adds an initial set of detections
        and ground truth points.

        Args:
            new_points (np.ndarray, optional): Nx3 array of [x, y, z] points to
                initialize the accumulator with. Defaults to empty.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground
                truth points to initialize with. Defaults to empty.
        """
        self.points_raw = np.empty(shape=(0,4))
        self.gt_points_raw = np.empty(shape=(0,4))

        self.points = np.empty(shape=(0,4))
        self.gt_points = np.empty(shape=(0,4))

        if new_points.shape[0] > 0:
            self.add_points(
                new_points=new_points,
                new_gt_points=new_gt_points
            )
        return

    def add_points(
            self,
            new_points: np.ndarray,
            new_gt_points: np.ndarray=np.empty(shape=(0,3))
    ):
        """
        Add new points to the accumulator, updating persistence timers.

        This method performs the following steps:
        1. Decrements the timer for existing points and removes expired ones.
        2. Appends the `new_points` with a fresh timer initialized to
           `self.num_frames_history`.
        3. Similarly updates and adds ground truth points if provided.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] detected points to add.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground
                truth points. Defaults to empty.

        Raises:
            ValueError: If `new_points` does not have 3 columns (x, y, z).
        """
        if new_points.shape[1] != 3 :
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")
        
        #prune out expired points
        self._prune_points_array()
        
        self._update_points(new_points)

        if new_gt_points.shape[0] > 0:
            self._update_gt_points(new_gt_points)
            
        self._update_gt_labels()
    
    def _prune_points_array(self):
        """
        Prune the points array by removing expired points.
        """

        #prune any expired detected points
        if self.points_raw.shape[0] > 0:

            #decay the detection history by a step
            self.points_raw[:,3] = \
                self.points_raw[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.points_raw[:,3] > 0
            self.points_raw = self.points_raw[valid_idxs]

            #remove points outside the grid
            self.points_raw = self.pc_range_filter.get_points_in_detection_range(self.points_raw)
        
        #prune any expired gt points
        if self.gt_points_raw.shape[0] > 0:

            #decay the detection history by a step
            self.gt_points_raw[:,3] = \
                self.gt_points_raw[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.gt_points_raw[:,3] > 0
            self.gt_points_raw = self.gt_points_raw[valid_idxs]

            #remove points outside the grid
            self.gt_points_raw = self.pc_range_filter.get_points_in_detection_range(self.gt_points_raw)
        
    
    def _update_points(self, new_points: np.ndarray):
        """
        Update the detected points and raw points array.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] detected points to add.
        """
        # Append new points to the existing collection
        new_points = np.hstack((
            new_points,
            np.zeros(shape=
                        (new_points.shape[0],1))
        ))
        new_points[:,3] = self.num_frames_history
        self.points_raw = np.vstack((self.points_raw, new_points))

        if self.occlusion_aware_clustering is not None:
            self.points, _, _ = self.occlusion_aware_clustering.process(
                pc_cartesian=self.points_raw
            )
        else:
            self.points = self.points_raw.copy()

    def _update_gt_points(self, new_gt_points:np.ndarray):
        """
        Update the gt points.

        Args:
            new_gt_points (np.ndarray): At least Nx3 array of [x, y, z] ground truth points.
        """

        #remove occluded ground truth points
        if self.gt_occlusion_aware_clustering is not None:
            new_gt_points, _, _ = self.gt_occlusion_aware_clustering.process(
                pc_cartesian=new_gt_points
            )
            
        new_gt_points = np.hstack((
            new_gt_points,
            np.zeros(shape=
                        (new_gt_points.shape[0],self.num_frames_history_gt))
        ))
        new_gt_points[:,3] = self.num_frames_history_gt
        self.gt_points_raw = np.vstack((self.gt_points_raw, new_gt_points))
        
    
    def _update_gt_labels(self):
        """
        Update the gt labels for the new gt points.
        """

        if self.gt_occlusion_aware_clustering is not None:
            gt_points_filtered, _, _ = self.gt_occlusion_aware_clustering.process(
                pc_cartesian=self.gt_points_raw
            )
        else:
            gt_points_filtered = self.gt_points_raw.copy()

        #using only current frame detections to classify new gt points
        if self.gt_point_labeling_strategy == GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION:
            self.gt_points = self._get_dets_close_to_gt_points(
                dets=self.points[:,0:3],
                gt_points=gt_points_filtered[:,0:3],
                threshold=self.gt_distance_threshold_m
            )
        elif self.gt_point_labeling_strategy == GtPointLabelingStrategy.USE_GT_POINTS_FOR_GT_CLASSIFICATION:
            self.gt_points = self._get_gt_points_close_to_dets(
                dets=self.points[:,0:3],
                gt_points=gt_points_filtered[:,0:3],
                threshold=self.gt_distance_threshold_m
            )
        

    def apply_transformation(self, transformation: Transformation):
        """
        Apply a rigid body transformation to all stored points.

        Updates the spatial coordinates (x, y, z) of both detected points and
        ground truth points using the provided transformation object.
        The point cloud representation is then updated to reflect the new positions.

        Args:
            transformation (Transformation): The transformation object containing
                rotation and translation to apply.
        """
        self.points_raw[:,0:3] = transformation.apply_transformation(self.points_raw[:,0:3])
        if self.points.shape[0] > 0:
            self.points[:,0:3] = transformation.apply_transformation(self.points[:,0:3])

        #handle the ground truth points
        if self.gt_points_raw.shape[0] > 0:

            self.gt_points_raw[:,0:3] = transformation.apply_transformation(self.gt_points_raw[:,0:3])
            
        if self.gt_points.shape[0] > 0:
            self.gt_points[:,0:3] = transformation.apply_transformation(self.gt_points[:,0:3])
    
    def get_points(self, raw:bool = False)->np.ndarray:
        """
        Retrieve the currently valid detected points.

        Args:
            raw (bool, optional): If True, returns the points including their
                persistence timer. If False, returns only spatial coordinates.
                Defaults to False.

        Returns:
            np.ndarray: If raw is False, an Nx3 array of [x, y, z].
                If raw is True, an Nx4 array of [x, y, z, frames_remaining].
        """
        if raw:
            return self.points_raw
        else:
            return self.points[:,0:3]
    def get_gt_points(self, raw:bool = False)->np.ndarray:
        """
        Retrieve the currently valid ground truth points.

        Args:
            raw (bool, optional): If True, returns the points including their
                persistence timer. If False, returns only spatial coordinates.
                Defaults to False.

        Returns:
            np.ndarray: If raw is False, an Nx3 array of [x, y, z].
                If raw is True, an Nx4 array of [x, y, z, frames_remaining].
        """

        if raw:
            return self.gt_points_raw
        else:
            return self.gt_points[:,0:3]
    
    def get_nodes(self, normalize_frames:bool = False)->tuple:
        """
        Retrieve points as nodes with ground truth labels.

        Useful for graph-based processing where points are treated as nodes.

        Args:
            normalize_frames (bool, optional): If True, normalizes the frames
                remaining by the total number of frames. Defaults to False.

        Returns:
            tuple: A pair (nodes, labels).
                - nodes (np.ndarray): Nx4 array of [x, y, z, frames_remaining].
                - labels (np.ndarray): N-element boolean array, where True indicates
                  the point is close to a ground truth point (true positive).
        """
        
        if self.points_raw.shape[0] > 0:
                
            nodes = self.points_raw.copy()

            if normalize_frames:
                nodes[:,3] = nodes[:,3] / self.num_frames_history

            if self.gt_points.size > 0:
                tree = cKDTree(self.gt_points[:,0:3])
                dists, _ = tree.query(self.points[:,0:3], distance_upper_bound=1e-5)
                label = dists <= 1e-5
            else:
                label = np.zeros(len(self.points), dtype=bool)
            
            

            return nodes, label

        else:
            return np.empty(shape=(0, 4)),np.empty(shape=0)

    def _get_dets_close_to_gt_points(
            self,dets:np.ndarray,
            gt_points:np.ndarray,
            threshold:float=0.05)->np.ndarray:
        """
        Filter detections to find those close to ground truth points.

        Args:
            dets (np.ndarray): Nx3 array of [x, y, z] detected points.
            gt_points (np.ndarray): Mx3 array of [x, y, z] ground truth points.
            threshold (float, optional): Maximum Euclidean distance to consider
                a detection as matching a ground truth point. Defaults to 0.05.

        Returns:
            np.ndarray: Subset of `dets` that are within `threshold` distance
            of any point in `gt_points`.

        Raises:
            ValueError: If inputs are not Nx3 arrays.
        """
        if gt_points.shape[1] == 3 and dets.shape[1] == 3:
            
            if gt_points.shape[0] == 0 or dets.shape[0] == 0:
                return np.empty(shape=(0, 3))
            
            # Use cKDTree for efficient nearest neighbor search
            tree = cKDTree(gt_points)
            dists, _ = tree.query(dets, distance_upper_bound=threshold)
            
            mask = dists <= threshold
            dets = dets[mask]

            return dets
        else:
            raise ValueError("Detections and ground truth detections must be a 3D point (3,) or an Nx3 array of points.")
    
    def _get_gt_points_close_to_dets(
            self,dets:np.ndarray,
            gt_points:np.ndarray,
            threshold:float=0.05)->np.ndarray:
        """
        Filter ground truth points to find those close to detections.

        Args:
            dets (np.ndarray): Nx3 array of [x, y, z] detected points.
            gt_points (np.ndarray): Mx3 array of [x, y, z] ground truth points.
            threshold (float, optional): Maximum Euclidean distance to consider
                a detection as matching a ground truth point. Defaults to 0.05.

        Returns:
            np.ndarray: Subset of `gt_points` that are within `threshold` distance
            of any point in `dets`.

        Raises:
            ValueError: If inputs are not Nx3 arrays.
        """
        if gt_points.shape[1] == 3 and dets.shape[1] == 3:
            
            if gt_points.shape[0] == 0 or dets.shape[0] == 0:
                return np.empty(shape=(0, 3))
            
            # Use cKDTree for efficient nearest neighbor search
            tree = cKDTree(dets)
            dists, _ = tree.query(gt_points, distance_upper_bound=threshold)
            
            mask = dists <= threshold
            gt_points = gt_points[mask]

            return gt_points
        else:
            raise ValueError("Detections and ground truth detections must be a 3D point (3,) or an Nx3 array of points.")

    def _get_grid_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Convert a set of 3D points into a 2D grid representation (binary occupancy).

        Args:
            points (np.ndarray): Nx3 array of [x, y, z] points.

        Returns:
            np.ndarray: MxM binary grid where 1 indicates occupancy.
        
        Raises:
            ValueError: If input points dimensions are incorrect.
        """
        # Allow Nx4 input by slicing, but check dimension at least 3
        if points.ndim > 1 and points.shape[1] >= 3:

            points = self.filter_points_outside_grid(points)

            ret_grid = np.zeros(
                shape=(self.grid_bins.shape[0], self.grid_bins.shape[0]),
                dtype=np.int8
            )
            
            # Find the closest grid bin indices for x and y coordinates
            # Note: Assuming points are already filtered to be within range
            x_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 0]), axis=0)
            y_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 1]), axis=0)

            # Mark the grid cells as occupied (1)
            ret_grid[x_idx, y_idx] = 1
            return ret_grid
        elif points.shape[0] == 0:
             return np.zeros(
                shape=(self.grid_bins.shape[0], self.grid_bins.shape[0]),
                dtype=np.int8
            )
        else:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")

    def get_grid(self, density:bool = False) -> np.ndarray:
        """
        Get the current grid.

        Args:
            density (bool, optional): If True, return the density grid. Defaults to False.

        Returns:
            np.ndarray: MxM grid where 1 indicates occupancy or log-normalized point densities.
        """
        if density:
            return self._get_density_grid_from_points(self.points)
        else:
            return self._get_grid_from_points(self.points)
    
    def get_gt_grid(self, density:bool = False) -> np.ndarray:
        """
        Get the current ground truth grid.

        Args:
            density (bool, optional): If True, return the density grid. Defaults to False.

        Returns:
            np.ndarray: MxM grid where 1 indicates occupancy or log-normalized point densities.
        """
        if density:
            return self._get_density_grid_from_points(self.gt_points)
        else:
            return self._get_grid_from_points(self.gt_points)

    def _get_density_grid_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Convert a set of 3D points into a 2D log-normalized density grid representation.

        Args:
            points (np.ndarray): Nx3 or Nx4 array of points.

        Returns:
            np.ndarray: MxM grid where values are log-normalized point densities scaled to [0, 1].
        
        Raises:
            ValueError: If input points dimensions are incorrect.
        """
        # Allow Nx4 input by slicing, but check dimension at least 3
        if points.ndim > 1 and points.shape[1] >= 3:

            points = self.filter_points_outside_grid(points)

            # Initialize as float32 since we will be applying logs and division
            ret_grid = np.zeros(
                shape=(self.grid_bins.shape[0], self.grid_bins.shape[0]),
                dtype=np.float32 
            )
            
            # Find the closest grid bin indices for x and y coordinates
            x_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 0]), axis=0)
            y_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 1]), axis=0)

            # Safely increment the count for cells with multiple points
            np.add.at(ret_grid, (x_idx, y_idx), 1.0)
            
            # --- LOG NORMALIZATION ---
            # Use log1p (log(1 + x)) to safely handle cells with 0 points
            ret_grid = np.log1p(ret_grid)
            
            # Scale the final grid to exactly [0.0, 1.0] for the neural network
            max_density = np.max(ret_grid)
            if max_density > 0:
                ret_grid = ret_grid / max_density
                
            return ret_grid
            
        elif points.shape[0] == 0:
             return np.zeros(
                shape=(self.grid_bins.shape[0], self.grid_bins.shape[0]),
                dtype=np.float32
            )
        else:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")

    def _get_points_from_pc_grid(self, pc_grid: np.ndarray) -> np.ndarray:
        """
        Convert a point cloud grid back into an array of points (cell centers).

        Args:
            pc_grid (np.ndarray): NxN grid representation.

        Returns:
            np.ndarray: Nx3 array of reconstructed points [x, y, 0].
        """
        # Identify the indices of occupied grid cells
        x_idxs, y_idxs = np.nonzero(pc_grid)

        if x_idxs.shape[0] > 0:
            # Convert grid indices back to coordinate values
            x_vals = self.grid_bins[x_idxs]
            y_vals = self.grid_bins[y_idxs]
            z_vals = np.zeros_like(x_vals)  # Assume z=0 as it’s a 2D representation

            return np.column_stack((x_vals, y_vals, z_vals))
        else:
            return np.empty(shape=(0, 3))
    
    def filter_points_outside_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Remove points that fall outside the defined grid boundaries.

        Args:
            points (np.ndarray): Nx3 (or Nx4) array of points.

        Returns:
            np.ndarray: Filtered array containing only points within `grid_max_distance_m`.
        """
        if points.shape[0] == 0:
            return points
        return points[np.all(np.abs(points[:, 0:3]) <= self.grid_max_distance_m, axis=1)]
