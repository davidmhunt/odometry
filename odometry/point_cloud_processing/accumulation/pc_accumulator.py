import numpy as np

from geometries.transforms.transformation import Transformation
from scipy.spatial import cKDTree

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
        points (np.ndarray): Nx4 array storing [x, y, z, frames_remaining] for detections.
        gt_points (np.ndarray): Nx4 array storing [x, y, z, frames_remaining] for ground truth.
    """
    def __init__(
            self,
            gt_distance_threshold_m: float = 0.05,
            num_frames_history:int = 30,
    ):
        """
        Initialize the PcAccumulator.

        Args:
            gt_distance_threshold_m (float, optional): Maximum distance in meters to
                associate a detection with a ground truth point. Defaults to 0.05.
            num_frames_history (int, optional): The number of frames a point should
                persist in the accumulator before expiring. Defaults to 30.
        """

        self.gt_distance_threshold_m:float = gt_distance_threshold_m
        self.num_frames_history:int = num_frames_history

        # Collection of currently available points
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
        self.points = np.empty(shape=(0,4))
        self.gt_points = np.empty(shape=(0,4))

        if new_points.shape[0] > 0:
            self.add_points(
                new_points=new_points,
                new_gt_points=new_gt_points
            )
        return

    def add_points(self, new_points: np.ndarray, new_gt_points: np.ndarray=np.empty(shape=(0,3))):
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
        
        #prune any expired detected points
        if self.points.shape[0] > 0:

            #decay the detection history by a step
            self.points[:,3] = \
                self.points[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.points[:,3] > 0
            self.points = self.points[valid_idxs]
        
        #prune any expired gt points
        if self.gt_points.shape[0] > 0:

            #decay the detection history by a step
            self.gt_points[:,3] = \
                self.gt_points[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.gt_points[:,3] > 0
            self.gt_points = self.gt_points[valid_idxs]
        
        # Append new points to the existing collection
        new_points = np.hstack((
            new_points,
            np.zeros(shape=
                        (new_points.shape[0],1))
        ))
        new_points[:,3] = self.num_frames_history
        self.points = np.vstack((self.points, new_points))
        
        if new_gt_points.shape[0] > 0:
            new_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points[:,0:3],
                gt_points=new_gt_points,
                threshold=self.gt_distance_threshold_m
            )
            new_gt_points = np.hstack((
                new_gt_points,
                np.zeros(shape=
                            (new_gt_points.shape[0],1))
            ))
            new_gt_points[:,3] = self.num_frames_history
            self.gt_points = np.vstack((self.gt_points, new_gt_points))

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
        self.points[:,0:3] = transformation.apply_transformation(self.points[:,0:3])

        #handle the ground truth points
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
            return self.points
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
            return self.gt_points
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
        
        if self.points.shape[0] > 0:
                
            nodes = self.points.copy()

            if normalize_frames:
                nodes[:,3] = nodes[:,3] / self.num_frames_history

            if self.gt_points.size > 0:
                tree = cKDTree(self.gt_points)
                dists, _ = tree.query(self.points, distance_upper_bound=1e-5)
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
