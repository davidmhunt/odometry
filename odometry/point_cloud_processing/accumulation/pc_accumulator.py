import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter

class PcAccumulator:
    """Basic point cloud accumulator
    """
    def __init__(
            self,
            gt_distance_threshold_m: float = 0.05,
            num_frames_history:int = 30,
    ):

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
        Reset the saved point cloud and optionally initialize it with new points.
        NOTE: Must be updated by any child to change functionality
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved point cloud with these points. 
                Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                saved ground truth point cloud with these points. 
                Defaults to an empty set of 3D points.
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
        Add new points to the point cloud grid and update the grid representation.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.
            gt_points (np.ndarray, optional): Nx3 array of [x,y,z] ground 
                truth detections (if available). 
                Defaults to np.empty(shape=0,3).
        Raises:
            ValueError: If the input points do not have a shape of Nx3.
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
        new_points[:,3] = self.num_frames_persistance
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
            new_gt_points[:,3] = self.num_frames_persistance
            self.gt_points = np.vstack((self.gt_points, new_gt_points))

    def apply_transformation(self, transformation: Transformation):
        """
        Apply a coordinate transformation to the current set of points.

        Args:
            transformation (Transformation): Transformation to apply to the points.
        """
        self.points[:,0:3] = transformation.apply_transformation(self.points[:,0:3])

        # Update the grid after transformation
        self.grid = self._get_grid_from_points(self.points[:,0:3])

        #handle the ground truth points
        if self.gt_points.shape[0] > 0:

            self.gt_points[:,0:3] = transformation.apply_transformation(self.gt_points[:,0:3])
    
    def get_points(self, raw:bool = False)->np.ndarray:
        """Return a quantized set of points from the grid

        Returns:
            np.ndarray: Nx3 array of points obtained from the point cloud grid
        """
        if raw:
            return self.points
        else:
            return self.points[:,0:3]
    
    def get_gt_points(self, raw:bool = False)->np.ndarray:
        """Return a quantized set of points from the grid

        Returns:
            np.ndarray: Nx3 array of points obtained from the ground truth point cloud grid
        """
        if raw:
            return self.gt_points
        else:
            return self.gt_points[:,0:3]
    
    def get_nodes(self)->tuple:
        """Get the nodes and associated labels from a point cloud grid

        Returns:
            tuple: (nodes,labels), A tuple of an Nx4 array of points containing
              the (x,y,z,grid_value) for each point in the point cloud and a N-element
              array with the gt label for each node (if gt disabled, labels are all 0's) 
        """
        
        if self.points.shape[0] > 0:
                
            #encode all points as nodes with the "grid value" being the number of frames until the point expires
            nodes = self.points

            label = np.array(
                [np.any(np.all(p == self.gt_points, axis=1)) for p in self.points]
            ) if self.gt_points.size > 0 else np.zeros(len(self.points), dtype=bool)

            return nodes, label

        else:
            return np.empty(shape=(0, 4)),np.empty(shape=0)

    def _get_dets_close_to_gt_points(
            self,dets:np.ndarray,
            gt_points:np.ndarray,
            threshold:float=0.05)->np.ndarray:
        """Get ground truth points that are close to a given set of detections

        Args:
            dets (np.ndarray): Nx2 set of [x,y,z] detections
            gt_points (np.ndarray): Nx2 set of [x,y,z] ground truth detections
            threshold (float, optional): Euclidian distance to identify the
                corresponding ground truth detections. Defaults to 0.05.

        Raises:
            ValueError: If dets or 

        Returns:
            np.ndarray: Nx2 array of gt points that 
        """
        #append a column of grid detections to make detection array 2D
        if gt_points.shape[1] == 3 and dets.shape[1] == 3:
            
            #get the current set of points
            dists = np.linalg.norm(gt_points[:, None, :] - dets[None, :, :], axis=-1)  
            # Find points in gt_points that have at least one match in detected_pts within the threshold
            mask = np.any(dists <= threshold, axis=0)
            dets = dets[mask]

            return dets
        else:
            raise ValueError("Detections and ground truth detections must be a 3D point (3,) or an Nx3 array of points.")
