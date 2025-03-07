import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.pc_grid.pc_grid import PCGrid

class HistoricalPCGrid(PCGrid):

    def __init__(
            self,
            grid_resolution_m = 0.05,
            grid_max_distance_m = 10,
            num_frames_persistance:int=30):

        self.num_frames_persistance:int = num_frames_persistance

        super().__init__(grid_resolution_m, grid_max_distance_m)

        return
    
    def _reset_points(
            self,
            new_points:np.ndarray=np.empty(shape=(0,3)),
            new_gt_points:np.ndarray=np.empty(shape=(0,3))):
        """
        Reset the saved point cloud and optionally initialize it with new points.
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved point cloud with these points. 
                Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                saved ground truth point cloud with these points. 
                Defaults to an empty set of 3D points.
        """
        self.points:np.ndarray = np.zeros(shape=(0,4))
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
            new_gt_points: np.ndarray=np.empty(shape=(0,3))):
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
        if new_points.shape[1] != 3:
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
        new_points = self.filter_points_outside_grid(new_points)
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
                threshold=self.grid_resolution_m
            )
            new_gt_points = np.hstack((
                new_gt_points,
                np.zeros(shape=
                            (new_gt_points.shape[0],1))
            ))
            new_gt_points[:,3] = self.num_frames_persistance
            self.gt_points = np.vstack((self.gt_points, new_gt_points))

        # Update grid representation with new points
        self.grid = self._get_grid_from_points(self.points[:,0:3])

        self.gt_grid = self._get_grid_from_points(self.gt_points[:,0:3])
        self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)

    def apply_transformation(self, transformation):
        """
        Apply a coordinate transformation to the current set of points.

        Args:
            transformation (Transformation): Transformation to apply to the points.
        """
        self.points[:,0:3] = transformation.apply_transformation(self.points[:,0:3])

        #filter out points no longer in the grid
        self.points = self.filter_points_outside_grid(self.points)
        
        # Update the grid after transformation
        self.grid = self._get_grid_from_points(self.points[:,0:3])

        #handle the ground truth points
        if self.gt_points.shape[0] > 0:
            self.gt_points[:,0:3] = transformation.apply_transformation(self.gt_points[:,0:3])

            #filter out points no longer in the grid
            self.gt_points = self.filter_points_outside_grid(self.gt_points)
            
            # Update the grid after transformation
            self.gt_grid = self._get_grid_from_points(self.gt_grid[:,0:3])
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)

    


        

    
