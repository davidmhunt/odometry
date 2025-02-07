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
    
    def _reset_points(self, new_points = np.empty(shape=(0,3))):
        """
        Reset the point cloud grid and optionally initialize it with new points.

        Args:
            new_points (np.ndarray, optional): If provided, initializes the grid 
                with these points. Defaults to an empty set of 3D points.
        """
        self.points:np.ndarray = np.zeros(shape=(0,4))
            
        new_points = self.filter_points_outside_grid(new_points)

        if new_points.shape[0] > 0:
            new_points = np.hstack((
                new_points,
                np.zeros(shape=
                         (new_points.shape[0],1))
            ))
            new_points[:,3] = self.num_frames_persistance

        return
    
    def _reset_grid(self, new_points = np.empty(shape=(0,3))):
        """
        Reset the saved grid and optionally initialize it with new points.
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved grid with these points. 
                Defaults to an empty set of 3D points.
        """

        if new_points.shape[0] > 0:
            self.grid = self._get_grid_from_points(new_points)
        else:
            # Reset grid to empty state
            self.grid: np.ndarray = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )
        return
    
    def add_points(self, new_points):
        """
        Add new points to the point cloud grid and update the probability map

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.

        Raises:
            ValueError: If the input points do not have a shape of Nx3.
        """
        if new_points.shape[1] != 3:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")
        
        #prune any expired points
        if self.points.shape[0] > 0:

            #decay the detection history by a step
            self.points[:,3] = \
                self.points[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.points[:,3] > 0
            self.points = self.points[valid_idxs]

        # Append new points to the existing collection
        new_points = self.filter_points_outside_grid(new_points)
        new_points = np.hstack((
            new_points,
            np.zeros(shape=
                        (new_points.shape[0],1))
        ))
        new_points[:,3] = self.num_frames_persistance


        self.points = np.vstack((self.points, new_points))
        
        # Update grid representation with new points
        self.grid = self._get_grid_from_points(self.points[:,0:3])

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
    
    def get_points(self):
        return self._get_points_from_pc_grid(self.grid)

    


        

    
