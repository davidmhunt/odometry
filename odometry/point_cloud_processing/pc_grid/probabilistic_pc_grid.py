import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.pc_grid.pc_grid import PCGrid

class ProbabilisticPCGrid(PCGrid):

    def __init__(
            self,
            grid_resolution_m = 0.05,
            grid_max_distance_m = 3,
            num_frames_history:int=10,
            occupancy_threshold:float=0.5):

        self.num_frames_history:int = num_frames_history
        self.threshold:float = occupancy_threshold

        super().__init__(grid_resolution_m, grid_max_distance_m)

        return
    
    def _reset_points(self, new_points = np.empty(shape=(0,3))):
        """
        Reset the point cloud grid and optionally initialize it with new points.

        Args:
            new_points (np.ndarray, optional): If provided, initializes the grid 
                with these points. Defaults to an empty set of 3D points.
        """
        self.points:list = [np.empty(shape=(0,3)) for _ in range(self.num_frames_history)]

        new_points = self.filter_points_outside_grid(new_points)

        self.points[0] = new_points

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
            self.grid = self._get_grid_from_points(new_points).astype(np.float32) \
                / float(self.num_frames_history)
        else:
            # Reset grid to empty state
            self.grid: np.ndarray = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.float32
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
        
        #remove the last element of the array
        self.points.pop()
        self.points.insert(0,new_points)

        #update the probability grid
        self.update_probability_grid()

    def apply_transformation(self, transformation):
        """
        Apply a coordinate transformation to the current set of points.

        Args:
            transformation (Transformation): Transformation to apply to the points.
        """
        #transform all points in the grid, then update the probability grid
        for i in range(self.num_frames_history):
            new_points = transformation.apply_transformation(self.points[i])
            new_points = self.filter_points_outside_grid(new_points)
            self.points[i] = new_points
        
        #then reset the probability grid
        self.update_probability_grid()
    
    def get_points(self):
        return self._get_points_from_pc_grid(
            pc_grid=(self.grid >= self.threshold)
        )

    def update_probability_grid(self):
        
        # reset the grid
        self._reset_grid()

        # create an array of grids from points
        grids = np.array([
            self._get_grid_from_points(points=self.points[i]).astype(np.float32)\
                for i in range(self.num_frames_history)
            ])

        # sum all grids element-wise
        self.grid = np.sum(grids, axis=0)

        # average the grid
        self.grid /= float(self.num_frames_history)

    


        

    
