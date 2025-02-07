import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter

class PCGrid:
    """Basic point cloud grid
    """
    def __init__(
            self,
            grid_resolution_m: float = 5e-2,
            grid_max_distance_m: float = 3
    ):
        """
        Initialize a 2D point cloud grid with a given resolution and max distance.
        
        Args:
            grid_resolution_m (float): Resolution of the grid in meters.
            grid_max_distance_m (float): Maximum distance to consider in meters.
        """
        # Grid parameters
        self.grid_resolution_m: float = grid_resolution_m
        self.grid_max_distance_m: float = grid_max_distance_m

        # Define the grid bins for storing samples
        self.grid_bins: np.ndarray = np.arange(
            start=-1 * self.grid_max_distance_m,
            stop=self.grid_max_distance_m + self.grid_resolution_m,
            step=self.grid_resolution_m
        )

        # Initialize an empty grid with integer values (0: no point, 1: point present)
        self.grid:np.ndarray = None

        # Collection of currently available points
        self.points:np.ndarray = None
        
        self.reset()

    def _reset_grid(self,new_points:np.ndarray = np.empty(shape=(0,3))):
        """
        Reset the saved grid and optionally initialize it with new points.
        NOTE: Must be updated by any child to change functionality
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
    
    def _reset_points(self,new_points:np.ndarray=np.empty(shape=(0,3))):
        """
        Reset the saved point cloud and optionally initialize it with new points.
        NOTE: Must be updated by any child to change functionality
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved point cloud with these points. 
                Defaults to an empty set of 3D points.
        """
        self.points =  new_points
        return
    
    def reset(self, new_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Reset the point cloud grid and optionally initialize it with new points.

        Args:
            new_points (np.ndarray, optional): If provided, initializes the grid 
                with these points. Defaults to an empty set of 3D points.
        """

        self._reset_points(new_points)

        self._reset_grid(new_points)

        return

    def add_points(self, new_points: np.ndarray):
        """
        Add new points to the point cloud grid and update the grid representation.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.

        Raises:
            ValueError: If the input points do not have a shape of Nx3.
        """
        if new_points.shape[1] != 3:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")
        
        # Append new points to the existing collection
        new_points = self.filter_points_outside_grid(new_points)
        self.points = np.vstack((self.points, new_points))
        
        # Update grid representation with new points
        self.grid = self._get_grid_from_points(self.points)

    def apply_transformation(self, transformation: Transformation):
        """
        Apply a coordinate transformation to the current set of points.

        Args:
            transformation (Transformation): Transformation to apply to the points.
        """
        self.points = transformation.apply_transformation(self.points)

        #filter out points no longer in the grid
        self.points = self.filter_points_outside_grid(self.points)
        
        # Update the grid after transformation
        self.grid = self._get_grid_from_points(self.points)
    
    def get_points(self)->np.ndarray:
        """Return a quantized set of points from the grid

        Returns:
            np.ndarray: Nx3 array of points obtained from the point cloud grid
        """
        return self._get_points_from_pc_grid(self.grid)

    def _get_grid_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Convert a set of 3D points into a 2D grid representation.

        Args:
            points (np.ndarray): Nx3 array of [x, y, z] point cloud points.

        Returns:
            np.ndarray: MxM point cloud grid, where each index represents a grid bin.
        """
        if points.shape[1] == 3:
            ret_grid = np.zeros(
                shape=(self.grid_bins.shape[0], self.grid_bins.shape[0]),
                dtype=np.int8
            )
            
            # Find the closest grid bin indices for x and y coordinates
            x_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 0]), axis=0)
            y_idx = np.argmin(np.abs(self.grid_bins[:, None] - points[:, 1]), axis=0)

            # Mark the grid cells as occupied (1)
            ret_grid[x_idx, y_idx] = 1
        else:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")

        return ret_grid
    
    def _get_points_from_pc_grid(self, pc_grid: np.ndarray) -> np.ndarray:
        """
        Convert a point cloud grid back into an array of points.

        Args:
            pc_grid (np.ndarray): NxN grid representation where 1 indicates a point exists
                at that location, and 0 indicates no point is present.

        Returns:
            np.ndarray: Nx3 array of points reconstructed from the grid.
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
            return np.empty(shape=(0, 3))  # Return an empty array if no points exist
        
    def filter_points_outside_grid(self,points)->np.ndarray:
        """
        Remove points from an Nx3 array where any x, y, or z coordinate is outside of the grid.
        
        Args:
            points (np.ndarray): Nx3 array of [x, y, z] points. 4th dimmensions
                are returned but have no impact
        
        Returns:
            np.ndarray: Filtered array of points within the grid
        """
        return points[np.all(np.abs(points[:,0:3]) <= self.grid_max_distance_m,axis=1)]
