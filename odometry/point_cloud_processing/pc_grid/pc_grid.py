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
        self.gt_grid:np.ndarray = None

        # Collection of currently available points
        self.points:np.ndarray = None
        self.gt_points:np.ndarray = None
        
        self.reset()

    def _reset_grid(
            self,
            new_points:np.ndarray = np.empty(shape=(0,3)),
            new_gt_points:np.ndarray = np.empty(shape=(0,3))):
        """
        Reset the saved grid and optionally initialize it with new points.
        NOTE: Must be updated by any child to change functionality
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved grid with these points. 
                Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                saved ground truth point cloud with these points. 
                Defaults to an empty set of 3D points.
        """
        #initialize point cloud grid
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
        #initialize ground truth grid
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            new_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m
            )
            self.gt_grid = self._get_grid_from_points(new_gt_points)
        else:
            # Reset grid to empty state
            self.gt_gridgrid: np.ndarray = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )
        return
    
    def _reset_points(
            self,
            new_points:np.ndarray=np.empty(shape=(0,3)),
            new_gt_points:np.ndarray=np.empty(shape=(0,3))):
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
        self.points = np.empty(shape=(0,3))
        self.gt_points = np.empty(shape=(0,3))

        if new_points.shape[0] > 0:
            self.add_points(
                new_points=new_points,
                new_gt_points=new_gt_points
            )

        return
    
    def reset(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points:np.ndarray = np.empty(shape=(0,3))):
        """
        Reset the point cloud grid and optionally initialize it with new points.

        Args:
            new_points (np.ndarray, optional): If provided, initializes the grid 
                with these points. Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                ground truth grid with these ground truth points. 
                Defaults to an empty set of 3D points.
        """

        self._reset_points(new_points,new_gt_points)

        self._reset_grid(new_points,new_gt_points)

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
        
        # Append new points to the existing collection
        new_points = self.filter_points_outside_grid(new_points)
        self.points = np.vstack((self.points, new_points))
        
        # Update grid representation with new points
        self.grid = self._get_grid_from_points(self.points)

        #handle ground truth detections
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            new_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m
            )

            self.gt_points = np.vstack((self.gt_points, new_gt_points))

            self.gt_grid = self._get_grid_from_points(self.gt_points)
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)

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

        #handle the ground truth points
        if self.gt_points.shape[0] > 0:
            self.gt_points = transformation.apply_transformation(self.gt_points)

            #filter out points no longer in the grid
            self.gt_points = self.filter_points_outside_grid(self.gt_points)
            
            # Update the grid after transformation
            self.gt_grid = self._get_grid_from_points(self.gt_grid)
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)
    
    def get_points(self)->np.ndarray:
        """Return a quantized set of points from the grid

        Returns:
            np.ndarray: Nx3 array of points obtained from the point cloud grid
        """
        return self._get_points_from_pc_grid(self.grid)
    
    def get_gt_points(self)->np.ndarray:
        """Return a quantized set of points from the grid

        Returns:
            np.ndarray: Nx3 array of points obtained from the ground truth point cloud grid
        """
        return self._get_points_from_pc_grid(self.gt_grid)
    
    def get_nodes(self)->tuple:
        """Get the nodes and associated labels from a point cloud grid

        Returns:
            tuple: (nodes,labels), A tuple of an Nx4 array of points containing
              the (x,y,z,grid_value) for each point in the point cloud and a N-element
              array with the gt label for each node (if gt disabled, labels are all 0's) 
        """
        
        # Identify the indices of occupied grid cells
        x_idxs, y_idxs = np.nonzero(self.grid)

        if x_idxs.shape[0] > 0:
            # Convert grid indices back to coordinate values
            x_vals = self.grid_bins[x_idxs]
            y_vals = self.grid_bins[y_idxs]
            z_vals = np.zeros_like(x_vals)  # Assume z=0 as it’s a 2D representation
            grid_values = self.grid[x_idxs, y_idxs]

            nodes = np.column_stack((x_vals, y_vals, z_vals,grid_values))
            labels = self.gt_grid[x_idxs,y_idxs]

            return nodes,labels
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
    
    def _get_gt_grid_dets_only(self, gt_points:np.ndarray)->np.ndarray:
        """Compute a "ground truth" grid of lidar detections in the detection region.
        Only generates a ground truth grid of points corresponding to locations that the 
        radar actually detected objects at though. Defaults to True

        Args:
            gt_points (np.ndarray): Nx3 array of lidar detections captured from the
                same location and at the same time as the most recent radar data

        Returns:
            np.ndarray: grid
        """

        #append a column of grid detections to make detection array 2D
        if gt_points.shape[1] == 3:

            #get a grid from the ground truth points
            grid_gt_raw = self._get_grid_from_points(gt_points)

            return ((self.grid > 0) & (grid_gt_raw > 0)).astype(np.int8)
        else:
            raise ValueError("Ground truth points must be a 3D point (3,) or an Nx3 array of points.")


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
    
    def _get_nodes_from_pc_grid(self, pc_grid: np.ndarray) -> np.ndarray:
        """
        Get a set of nodes from a given point cloud grid

        Args:
            pc_grid (np.ndarray): NxN grid representation where 1 indicates a point exists
                at that location, and 0 indicates no point is present.

        Returns:
            np.ndarray: Nx4 array of points reconstructed from the grid containing the 
                (x,y,z,grid_value) for all points in a grid
        """
        # Identify the indices of occupied grid cells
        x_idxs, y_idxs = np.nonzero(pc_grid)

        if x_idxs.shape[0] > 0:
            # Convert grid indices back to coordinate values
            x_vals = self.grid_bins[x_idxs]
            y_vals = self.grid_bins[y_idxs]
            z_vals = np.zeros_like(x_vals)  # Assume z=0 as it’s a 2D representation
            grid_values = pc_grid[x_idxs, y_idxs]

            return np.column_stack((x_vals, y_vals, z_vals,grid_values))
        else:
            return np.empty(shape=(0, 4))  # Return an empty array if no points exist

        
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
