import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator

class _PCGrid(PcAccumulator):
    """
    A grid-based point cloud accumulator that maintains a 2D occupancy grid.

    Inherits from PcAccumulator to manage point storage and persistence, while
    adding functionality to represent the points as a fixed-resolution 2D grid.
    Enforces grid boundaries by filtering points that fall outside the defined region.

    Attributes:
        grid_resolution_m (float): The resolution of the grid cells in meters.
        grid_max_distance_m (float): The maximum extension of the grid in meters (half-width).
        grid_bins (np.ndarray): Array of bin edges for the grid.
        grid (np.ndarray): 2D integer array representing the occupancy grid (NxN).
        gt_grid (np.ndarray): 2D integer array representing the ground truth occupancy grid (NxN).
    """
    def __init__(
            self,
            grid_resolution_m: float = 5e-2,
            grid_max_distance_m: float = 3,
            num_frames_history: int = 30,
            num_frames_history_gt: int = 30,
    ):
        """
        Initialize the _PCGrid.

        Args:
            grid_resolution_m (float, optional): Resolution of the grid in meters.
                Defaults to 5e-2.
            grid_max_distance_m (float, optional): Maximum distance from center in meters.
                Defaults to 3.
            num_frames_history (int, optional): Number of frames to persist points.
                Defaults to 30.
            num_frames_history_gt (int, optional): Number of frames to persist gt points.
                Defaults to 1.
        """

        # Initialize grids
        self.grid: np.ndarray = None
        self.gt_grid: np.ndarray = None
        
        super().__init__(
            gt_distance_threshold_m=grid_resolution_m,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            grid_resolution_m=grid_resolution_m,
            max_detection_range=grid_max_distance_m
        )

    def reset(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Reset the accumulator and grid, optionally seeding with new points.

        Clears stored points via the parent reset, and resets the grid arrays.

        Args:
            new_points (np.ndarray, optional): Nx3 array of [x, y, z] points to
                initialize with. Defaults to empty.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground
                truth points to initialize with. Defaults to empty.
        """
        # Reset the accumulator points
        super().reset(new_points=new_points, new_gt_points=new_gt_points)
        
        # Reset the grid representation
        self._reset_grid(new_points=new_points, new_gt_points=new_gt_points)

    def _reset_grid(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Internal method to reset the grid arrays and populate if points correspond.

        Args:
            new_points (np.ndarray, optional): Nx3 array of [x, y, z] points.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground truth points.
        """
        # Initialize point cloud grid
        if new_points.shape[0] > 0:
            self.grid = self._get_grid_from_points(new_points)
        else:
            self.grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )

        # Initialize ground truth grid
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            # Recalculate matches for grid initialization
            matched_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.gt_distance_threshold_m
            )
            self.gt_grid = self._get_grid_from_points(matched_gt_points)
        else:
             self.gt_grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )

    def add_points(self,
                    new_points: np.ndarray = np.empty(shape=(0, 3)),
                   new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Add new points to the accumulator and update the grid.

        Points falling outside the grid boundaries are filtered out before being
        added to the accumulator.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground
                truth points. Defaults to empty.
        """
        # Filter points to ensure they fit in the grid
        new_points = self.filter_points_outside_grid(new_points)
        
        # Add to the accumulator (handles persistence and storage)
        super().add_points(new_points=new_points, new_gt_points=new_gt_points)
        
        # Update grid representations from the stored points
        # usage of [:, 0:3] serves to drop the timestamp column
        if self.points.shape[0] > 0:
            self.grid = self._get_grid_from_points(self.points[:, 0:3])
        else:
            self.grid = np.zeros_like(self.grid)

        if self.gt_points.shape[0] > 0:
             self.gt_grid = self._get_grid_from_points(self.gt_points[:, 0:3])
             self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)
        else:
            self.gt_grid = np.zeros_like(self.gt_grid)

    def apply_transformation(self, transformation: Transformation):
        """
        Apply a rigid body transformation to the points and update the grid.

        After transformation, points that fall outside the grid boundaries are
        removed.

        Args:
            transformation (Transformation): Transformation to apply.
        """
        # Apply transformation to the internal points via parent implementation
        super().apply_transformation(transformation)

        # Filter out points no longer in the grid
        if self.points_raw.shape[0] > 0:
            self.points_raw = self.filter_points_outside_grid(self.points_raw)
            
        if self.points.shape[0] > 0:
            self.points = self.filter_points_outside_grid(self.points)
        
        if self.gt_points_raw.shape[0] > 0:
            self.gt_points_raw = self.filter_points_outside_grid(self.gt_points_raw)
            
        if self.gt_points.shape[0] > 0:
            self.gt_points = self.filter_points_outside_grid(self.gt_points)

        # Re-compute the grid from the valid, transformed points
        if self.points.shape[0] > 0:
            self.grid = self._get_grid_from_points(self.points[:, 0:3])
        else:
            # It's possible all points moved out of frame
            self.grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )

        if self.gt_points is not None and self.gt_points.shape[0] > 0:
            self.gt_grid = self._get_grid_from_points(self.gt_points[:, 0:3])
            # Intersection logic as per PCGrid
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.int8)
        else:
             self.gt_grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )

    def get_points(self, raw: bool = False) -> np.ndarray:
        """
        Retrieve points, either raw from the accumulator or quantized from the grid.

        Args:
            raw (bool, optional): If True, returns the raw points from the accumulator
                (including sub-grid precision and timestamps).
                If False, returns points reconstructed from the grid cell centers.
                Defaults to False.

        Returns:
            np.ndarray: Nx3 (or Nx4 if raw=True) array of points.
        """
        if raw:
            return super().get_points(raw=True)
        else:
            return self._get_points_from_pc_grid(self.grid)

    def get_gt_points(self, raw: bool = False) -> np.ndarray:
        """
        Retrieve ground truth points, either raw or quantized.

        Args:
            raw (bool, optional): If True, returns raw gt points.
                If False, returns points reconstructed from the gt grid.
                Defaults to False.

        Returns:
            np.ndarray: Nx3 (or Nx4 if raw=True) array of points.
        """
        if raw:
            return super().get_gt_points(raw=True)
        else:
            return self._get_points_from_pc_grid(self.gt_grid)

    def _get_nodes_from_pc_grid(self, pc_grid: np.ndarray) -> np.ndarray:
        """
        Get a set of nodes (x, y, z, value) from a grid.

        Args:
            pc_grid (np.ndarray): NxN grid representation.

        Returns:
            np.ndarray: Nx4 array of nodes and values.
        """
        # Identify the indices of occupied grid cells
        x_idxs, y_idxs = np.nonzero(pc_grid)

        if x_idxs.shape[0] > 0:
            # Convert grid indices back to coordinate values
            x_vals = self.grid_bins[x_idxs]
            y_vals = self.grid_bins[y_idxs]
            z_vals = np.zeros_like(x_vals)  # Assume z=0 as it’s a 2D representation
            grid_values = pc_grid[x_idxs, y_idxs]
            
            return np.column_stack((x_vals, y_vals, z_vals, grid_values))
        else:
             return np.empty(shape=(0, 4))
             
    
    def get_nodes(self, raw: bool = False) -> tuple:
        """
        Retrieve points as nodes with ground truth labels.

        Args:
            raw (bool, optional): If True, uses the raw point cloud data.
                If False, uses the quantized grid data. Defaults to False.

        Returns:
            tuple: (nodes, labels).
                - nodes: Nx4 array of [x, y, z, value].
                - labels: Boolean array of ground truth labels.
        """
        if raw:
            return super().get_nodes()
        else:
            # Reconstruct nodes and labels from the grid
            nodes = self._get_nodes_from_pc_grid(self.grid)
            
            if nodes.shape[0] > 0:
                # We need to extract labels corresponding to these nodes from the gt_grid
                x_idxs, y_idxs = np.nonzero(self.grid)
                labels = self.gt_grid[x_idxs, y_idxs]
                return nodes, labels
            else:
                return np.empty(shape=(0, 4)), np.empty(shape=0)
