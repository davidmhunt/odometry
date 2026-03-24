import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.accumulation.grids._pc_grid import _PCGrid

class ProbabilisticPCGrid(_PCGrid):
    """
    A probabilistic grid accumulator that maintains a history of frames to compute occupancy probability.

    Unlike the standard _PCGrid which accumulates a single coalesced point cloud,
    this class maintains distinct point clouds for the last N frames. The occupancy
    grid is computed as the average occupancy across these frames.
    """

    def __init__(
            self,
            grid_resolution_m: float = 0.05,
            grid_max_distance_m: float = 3,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_history: int = 10,
            occupancy_threshold: float = 0.5):
        """
        Initialize the ProbabilisticPCGrid.

        Args:
            grid_resolution_m (float, optional): Resolution of the grid in meters.
                Defaults to 0.05.
            grid_max_distance_m (float, optional): Maximum distance from center in meters.
                Defaults to 3.
            valid_fovs_deg (list[tuple[float, float]], optional): A list of valid FOVs in degrees, e.g. [(-60, 60)].
                0 degrees is the +x axis, +90 degrees is the +y axis. Defaults to [(-180, 180)].
            num_frames_history (int, optional): Number of frames to average over.
                Defaults to 10.
            occupancy_threshold (float, optional): Threshold probability (0.0 to 1.0)
                to consider a cell occupied. Defaults to 0.5.
        """
        self.occupancy_threshold: float = occupancy_threshold
        
        # Initialize storage for history frames
        # usage of a list of arrays instead of a single array
        self.frames_list = []
        self.gt_frames_list = []

        super().__init__(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=grid_max_distance_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history
        )
        
        # Re-initialize lists after super() might have reset things differently
        self._reset_points()
    
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
        self._reset_points(new_points=new_points, new_gt_points=new_gt_points)

    def _reset_points(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Reset the saved point history and optionally initialize it.

        Args:
            new_points (np.ndarray, optional): Nx3 array of points.
            new_gt_points (np.ndarray, optional): Nx3 array of ground truth points.
        """
        # Initialize lists of empty arrays
        self.frames_list = [np.empty(shape=(0, 3)) for _ in range(self.num_frames_history)]
        self.gt_frames_list = [np.empty(shape=(0, 3)) for _ in range(self.num_frames_history)]

        if new_points.shape[0] > 0:
            self.add_points(
                new_points=new_points,
                new_gt_points=new_gt_points
            )

    def _reset_grid(
            self,
            new_points: np.ndarray = np.empty(shape=(0, 3)),
            new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Reset the probability grid.

        Args:
            new_points (np.ndarray, optional): Nx3 array of points.
            new_gt_points (np.ndarray, optional): Nx3 array of ground truth points.
        """
        # Setup point cloud grid
        if new_points.shape[0] > 0:
            # Calculate initial probability grid from the single frame
            self.grid = self._get_grid_from_points(new_points).astype(np.float32) \
                / float(self.num_frames_history)
        else:
            self.grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.float32
            )
        
        # Initialize ground truth grid
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
             # Recalculate matches
            matched_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m # Using grid_res to match original probabilistic behavior
            )
            self.gt_grid = self._get_grid_from_points(matched_gt_points).astype(np.float32)
            # Intersection logic
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.float32)
        else:
             self.gt_grid = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.float32
            )
            
    def add_points(self, new_points: np.ndarray, new_gt_points: np.ndarray = np.empty(shape=(0, 3))):
        """
        Add new points to the history and update the probability grid.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.
            new_gt_points (np.ndarray, optional): Nx3 array of [x, y, z] ground
                truth points. Defaults to empty.
        
        Raises:
            ValueError: If the input points do not have a shape of Nx3.
        """
        if new_points.shape[1] != 3:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")
        
        # Remove the oldest frame
        self.frames_list.pop()
        # Insert new frame at the beginning
        new_points = self.filter_points_outside_grid(new_points)
        self.frames_list.insert(0, new_points)

        # Handle ground truth
        self.gt_frames_list.pop()
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            matched_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m
            )
            self.gt_frames_list.insert(0, matched_gt_points)
        else:
            self.gt_frames_list.insert(0, np.empty(shape=(0, 3)))

        # Update the probability grid
        self.update_probability_grid()

    def apply_transformation(self, transformation: Transformation):
        """
        Apply a coordinate transformation to all historical frames.

        Args:
            transformation (Transformation): Transformation to apply.
        """
        # Transform all points in the history
        for i in range(len(self.frames_list)):
            
            # Detections
            if self.frames_list[i].shape[0] > 0:
                new_points = transformation.apply_transformation(self.frames_list[i])
                new_points = self.filter_points_outside_grid(new_points)
                self.frames_list[i] = new_points

            # Ground truth detections
            if self.gt_frames_list[i].shape[0] > 0:
                new_points = transformation.apply_transformation(self.gt_frames_list[i])
                new_points = self.filter_points_outside_grid(new_points)
                self.gt_frames_list[i] = new_points

        # Reset the probability grid based on transformed points
        self.update_probability_grid()

    def update_probability_grid(self):
        """
        Recompute the probability grid by averaging occupancy across history frames.
        """
        # Create an array of grids from points
        # Using list comprehension then sum
        grids = []
        for points in self.frames_list:
            if points.shape[0] > 0:
                grids.append(self._get_grid_from_points(points=points).astype(np.float32))
            else:
                grids.append(np.zeros((self.grid_bins.shape[0], self.grid_bins.shape[0]), dtype=np.float32))
        grids = np.array(grids)

        gt_grids = []
        for points in self.gt_frames_list:
            if points.shape[0] > 0:
                gt_grids.append(self._get_grid_from_points(points=points).astype(np.float32))
            else:
                gt_grids.append(np.zeros((self.grid_bins.shape[0], self.grid_bins.shape[0]), dtype=np.float32))
        gt_grids = np.array(gt_grids)

        # Sum all grids element-wise
        self.grid = np.sum(grids, axis=0)
        self.gt_grid = np.sum(gt_grids, axis=0)

        # Average the det grid
        if self.num_frames_history > 0:
            self.grid /= float(self.num_frames_history)
        
        self.gt_grid[self.gt_grid > 0] = 1.0
        self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.float32)

    def get_points(self, raw: bool = False) -> np.ndarray:
        """
        Return points from the grid where probability exceeds threshold.

        Args:
            raw (bool): Ignored for ProbabilisticPCGrid (always returns grid-derived points).
        
        Returns:
             np.ndarray: Nx3 array of points.
        """
        return self._get_points_from_pc_grid(
            pc_grid=(self.grid >= self.occupancy_threshold)
        )
    
    def get_gt_points(self, raw: bool = False) -> np.ndarray:
        """
        Return ground truth points from the grid.

        Args:
             raw (bool): Ignored.
        
        Returns:
             np.ndarray: Nx3 array of points.
        """
        return self._get_points_from_pc_grid(
            pc_grid=(self.gt_grid > 0)
        )
