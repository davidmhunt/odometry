import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.accumulation.grids._pc_grid import _PCGrid

class TemporalDensityPCGrid(_PCGrid):
    """
    A grid-based point cloud accumulator that records log-density and temporal data.

    Leverages `_PCGrid` with `density_grid_enabled=True` to maintain the density grid
    in `self.grid`. Additionally maintains `self.temporal_grid` for storing the 
    most recent `frames_remaining` value of detections in each cell.
    """
    def __init__(
            self,
            grid_resolution_m: float = 0.05,
            grid_max_distance_m: float = 3,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_history: int = 30,
            num_frames_history_gt: int = 1,
            subsample_percentage: float = 1.0,
            gt_distance_threshold_m: float = None,
            **kwargs
    ):
        """
        Initialize the TemporalDensityPCGrid.

        Args:
            grid_resolution_m (float, optional): Resolution of the grid in meters.
            grid_max_distance_m (float, optional): Maximum distance from center.
            valid_fovs_deg (list[tuple[float, float]], optional): FOVs list.
            num_frames_history (int, optional): Frames remaining to persist.
            num_frames_history_gt (int, optional): Frames remaining for gt.
            subsample_percentage (float, optional): Percentage of new points to keep.
                Should be between 0.0 and 1.0. Defaults to 1.0 (no subsampling).
            gt_distance_threshold_m (float, optional): Distance threshold for gt.
            **kwargs: Additional parameters for PcAccumulator.
        """
        self.temporal_grid: np.ndarray = None

        super().__init__(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=grid_max_distance_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            num_frames_history_gt=num_frames_history_gt,
            subsample_percentage=subsample_percentage,
            density_grid_enabled=True,
            gt_distance_threshold_m=gt_distance_threshold_m,
            **kwargs
        )
        
    def _update_grids(self):
        """
        Overrides the base update to also include temporal grid logic.
        """
        super()._update_grids()
        
        # Initialize temporal grid with same shape as base grid
        t_grid = np.zeros_like(self.grid, dtype=np.float32)

        if self.points is not None and self.points.shape[0] > 0:
            pts = self.points
            if pts.shape[1] >= 4:
                # Find the closest grid bin indices for x and y coordinates
                x_idx = np.argmin(np.abs(self.grid_bins[:, None] - pts[:, 0]), axis=0)
                y_idx = np.argmin(np.abs(self.grid_bins[:, None] - pts[:, 1]), axis=0)
                
                # We want maximum frames_remaining for each cell
                np.maximum.at(t_grid, (x_idx, y_idx), pts[:, 3])
                
        self.temporal_grid = t_grid

    def get_points(self, raw: bool = False) -> np.ndarray:
        """
        Retrieve points reconstructed from the grid.
        
        Args:
            raw (bool, optional): If True, returns raw accumulator points.
        """
        if raw:
            return super().get_points(raw=True)
        else:
            return self._get_points_from_pc_grid(self.grid > 0)

    def get_nodes(self, normalize_frames: bool = False, raw: bool = False) -> tuple:
        """
        Retrieve points as nodes with ground truth labels.

        Args:
            normalize_frames (bool, optional): If True, normalizes the frames
                remaining by the total number of frames history.
            raw (bool, optional): Uses raw accumulator points if True.

        Returns:
            tuple: (nodes, labels).
                - nodes (np.ndarray): Nx5 array of [x, y, z=0, density, temporal].
                - labels (np.ndarray): Boolean array.
        """
        if raw:
            return super().get_nodes(normalize_frames=normalize_frames, raw=True)
        else:
            x_idxs, y_idxs = np.nonzero(self.grid > 0)
            
            if x_idxs.shape[0] > 0:
                x_vals = self.grid_bins[x_idxs]
                y_vals = self.grid_bins[y_idxs]
                z_vals = np.zeros_like(x_vals)
                density_vals = self.grid[x_idxs, y_idxs]
                temporal_vals = self.temporal_grid[x_idxs, y_idxs]
                
                if normalize_frames and self.num_frames_history > 0:
                    temporal_vals = temporal_vals / float(self.num_frames_history)
                    
                nodes = np.column_stack((x_vals, y_vals, z_vals, density_vals, temporal_vals))
                labels = self.gt_grid[x_idxs, y_idxs] > 0
                return nodes, labels
            else:
                return np.empty(shape=(0, 5)), np.empty(shape=0)
