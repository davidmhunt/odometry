from odometry.point_cloud_processing.accumulation.grids._pc_grid import _PCGrid

class HistoricalPCGrid(_PCGrid):
    """
    A grid-based point cloud accumulator that maintains points over a historical window.

    This class serves as a compatibility wrapper around _PCGrid, providing the
    `num_frames_persistance` interface used in previous implementations.
    """
    def __init__(
            self,
            grid_resolution_m: float = 0.05,
            grid_max_distance_m: float = 10,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_persistance: int = 30,
            subsample_percentage: float = 1.0,
            **kwargs
    ):
        """
        Initialize the HistoricalPCGrid.

        Args:
            grid_resolution_m (float, optional): Resolution of the grid in meters.
                Defaults to 0.05.
            grid_max_distance_m (float, optional): Maximum distance from center in meters.
                Defaults to 10.
            valid_fovs_deg (list[tuple[float, float]], optional): A list of valid FOVs in degrees, e.g. [(-60, 60)].
                0 degrees is the +x axis, +90 degrees is the +y axis. Defaults to [(-180, 180)].
            num_frames_persistance (int, optional): Number of frames to persist points.
                Defaults to 30.
            subsample_percentage (float, optional): Percentage of new points to keep.
                Should be between 0.0 and 1.0. Defaults to 1.0 (no subsampling).
        """
        # Map num_frames_persistance to num_frames_history for the base class
        self.num_frames_persistance = num_frames_persistance
        super().__init__(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=grid_max_distance_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_persistance,
            subsample_percentage=subsample_percentage,
            **kwargs
        )
