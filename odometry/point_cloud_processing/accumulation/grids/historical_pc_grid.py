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
            num_frames_persistance: int = 30):
        """
        Initialize the HistoricalPCGrid.

        Args:
            grid_resolution_m (float, optional): Resolution of the grid in meters.
                Defaults to 0.05.
            grid_max_distance_m (float, optional): Maximum distance from center in meters.
                Defaults to 10.
            num_frames_persistance (int, optional): Number of frames to persist points.
                Defaults to 30.
        """
        # Map num_frames_persistance to num_frames_history for the base class
        self.num_frames_persistance = num_frames_persistance
        super().__init__(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=grid_max_distance_m,
            num_frames_history=num_frames_persistance
        )
