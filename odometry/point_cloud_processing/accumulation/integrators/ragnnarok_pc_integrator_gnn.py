import numpy as np

from geometries.pose.pose import Pose

from odometry.point_cloud_processing.accumulation.grids.probabilistic_pc_grid_gnn import ProbabilisticPCGridGNN
from odometry.point_cloud_processing.accumulation.grids.historical_pc_grid import HistoricalPCGrid
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.point_cloud_processing.accumulation.pc_accumulator import PcAccumulator

from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner

class RagnnarokPointCloudIntegratorGNN(_PointCloudIntegrator):
    """
    Integrates point clouds using both probabilistic and historical grid approaches.

    Inherits from _PointCloudIntegrator and extends it by maintaining two specialized
    grid representations: a ProbabilisticPCGridGNN for probabilistic occupancy with
    GNN filtering, and a HistoricalPCGrid for raw point persistence.

    Attributes:
        probabilistic_pc_grid (ProbabilisticPCGridGNN): Grid for probabilistic
            occupancy mapping.
        historical_pc_grid (HistoricalPCGrid): Grid for historical point persistence.
    """

    def __init__(
            self,
            gnn_runner: GNNRunner,
            grid_resolution_m_prob: float = 0.10,
            grid_max_distance_m_prob: float = 5.0,
            num_frames_history_prob: int = 20,
            grid_resolution_m_hist: float = 0.10,
            grid_max_distance_m_hist: float = 5.0,
            num_frames_history_hist: int = 20,
            min_detection_radius: float = 0.25,
            max_detection_radius: float = 20.0,
    ) -> None:
        """
        Initialize the RagnnarokPointCloudIntegrator.

        Args:
            gnn_runner (GNNRunner): the GNN runner instance used for point classification
                in the probabilistic grid.
            grid_resolution_m_prob (float, optional): Resolution for the probabilistic
                grid. Defaults to 0.10.
            grid_max_distance_m_prob (float, optional): Max distance for the
                probabilistic grid. Defaults to 5.0.
            num_frames_history_prob (int, optional): History length for the
                probabilistic grid. Defaults to 20.
            grid_resolution_m_hist (float, optional): Resolution for the historical
                grid. Defaults to 0.10.
            grid_max_distance_m_hist (float, optional): Max distance for the
                historical grid. Defaults to 5.0.
            num_frames_history_hist (int, optional): History length for the
                historical grid. Defaults to 20.
            min_detection_radius (float, optional): Minimum point detection radius.
                Defaults to 0.25.
            max_detection_radius (float, optional): Maximum point detection radius.
                Defaults to 20.0.
        """
        
        super().__init__(
            gt_distance_threshold_m=grid_resolution_m_prob,
            num_frames_history=num_frames_history_hist + num_frames_history_prob,
            min_detection_radius=min_detection_radius,
            max_detection_radius=max_detection_radius
        )
        
        #probabilistic point grid for initial detections
        self.probabilistic_pc_grid:ProbabilisticPCGridGNN = \
            ProbabilisticPCGridGNN(
                runner=gnn_runner,
                grid_resolution_m=grid_resolution_m_prob,
                grid_max_distance_m=grid_max_distance_m_prob,
                num_frames_history=num_frames_history_prob,
                occupancy_threshold=0.20
            )      
        self.historical_pc_grid:HistoricalPCGrid = HistoricalPCGrid(
            grid_resolution_m=grid_resolution_m_hist,
            grid_max_distance_m=grid_max_distance_m_hist,
            num_frames_persistance=num_frames_history_hist
        )

    def reset(self): 
        """
        Reset the integrator state.

        Resets the base class integrator state as well as the probabilistic and
        historical grids.
        """
        super().reset()
        self.probabilistic_pc_grid.reset()
        self.historical_pc_grid.reset()

    def add_points(
            self,
            static_points: np.ndarray,
            current_pose: Pose,
            gt_points: np.ndarray = np.empty(shape=(0, 3))
    ):
        """
        Add new points to the integrator tables.

        Updates the base class history (which computes transformations), then
        updates the probabilistic and historical grids with the transformed
        points.

        Args:
            static_points (np.ndarray): Nx3 detection points.
            current_pose (Pose): Current vehicle pose.
            gt_points (np.ndarray, optional): Ground truth points. Defaults to
                empty.
        """
        
        # Call parent to update raw history and calculate transformation
        super().add_points(
            static_points=static_points,
            current_pose=current_pose,
            gt_points=gt_points
        )

        if self.current_transformation:
            
            #add points to the probabilistic point cloud grid
            self.probabilistic_pc_grid.apply_transformation(self.current_transformation)
            self.probabilistic_pc_grid.add_points(
                new_points=self.current_static_points[:,:3], #only send x,y,z (not vel)
                new_gt_points=gt_points) 

            #add points to the historical point cloud grid
            self.historical_pc_grid.apply_transformation(self.current_transformation)
            self.historical_pc_grid.add_points(
                new_points=self.probabilistic_pc_grid.get_points(),
                new_gt_points=gt_points
            )

    def get_points(self) -> np.ndarray:
        """
        Retrieve points from the historical grid.

        Returns:
            np.ndarray: Points from the historical grid.
        """
        
        return self.historical_pc_grid.get_points()
    
    def get_nodes(self, **kwargs) -> tuple:
        """
        Retrieve nodes and labels from the probabilistic grid.

        Returns:
            tuple: (nodes, labels) from the probabilistic grid.
        """
        
        return self.probabilistic_pc_grid.get_nodes()