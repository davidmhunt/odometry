from odometry.point_cloud_processing.accumulation.grids.probabilistic_pc_grid import ProbabilisticPCGrid

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner

class ProbabilisticPCGridGNN(ProbabilisticPCGrid):
    """
    Probabilistic point cloud grid that uses a GNN to filtering/predict points.
    
    Inherits from ProbabilisticPCGrid but restricts point retrieval to those
    predicted by a Graph Neural Network (GNN).
    """
    def __init__(
            self,
            runner: GNNRunner,
            grid_resolution_m: float = 0.05,
            grid_max_distance_m: float = 3,
            valid_fovs_deg: list[tuple[float, float]] = [(-180, 180)],
            num_frames_history: int = 10,
            occupancy_threshold: float = 0.5):
        """
        Initialize the ProbabilisticPCGridGNN.

        Args:
            runner (GNNRunner): The GNN model runner to use for inference.
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
        super().__init__(
            grid_resolution_m=grid_resolution_m,
            grid_max_distance_m=grid_max_distance_m,
            valid_fovs_deg=valid_fovs_deg,
            num_frames_history=num_frames_history,
            occupancy_threshold=occupancy_threshold
        )
        
        self.input_encoder = _NodeEncoder()
        self.runner: GNNRunner = runner
    
    def get_points(self):
        """
        Retrieve points predicted by the GNN.

        Returns:
            np.ndarray: Nx3 array of predicted points.
        """
        nodes, _ = self.get_nodes()
        if nodes.shape[0] > 0:
            pred = self.runner.make_prediction(nodes)
            return pred[:, 0:3]
        else:
            return nodes[:, 0:3] # Return empty array with correct shape
