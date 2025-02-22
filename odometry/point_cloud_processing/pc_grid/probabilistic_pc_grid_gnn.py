import numpy as np

from odometry.point_cloud_processing.pc_grid.probabilistic_pc_grid import ProbabilisticPCGrid

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner

class ProbabilisticPCGridGNN(ProbabilisticPCGrid):
    """Probabilistic point cloud grid, but using a GNN to get points instead of a threshold

    Args:
        ProbabilisticPCGrid (_type_): _description_
    """
    def __init__(
            self,
            runner:GNNRunner,
            grid_resolution_m=0.05,
            grid_max_distance_m=3,
            num_frames_history = 10,
            occupancy_threshold = 0.5):
        
        super().__init__(
            grid_resolution_m,
            grid_max_distance_m,
            num_frames_history,
            occupancy_threshold)
        
        self.input_encoder = _NodeEncoder()
        self.runner:GNNRunner = runner
    
    def get_points(self):

        nodes,_ = self.get_nodes()
        pred = self.runner.make_prediction(nodes)
        return pred[:,0:3]