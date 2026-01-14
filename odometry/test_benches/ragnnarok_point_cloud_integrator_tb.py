import numpy as np

import matplotlib.pyplot as plt

from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.localization.icp2D_localization import icp2DLocalization

from geometries.pose.pose import Pose

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator

class RaGNNPointCloudIntegratorTB(PointCloudIntegratorTB):

    def __init__(
            self,
            gt_localizer:icp2DLocalization,
            map_handler:MapHandler,
            dataset:CpslDS,
            point_cloud_integrator:_PointCloudIntegrator,
            localizer = None,
            model_dataset_generator:_OnlineDatasetGenerator=None):
        
        super().__init__(gt_localizer, map_handler, dataset, point_cloud_integrator,localizer, model_dataset_generator)

        return

    def plot_compilation(
            self,
            idx=-1,
            axs:plt.Axes=[],
            show=False
        ):

        axs = super().plot_compilation(idx, axs, show=False)

        #bottom: occupancy grids
        grid = self.point_cloud_integrator.probabilistic_pc_grid.grid
        bins = self.point_cloud_integrator.probabilistic_pc_grid.grid_bins
        valid_pts = self.point_cloud_integrator.probabilistic_pc_grid.get_points()
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            # valid_points=valid_pts,
            ax=axs[2,0],
            show=False
        )

        #plotting the ground truth occupancy grid
        grid = self.point_cloud_integrator.probabilistic_pc_grid.gt_grid
        bins = self.point_cloud_integrator.probabilistic_pc_grid.grid_bins
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            ax=axs[2,1],
            show=False
        )
        axs[2,1].set_title(
            "GT Occupancy Grid",
            fontsize=self.plotter_pc_grid.font_size_title
        )

        if show:

            plt.show()

        return axs
