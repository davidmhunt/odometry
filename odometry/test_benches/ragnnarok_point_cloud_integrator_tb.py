import numpy as np

import matplotlib.pyplot as plt

from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB
from odometry.test_benches._test_bench import PredictionSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.localization.icp2D_localization import icp2DLocalization

from geometries.pose.pose import Pose

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator

class RaGNNPointCloudIntegratorTB(PointCloudIntegratorTB):
    """
    Test bench for RaGNN-based point cloud integration.

    This class extends the `PointCloudIntegratorTB` to include specific
    visualizations for probabilistic and ground truth occupancy grids.
    """

    def __init__(
            self,
            gt_localizer:icp2DLocalization,
            map_handler:MapHandler,
            dataset:CpslDS,
            point_cloud_integrator:_PointCloudIntegrator,
            dynamic_point_cloud_integrator: _PointCloudIntegrator = None,
            localizer = None,
            model_dataset_generator:_OnlineDatasetGenerator=None,
            use_filters:bool = True,
            prediction_source:PredictionSource = PredictionSource.IMU_AND_VEL,
            gt_source=None,
            odom_frame:OdomCoordinateFrame = OdomCoordinateFrame.FLU
            ):
        """
        Initializes the RaGNNPointCloudIntegratorTB.

        Args:
            gt_localizer (icp2DLocalization): Ground truth localizer.
            map_handler (MapHandler): Handles map and static environment data.
            dataset (CpslDS): The dataset containing the sensor logs.
            point_cloud_integrator (_PointCloudIntegrator): Integrator for processing the radar point cloud.
            dynamic_point_cloud_integrator (_PointCloudIntegrator, optional): Integrator for dynamic points. Defaults to None.
            localizer (optional): The localizer module to be evaluated. Defaults to None.
            model_dataset_generator (_OnlineDatasetGenerator, optional): Generator for online sample extraction. Defaults to None.
            use_filters (bool, optional): If True, uses filtering during processing. Defaults to True.
            prediction_source (PredictionSource, optional): Source for pose predictions. Defaults to PredictionSource.IMU_AND_VEL.
            gt_source (optional): Source configuration for ground truth data. Defaults to None.
            odom_frame (OdomCoordinateFrame, optional): Odometry coordinate frame specification. Defaults to OdomCoordinateFrame.FLU.
        """
        
        super().__init__(
            gt_localizer=gt_localizer,
            map_handler=map_handler,
            dataset=dataset,
            point_cloud_integrator=point_cloud_integrator,
            dynamic_point_cloud_integrator=dynamic_point_cloud_integrator,
            localizer=localizer,
            model_dataset_generator=model_dataset_generator,
            use_filters=use_filters,
            prediction_source=prediction_source,
            gt_source=gt_source,
            odom_frame=odom_frame
        )

        return

    def plot_compilation(
            self,
            idx=-1,
            axs:plt.Axes=[],
            show=False
        ):
        """
        Plot a compilation of the current test bench state.

        This method adds RaGNN-specific plots (occupancy grids) to the base
        compilation.

        Args:
            idx (int, optional): The frame index to plot. Defaults to -1.
            axs (plt.Axes, optional): Pre-existing axes to plot on. If empty,
                a 4x3 grid is created. Defaults to [].
            show (bool, optional): If True, calls plt.show(). Defaults to False.

        Returns:
            plt.Axes: The axes containing the plots.
        """

        if len(axs) == 0:
            fig, axs = plt.subplots(4, 3, figsize=(15, 20)) #(W,H)
            fig.subplots_adjust(wspace=0.3, hspace=0.30)

        axs = super().plot_compilation(idx, axs, show=False)

        #bottom: occupancy grids
        grid = self.point_cloud_integrator.probabilistic_pc_grid.grid
        bins = self.point_cloud_integrator.probabilistic_pc_grid.grid_bins
        valid_pts = self.point_cloud_integrator.probabilistic_pc_grid.get_points()
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            # valid_points=valid_pts,
            ax=axs[3,0],
            show=False
        )

        #plotting the ground truth occupancy grid
        grid = self.point_cloud_integrator.probabilistic_pc_grid.gt_grid
        bins = self.point_cloud_integrator.probabilistic_pc_grid.grid_bins
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            ax=axs[3,1],
            show=False
        )
        axs[3,1].set_title(
            "GT Occupancy Grid",
            fontsize=self.plotter_pc_grid.font_size_title
        )

        if show:

            plt.show()

        return axs
