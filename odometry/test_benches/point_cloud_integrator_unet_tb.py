import numpy as np

import matplotlib.pyplot as plt

from odometry.test_benches._test_bench import _TestBench, PredictionSource
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.localization.icp2D_localization import icp2DLocalization

from geometries.pose.pose import Pose

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from odometry.test_benches._test_bench import OdomCoordinateFrame
from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB

class PointCloudIntegratorUnetTB(PointCloudIntegratorTB):

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
        
        super().__init__(
            gt_localizer,
            map_handler,
            dataset,
            point_cloud_integrator,
            dynamic_point_cloud_integrator,
            localizer,
            model_dataset_generator,
            use_filters=use_filters,
            prediction_source=prediction_source,
            gt_source=gt_source,
            odom_frame=odom_frame
        )

        return
    
    def process_point_cloud(
            self,
            point_cloud_raw:np.ndarray,
            static_points:np.ndarray,
            dynamic_points:np.ndarray,
            current_pose:Pose,
            gt_points:np.ndarray=np.empty(shape=(0,3))) -> np.ndarray:
        """Implemented by the child class to process the point cloud

        Args:
            point_cloud_raw (np.ndarray): nx4 array for point cloud in agent frame
                corresponding to [x,y,z,vel] full radar point cloud
            static_points (np.ndarray): nx4 array for point cloud in agent frame
                corresponding to [x,y,z,vel] point cloud of static points
            dynamic_points (np.ndarray): nx4 array for point cloud in agent frame
                corresponding to [x,y,z,vel] point cloud of dynamic points
            current_pose (Pose): pose object corresponding to the currently 
                estimated position (from local odometry)
            gt_points (np.ndarray,optional): [x,y,z] point cloud corresponding to 
                the ground truth detections. Can be used for evaluation of generated
                point cloud
                Defaults to np.empty(shape=(0,2)).

        Returns:
            np.ndarray: [x,y,z] point cloud to be used for down stream localization
                tasks. Returns empty array if no points available or if no point 
                cloud ready to be used
        """
        
        self.point_cloud_integrator.add_points(
            static_points=static_points,
            current_pose=current_pose,
            gt_points=gt_points
        )

        if self.dynamic_point_cloud_integrator is not None:
            self.dynamic_point_cloud_integrator.add_points(
                static_points=dynamic_points,
                current_pose=current_pose
            )

        if self.generate_dataset:
            
            if self.point_cloud_integrator.check_valid_num_frames():

                grid = self.point_cloud_integrator.get_grid(density=True)
                gt_grid = self.point_cloud_integrator.get_gt_grid()
            
                self.model_dataset_generator.save_sample(
                    input_data=grid,
                    output_data=gt_grid
                )

        return self.point_cloud_integrator.get_points()

    def plot_compilation(
            self,
            idx=-1,
            axs:plt.Axes=[],
            show=False
        ):

        if len(axs) == 0:
            fig,axs=plt.subplots(3,3, figsize=(15,15)) #(W,H)
            fig.subplots_adjust(wspace=0.3,hspace=0.30)

        #top row pose(localization and heading) and camera view
        if self.dataset.camera_enabled:

            axs[0,0].imshow(
                self.dataset.get_camera_frame(idx)
            )
            axs[0,0].set_title("Camera View")

        self.plotter_localization.plot_position_history_m(
            history_position_m=self.history_position_m,
            history_position_m_gt=self.history_position_m_gt,
            history_position_m_inertial=self.history_position_m_inertial,
            idx=idx+1,
            ax=axs[0,1],
            show=False
        )

        self.plotter_localization.plot_heading_history_deg(
            history_heading_deg=self.history_heading_deg,
            history_heading_deg_gt=self.history_heading_deg_gt,
            history_heading_deg_inertial=self.history_heading_deg_inertial,
            idx=idx+1,
            ax=axs[0,2],
            show=False
        )


        #middle (kalman filtering), point cloud, raw point cloud
        
        # if len(self.history_filter_g) > 0:
        #     self.plotter_kalman.plot_chi_2_resp(
        #         g_thresh=self.filter.g_thresh[2],
        #         g_hist=np.array(self.history_filter_g),
        #         idx=idx,
        #         ax=axs[1,0],
        #         show=False
        #     )

        #raw accumulated point cloud
        self.plotter_localization.marker_size = 0.5
        self.plotter_localization.plot_x_max = 7.5
        self.plotter_localization.plot_y_max = 7.5
        accumulated_points = self.point_cloud_integrator.get_raw_point_history()
        if accumulated_points.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=accumulated_points[:,0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[1,0],
                show=False
            )
            axs[1,0].set_title(
                "Raw Detectections: {}".format(accumulated_points.shape[0]),
                fontsize=self.plotter_localization.font_size_title
            )
        
        #filtered point cloud
        detections = self.point_cloud_integrator.get_points()
        if detections.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=detections[:,0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[1,1],
                show=False
            )

        detections = self.point_cloud_integrator.get_gt_points()
        if detections.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=detections[:,0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[1,2],
                show=False
            )
            axs[1,2].set_title(
                "GT Detectections: {}".format(detections.shape[0]),
                fontsize=self.plotter_localization.font_size_title
            )
        
        #dynamic accumulated point cloud
        # if self.dynamic_point_cloud_integrator is not None:
        #     dynamic_points = self.dynamic_point_cloud_integrator.get_points()
        #     if dynamic_points.shape[0] > 0:
        #         self.plotter_localization.plot_detections_on_map(
        #             current_points=dynamic_points[:,0:2],
        #             heading_rad=np.deg2rad(self.history_heading_deg[idx]),
        #             pose_m=self.history_position_m[idx],
        #         ax=axs[2,1],
        #         show=False
        #     )
        #     axs[2,1].set_title(
        #         "Dynamic Detectections: {}".format(dynamic_points.shape[0]),
        #         fontsize=self.plotter_localization.font_size_title
        #     )
        

        #bottom: grids
        grid = self.point_cloud_integrator.get_grid(
            density=True
        )
        bins = self.point_cloud_integrator.raw_point_history.grid_bins
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            ax=axs[2,0],
            show=False
        )
        axs[2,0].set_title(
            "Density Grid",
            fontsize=self.plotter_pc_grid.font_size_title
        )

        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            ax=axs[2,1],
            valid_points=self.point_cloud_integrator.get_gt_points(),
            show=False
        )
        axs[2,1].set_title(
            "Density Grid (labeled)",
            fontsize=self.plotter_pc_grid.font_size_title
        )

        #plotting the ground truth occupancy grid
        grid = self.point_cloud_integrator.get_gt_grid(
            density=False
        )
        bins = self.point_cloud_integrator.raw_point_history.grid_bins
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid,
            grid_bins=bins,
            ax=axs[2,2],
            show=False
        )
        axs[2,2].set_title(
            "GT Grid Label",
            fontsize=self.plotter_pc_grid.font_size_title
        )

        if show:

            plt.show()

        return axs
