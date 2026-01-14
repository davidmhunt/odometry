import numpy as np

import matplotlib.pyplot as plt

from odometry.test_benches._test_bench import _TestBench
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.localization.icp2D_localization import icp2DLocalization

from geometries.pose.pose import Pose

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator

class PointCloudIntegratorTB(_TestBench):

    def __init__(
            self,
            gt_localizer:icp2DLocalization,
            map_handler:MapHandler,
            dataset:CpslDS,
            point_cloud_integrator:_PointCloudIntegrator,
            localizer = None,
            model_dataset_generator:_OnlineDatasetGenerator=None):
        
        super().__init__(gt_localizer, map_handler, dataset, localizer)

        self.point_cloud_integrator:_PointCloudIntegrator = point_cloud_integrator

        self.model_dataset_generator = model_dataset_generator

        self.normalize_frames = False
        self.generate_dataset = False
    
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

        if self.generate_dataset:

            nodes,labels = self.point_cloud_integrator.get_nodes(normalize_frames=self.normalize_frames)

            if nodes.shape[0]>0:
                self.model_dataset_generator.save_sample(
                    input_data=nodes,
                    output_data=labels
                )

        return self.point_cloud_integrator.get_points()
    
    def run(self, 
        max_frame=-1, 
        gt_enabled=True, 
        movie_generator = None,
        generate_dataset=False,
        normalize_frames=False):
        """Run the test bench

        Args:
            max_frame (int, optional): The frame to run the test bench up to.
                -1 indicates to run the entire dataset. Defaults to -1.
            gt_enabled (bool, optional): On True, additionally computes
                ground truth trajectories as well. Defaults to True.
            movie_generator (MovieGenerator, optional): When provided with a 
                MovieGenerator, additionally generates a movie. Defaults to None.
            generate_dataset (bool, optional): On True, generates a node dataset which
                can be used for training models
            normalize_frames (bool, optional): If True, normalizes the frames
                remaining by the total number of frames. Defaults to False.
                Only applies to point cloud integrators where the node stores 
                frame number. All other integrators ignore this flag.
        """

        #set the generate dataset flag
        self.generate_dataset = generate_dataset
        self.normalize_frames = normalize_frames

        return super().run(max_frame, gt_enabled, movie_generator)

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
            idx=idx+1,
            ax=axs[0,1],
            show=False
        )

        self.plotter_localization.plot_heading_history_deg(
            history_heading_deg=self.history_heading_deg,
            history_heading_deg_gt=self.history_heading_deg_gt,
            idx=idx+1,
            ax=axs[0,2],
            show=False
        )


        #middle (kalman filtering), point cloud, raw point cloud
        if len(self.history_filter_g) > 0:
            self.plotter_kalman.plot_chi_2_resp(
                g_thresh=self.filter.g_thresh[2],
                g_hist=np.array(self.history_filter_g),
                idx=idx,
                ax=axs[1,0],
                show=False
            )

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
                ax=axs[1,2],
                show=False
            )
            axs[1,2].set_title(
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
                ax=axs[2,2],
                show=False
            )
            axs[2,2].set_title(
                "GT Detectections: {}".format(detections.shape[0]),
                fontsize=self.plotter_localization.font_size_title
            )
        # self.plotter_localization.marker_size = 0.5

        #reset the marker size
        # self.plotter_localization.marker_size=10

        if show:

            plt.show()

        return axs