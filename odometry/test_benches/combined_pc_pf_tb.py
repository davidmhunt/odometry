import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from odometry.supportFns import rotation_functions
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.localization.particle_filter import particleFilter
from odometry.estimators.estimators import Inertial
from odometry.datasets.radnav_ds import radnavDS
from odometry.datasets.map_handler import MapHandler
from odometry.plotting.plotter_localization import PlotterLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.analyzers.analyzer import Analyzer
from odometry.point_cloud_processing.pc_stacker import pcStacker
from odometry.point_cloud_processing.multipath import MultiPath
from odometry.point_cloud_processing.vel_filtering import VelFiltering
from odometry.point_cloud_processing.ground_detection_filtering import groundDetectionFiltering
from odometry.plotting.movies import MovieGenerator

class combinedPCPFTb:

    def __init__(self,
                 localizer:particleFilter,
                 gt_localizer:icp2DLocalization,
                 map_handler:MapHandler,
                 dataset:radnavDS,
                 self_detection_radius_m=0.5,
                 vel_filter_enabled = True,
                 vel_filter_v_thresh = 1.0,
                 min_static_rejection_radius:float = 2.0,
                 dynamic_cluster_eps:float = 1.0,
                 dynamic_cluster_min_samples = 7) -> None:
        
        #initialize the localizer
        self.localizer:particleFilter = localizer
        self.gt_localizer:icp2DLocalization = gt_localizer
        
        #filter time tracking
        self.particle_filter_last_t:float = None
        self.particle_filter_next_vel_data:np.ndarray = None

        #load the datasets
        self.map_handler:MapHandler = map_handler
        self.dataset:radnavDS = dataset

        #initialize a plotter
        self.plotter_localization = PlotterLocalization(dataset,map_handler)

        #initialize an analyzer class
        self.analyzer = Analyzer()

        #localization histories
        self.history_position_m = None
        self.history_heading_deg = None
        self.history_position_m_gt = None
        self.history_heading_deg_gt = None
        self.history_localizers_reset()

        #point cloud processing
        self.point_cloud_stacker = pcStacker()
        self.multipath = MultiPath(
            clustering_eps = 1.0,
            clustering_min_samples= 12
        )
        self.vel_filtering_enabled = vel_filter_enabled
        self.vel_filtering = VelFiltering(
            v_thresh=vel_filter_v_thresh,
            min_static_rejection_radius=min_static_rejection_radius,
            dynamic_cluster_eps=dynamic_cluster_eps,
            dynamic_cluster_min_samples=dynamic_cluster_min_samples
        )
        self.ground_detection_filtering = groundDetectionFiltering(
            self_detection_radius_m
        )

        #combined point cloud processing history
        self.history_pc_stacker_point_clouds:list = None
        self.history_pc_stacker_position_m:list = None
        self.history_pc_stacker_heading_rad:list = None
        self.history_pc_stacker_reset()

        #motion model histories
        self.history_filter_est = None
        self.history_filter_p = None

        return
    
    ####################################################################
    #Initializing localizers and filters
    #################################################################### 
    
    def init_localization(self,
                          est_start_heading_rad,
                          est_start_pose_m,
                          n_particles = 1000,
                          hdg_range = (0,2 * np.pi),
                          show = False):
        
    
        #load the map points into the localizers
        self.gt_localizer.load_map_point_cloud(
            map_points=self.map_handler.map_points
        )

        #initialize the particle filter point clouds
        self.localizer.load_map_point_cloud(
            map_points=self.map_handler.map_points
        )
        
        self.localizer.load_map_free_points(
            map_free_points=self.map_handler.map_free_points
        )

        #initialize the gt localization
        init_gt_points = self.dataset.get_lidar_point_cloud(idx=0)

        new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
            points=init_gt_points,
            estimated_heading_rad=est_start_heading_rad,
            estimated_pose_m=est_start_pose_m
        )

        print("gt icp estimated heading:{} deg, pose:{}".format(
            np.rad2deg(new_heading_rad),new_pose_m))
        
        self.gt_localizer.reset_odometry(
            pose=new_pose_m,
            heading_rad=new_heading_rad
        )

        #initialize the particle filter
        self.localizer.initialize_from_uniform_dist(
            hdg_range=hdg_range,
            N=n_particles
        )

        #get the first points in the localizer point cloud
        # init_points = self.dataset.get_radar_detections(idx=0)
        # init_points = init_points[:,:2]
        
        #reset particle filter localization
        self.localizer.odometry_reset()
        
        #TODO: Add this back if needed
        # new_heading_rad,new_pose_m = self.localizer.update_odometry(
        #     points=init_points,
        #     estimated_heading_rad=est_start_heading_rad,
        #     estimated_pose_m=est_start_pose_m
        # )
        
        #TODO: add in reset_odometry to particle filter
        #to reset the particle filter odometry
        # self.localizer.reset_odometry(
        #     pose=new_pose_m,
        #     heading_rad=new_heading_rad
        # )


        if show:
            self.plotter_localization.plot_particles_on_map(
                particles=self.localizer.particles,
                show=True
            )

    def init_motion_model(self,
                    start_time_s:float):
        """Initialize the particle filter's motion
        model

        Args:
            start_time_s (float): time of first data sample
                in seconds
        """
        self.localizer.motion_model_reset(
            t0=start_time_s
        )

        #reset filter histories
        self.history_motion_model_reset()

        #reset filter time
        self.particle_filter_last_t = \
            self.dataset.get_imu_full_data(idx=0)[0,0]
            
    ####################################################################
    #Histories (localizers)
    #################################################################### 

    def history_localizers_reset(self):
        """reest the history tracking of the ground truth
        localizer and the particle filter
        """
        n = self.dataset.num_frames

        #reset the pose histories
        self.history_position_m = np.zeros(shape=(n,2),dtype=np.double)
        self.history_position_m_gt = np.zeros(shape=(n,2),dtype=np.double)

        #reset the orientation histories
        self.history_heading_deg = [None] * n
        self.history_heading_deg_gt = [None] * n
    
    def history_update_pose(self,
                                position_m:np.ndarray,
                                heading_rad:np.ndarray,
                                idx:int):
        """Update the pose history for the localizer (particle filter)

        Args:
            position_m (np.ndarray): the position from the localizer
            heading_rad (np.ndarray): the heading from the localizer
            idx (int): the index of the sample from the dataset
        """
        #TODO: modify this to work depending on the 
        #particle filter
        self.history_position_m[idx] = position_m
        self.history_heading_deg[idx] = np.rad2deg(heading_rad)
    
    def history_update_pose_gt(self,
                                position_m:np.ndarray,
                                heading_rad:np.ndarray,
                                idx:int):
        """Update the pose history for the ground truth localizer

        Args:
            position_m (np.ndarray): the position from the localizer
            heading_rad (np.ndarray): the heading from the localizer
            idx (int): the index of the sample from the dataset
        """
        self.history_position_m_gt[idx] = position_m
        self.history_heading_deg_gt[idx] = np.rad2deg(heading_rad)
    
    ####################################################################
    #Histories (motion models)
    ####################################################################
    def history_motion_model_reset(self):

        self.history_filter_est = [self.localizer.motion_model.x.copy()]
        # self.history_filter_p = [np.sqrt(np.diag(self.localizer.motion_model.P))]
    
    def history_motion_model_update_history(self):
        """Update the history tracking of the particle
        filter's motion model state
        """
        self.history_filter_est.append(self.localizer.motion_model.x.copy())
        # self.history_filter_p.append(np.sqrt(np.diag(self.localizer.motion_model.P)))

    
    ####################################################################
    #Histories (point cloud stacking)
    ####################################################################
    def history_pc_stacker_reset(self):
        """Reset the history tracking of the "point cloud stacker"
        which is used to generate more detailed point clouds
        """
        self.history_pc_stacker_point_clouds = []
        self.history_pc_stacker_position_m = []
        self.history_pc_stacker_heading_rad = []

        return

    def history_pc_stacker_update(
            self,
            valid_stacked_point_cloud:np.ndarray,
            position_m:np.ndarray,
            heading_rad:float):
        """Save the most recently computed stacked point cloud and its
        estimated position and heading

        Args:
            valid_stacked_point_cloud (np.ndarray): Nx2 array of valid points
                corresponding to the most recent stacked point cloud after
                ray tracing/multi-path rejection
            position_m (np.ndarray): Nx2 array corresponding to the
                position from the most recent icp estimate
            heading_rad (float): heading corresponding to the icp position
                estimate using the most recent stacked point cloud
        """
        
        self.history_pc_stacker_point_clouds.append(valid_stacked_point_cloud)
        self.history_pc_stacker_position_m.append(position_m)
        self.history_pc_stacker_heading_rad.append(heading_rad)

    ####################################################################
    #Handling time
    #################################################################### 
    def get_dataset_start_time(self,idx=0)->float:
        """Get the start time for the earliest imu or vehicle vel measurement
        for a dataset frame

        Args:
            idx (int, optional): the frame index. Defaults to 0.

        Returns:
            float: the start time in seconds
        """

        assert (self.dataset.imu_full_enabled or 
                self.dataset.vehicle_vel_enabled),\
                "cannot get time as IMU_full nor \
                vehicle_vel datasets not found"
        
        start_time = 0.0

        if self.dataset.imu_full_enabled:
            data = self.dataset.get_imu_full_data(idx=idx)
            start_time = max(data[0,0],start_time)
        
        if self.dataset.vehicle_vel_enabled:
            data = self.dataset.get_vehicle_vel_data(idx=idx)
            start_time = max(data[0,0],start_time)
        
        return start_time

    ####################################################################
    #Filter Predictions and Updates
    ####################################################################
    def motion_model_predict_from_frame_samples(self, idx = 0):
        """Predict forward the motion model from the IMU's
        gyroscope reading (in rad/sec) and the robot's
        velocity tracker

        Args:
            idx (int, optional): The frame index of data
                to use to predict the state forward. Defaults to 0.
        """
        imu_data = self.dataset.get_imu_full_data(idx)
        vel_data = self.dataset.get_vehicle_vel_data(idx)

        #make sure that the datasets have the same number of samples
        if imu_data.shape[0] != vel_data.shape[0]:
            print("frame {} is bad, (imu: {},data: {})".format(
                idx,imu_data.shape[0],vel_data.shape[0]))
            
            return #skip the frame as it must be bad

        for i in range(imu_data.shape[0]):
            
            #make sure that the time values line up
            assert np.isclose(imu_data[i,0],vel_data[i,0])

            #update time parameters
            dt = imu_data[i,0] - self.particle_filter_last_t

            #create inertial
            inertial = Inertial(
                gyro=imu_data[i,3],
                sencode=vel_data[i,1]
            )

            #predict the filter forward
            self.localizer.motion_model_predict(
                dt=dt,
                inertial=inertial
            )

            #save last time
            self.particle_filter_last_t = imu_data[i,0]

            #update histories
            self.history_motion_model_update_history()

        return
    
    def reset_motion_model(self,
                    start_time_s:float):
        """Reset the motion model of the particle filter

        Args:
            start_time_s (float): the time in seconds
                for the reset motion model to use as its 
                new start time
        """
        #reset the motion model with the most recent t
        self.localizer.motion_model_reset(
            t0=self.particle_filter_last_t
        )
        

    ####################################################################
    #Running localization for the dataset
    ####################################################################

    def run(
            self,
            max_frame=-1,
            gt_enabled=True,
            movie_generator:MovieGenerator = None):
        if max_frame == -1:
            max_frame = self.dataset.num_frames

        #TODO: David reset the localizer's motion model

        #reset the point cloud stacker to be at 0,0
        self.point_cloud_stacker.reset(
            initial_heading_rad=self.localizer.motion_model.x[2],
            initial_pose_m=np.array([
                self.localizer.motion_model.x[0],
                self.localizer.motion_model.x[1]
            ]),
            initial_time_s=self.particle_filter_last_t
        )

        for i in tqdm(range(max_frame)):
            if gt_enabled:
                # update the lidar ground truth
                gt_points = self.dataset.get_lidar_point_cloud(idx=i)

                new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
                    points=gt_points
                )

                self.history_update_pose_gt(
                    position_m=new_pose_m,
                    heading_rad=new_heading_rad,
                    idx = i
                )
            
            #predict the states forward
            self.motion_model_predict_from_frame_samples(idx=i)

            #generate combined point cloud
            radar_points = self.dataset.get_radar_detections(idx=i)

            #filter dynamic objects
            if self.vel_filtering_enabled:
                static_points = self.vel_filtering.get_static_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.localizer.motion_model.x[3],0.0])
                )

                dynamic_points = self.vel_filtering.get_dynamic_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.localizer.motion_model.x[3],0.0])
                )

                radar_points = self.vel_filtering.remove_dynamic_clusters_from_static_detections(
                    static_detections=static_points,
                    dynamic_detections=dynamic_points
                )

            #filter out ground detections, etc
            radar_points = \
                self.ground_detection_filtering.remove_sensor_self_detections(
                    radar_points[:,:2])

            self.point_cloud_stacker.add_points(
                current_points=radar_points,
                heading_rad=self.localizer.motion_model.x[2],
                pose_m= \
                    np.array([
                        self.localizer.motion_model.x[0],
                        self.localizer.motion_model.x[1]
                    ]),
                current_time_s=self.particle_filter_last_t
            )

            #check to see if the vehicle has moved a sufficient amount for using
            #a combined point cloud and updating the particle filter
            if (self.point_cloud_stacker.get_rel_distance_m() > 0.5) or \
                (self.point_cloud_stacker.get_rel_heading_deg() > 90) or \
                (self.point_cloud_stacker.get_elapsed_time() > 10):
                
                #get the stacked point cloud
                pc = self.point_cloud_stacker.get_points()

                #remove multipath detections
                pc = self.multipath.remove_multipath(pc)
            
                self.localizer.odometry_update_from_measurement(
                    measured_point_cloud=pc
                )

                if self.localizer.current_odom_valid:

                    #save the measurement and point cloud
                    self.history_pc_stacker_update(
                        valid_stacked_point_cloud=pc,
                        position_m=self.localizer.current_pose_m,
                        heading_rad=self.localizer.current_heading_rad
                    )

                    #TODO: update histories with valid odom measurement
                
                #reset the motion model
                self.localizer.motion_model_reset(self.particle_filter_last_t)

                # reset the point cloud stacker
                self.point_cloud_stacker.reset(
                    initial_heading_rad=self.localizer.motion_model.x[2],
                    initial_pose_m=np.array(
                        [self.localizer.motion_model.x[0],
                         self.localizer.motion_model.x[1]]),
                    initial_time_s=self.particle_filter_last_t
                )
            elif self.point_cloud_stacker.get_elapsed_time() > 10:

                #if the vehicle hasn't moved significantly over the last 10 seconds,
                #go ahead and reset the point cloud stacker to prevent 
                #accumulation of false points
                # reset the point cloud stacker
                self.point_cloud_stacker.reset(
                    initial_heading_rad=self.localizer.motion_model.x[2],
                    initial_pose_m=np.array(
                        [self.localizer.motion_model.x[0],
                         self.localizer.motion_model.x[1]]),
                    initial_time_s=self.particle_filter_last_t
                )

                #don't reset the particle filter's motion model though

            
            #compute the latest updated pose
            #TODO: edit behavior for when the pf doesn't have
                #accurate odometry yet
            
            pose_m = self.localizer.current_pose_m + \
                self.localizer.motion_model.x[0:2]
            
            heading_rad = self.localizer.current_heading_rad + \
                self.localizer.motion_model.x[2]
            self.history_update_pose(
                position_m=pose_m,
                heading_rad=heading_rad,
                idx=i
            )

            # if movie_generator:
            #     #plot the current state
            #     self.plot_compilation(
            #         idx = i,
            #         axs = movie_generator.axs,
            #         show=False
            #     )

                # movie_generator.save_frame(clear_axs=True)
        return
    
    ####################################################################
    #Performing Analysis
    ####################################################################
    def analyze(self):

        self.analyzer.show_summary_statistics(
            self.history_position_m,
            self.history_position_m_gt,
            self.history_heading_deg,
            self.history_heading_deg_gt
        )
    
    ####################################################################
    #Plot compilation of data
    ####################################################################
    def plot_compilation(
            self,
            idx=-1,
            axs:plt.Axes=[],
            show=False
        ):

        if len(axs) == 0:
            fig,axs=plt.subplots(2,3, figsize=(15,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)

        #top row pose(localization and heading) and camera view
        self.plotter_localization.plot_heading_history_deg(
            history_heading_deg=self.history_heading_deg,
            history_heading_deg_gt=self.history_heading_deg_gt,
            idx=idx+1,
            ax=axs[0,0],
            show=False
        )
        
        self.plotter_localization.plot_position_history_m(
            history_position_m=self.history_position_m,
            history_position_m_gt=self.history_position_m_gt,
            idx=idx+1,
            ax=axs[0,1],
            show=False
        )

        if self.dataset.camera_enabled:

            axs[0,2].imshow(
                self.dataset.get_camera_frame(idx)
            )
            axs[0,2].set_title("Camera View")

        #bottom row (combined point cloud) and kalman filtering
        self.plotter_localization.plot_particles_on_map(
            particles=self.localizer.particles,
            pose_m=self.localizer.current_pose_m,
            ax=axs[1,0],
            show=False
        )

        self.plotter_localization.marker_size = 0.5
        if len(self.history_pc_stacker_point_clouds) > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=self.history_pc_stacker_point_clouds[-1],
                heading_rad=self.history_pc_stacker_heading_rad[-1],
                pose_m=self.history_pc_stacker_position_m[-1],
                ax=axs[1,1],
                show=False
            )
            axs[1,1].set_title("Last Raytraced Point Cloud",
                               fontsize=self.plotter_localization.font_size_title)
            
        combined_pc = self.point_cloud_stacker.get_point_from_initial_pose()
        if combined_pc.shape[0] > 0:

            self.plotter_localization.plot_detections_on_map(
                current_points=combined_pc,
                heading_rad=self.point_cloud_stacker.initial_heading_rad,
                pose_m=self.point_cloud_stacker.initial_pose_m,
                ax=axs[1,2],
                show=False
            ) 
            axs[1,2].set_title("Current Stacked Point Cloud: {}".format(len(combined_pc)),
                               fontsize=self.plotter_localization.font_size_title)
        
        #reset the marker size
        self.plotter_localization.marker_size=10

        if show:
            plt.show()