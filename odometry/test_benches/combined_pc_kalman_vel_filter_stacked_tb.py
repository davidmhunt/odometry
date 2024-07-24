import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from odometry.supportFns import rotation_functions
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.datasets.radnav_ds import radnavDS
from odometry.datasets.map_handler import MapHandler
from odometry.plotting.plotter_localization import PlotterLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.analyzers.analyzer import Analyzer
from odometry.estimators.estimators import (
    _ExtendedKalmanFilter,
    KalmanXYPhiSpeedGyroEncoder,
    InertialIntegrator,
    Inertial)
from odometry.point_cloud_processing.pc_stacker import pcStacker
from odometry.point_cloud_processing.multipath import MultiPath
from odometry.point_cloud_processing.vel_filtering import VelFiltering
from odometry.plotting.movies import MovieGenerator

class combinedPCVelFilteringStackedTB:

    def __init__(self,
                 localizer:icp2DLocalization,
                 gt_localizer:icp2DLocalization,
                 map_handler:MapHandler,
                 dataset:radnavDS,
                 vel_filter_enabled = True,
                 vel_filter_v_thresh = 1.0,
                 min_static_rejection_radius:float = 2.0,
                 dynamic_cluster_eps:float = 1.0,
                 dynamic_cluster_min_samples = 7) -> None:
        
        #initialize the localizer
        self.localizer:icp2DLocalization = localizer
        self.gt_localizer:icp2DLocalization = gt_localizer

        #filter parameters
        self.filter:KalmanXYPhiSpeedGyroEncoder = None
        self.filter_R:np.ndarray = None
        self.filter_msmt_component:list = None
        self.filter_integrator = InertialIntegrator()
        
        #filter time tracking
        self.filter_last_t:float = None
        self.filter_next_vel_data:np.ndarray = None

        #load the datasets
        self.map_handler:MapHandler = map_handler
        self.dataset:radnavDS = dataset

        #initialize a plotter
        self.plotter_localization = PlotterLocalization(dataset,map_handler)
        self.plotter_kalman = PlotterKalman()
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
        self.dynamic_point_cloud_stacker = pcStacker()
        self.multipath = MultiPath(
            clustering_eps = 0.5,
            clustering_min_samples= 15
        )
        self.vel_filtering_enabled = vel_filter_enabled
        self.vel_filtering = VelFiltering(
            v_thresh=vel_filter_v_thresh,
            min_static_rejection_radius=min_static_rejection_radius,
            dynamic_cluster_eps=dynamic_cluster_eps,
            dynamic_cluster_min_samples=dynamic_cluster_min_samples
        )

        #combined point cloud processing history
        self.history_pc_stacker_point_clouds:list = None
        self.history_pc_stacker_position_m:list = None
        self.history_pc_stacker_heading_rad:list = None
        self.history_pc_stacker_reset()

        #kalman filter histories
        self.history_filter_est = None
        self.history_filter_g = None
        self.history_filter_y = None
        self.history_filter_p = None

        return
    
    ####################################################################
    #Initializing localizers and filters
    #################################################################### 
    
    def init_localization(self,
                          est_start_heading_rad,
                          est_start_pose_m,
                          show = False):
        
    
        #load the map points into the localizers
        self.gt_localizer.load_map_point_cloud(
            map_points=self.map_handler.map_points
        )
        self.localizer.load_map_point_cloud(
            map_points=self.map_handler.map_points
        )

        #get the first points in the gt point cloud
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

        #get the first points in the localizer point cloud
        init_points = self.dataset.get_radar_detections(idx=0)
        init_points = init_points[:,:2]

        new_heading_rad,new_pose_m = self.localizer.update_odometry(
            points=init_points,
            estimated_heading_rad=est_start_heading_rad,
            estimated_pose_m=est_start_pose_m
        )

        if new_heading_rad is None:
            new_heading_rad = est_start_heading_rad
            new_pose_m = est_start_pose_m
            print("radar icp failed to find initial location, using est start pose")

        print("radar icp estimated heading:{} deg, pose:{}".format(
            np.rad2deg(new_heading_rad),new_pose_m))
        
        self.localizer.reset_odometry(
            pose=new_pose_m,
            heading_rad=new_heading_rad
        )


        if show:
            self.plotter_localization.plot_detections_on_map(
                current_points=init_points,
                heading_rad=new_heading_rad,
                pose_m=new_pose_m,
                show=show
            )
        
        return new_heading_rad,new_pose_m
    
    def init_filter(self,
                    est_start_heading_rad:float,
                    est_start_position_m:np.ndarray,
                    start_time_s:float):
        
        #declare initial state [x,y,phi,speed,gyro bias, encoder bias]
        x0 = np.array([
            est_start_position_m[0],
            est_start_position_m[1],
            est_start_heading_rad,
            0,
            0,
            0
        ])

        #declare initial state covariance matrix
        P0 = np.diag([5,5,0.1,1,1e-2,1e-2])

        self.filter = KalmanXYPhiSpeedGyroEncoder(
            t0=start_time_s,
            x0 = x0,
            P0 = P0,
            chi2_pct=0.95, #originally 0.95
            do_chi2=True
        )

        #define the observation noise
        # self.filter_R = np.diag([0.02,0.02,0.005]) ** 2 #original values
        self.filter_R = np.diag([0.1,0.1,0.05]) ** 2
        self.filter_msmt_component = ["x","y","phi"]

        #reset filter histories
        self.history_filters_reset()

        #reset filter time
        self.filter_last_t = \
            self.dataset.get_imu_full_data(idx=0)[0,0]
        
        #initialize the integrator
        self.filter_integrator = InertialIntegrator()
            
    ####################################################################
    #Histories (localizers)
    #################################################################### 

    def history_localizers_reset(self):

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
        """Update the pose history for the localizer

        Args:
            position_m (np.ndarray): the position from the localizer
            heading_rad (np.ndarray): the heading from the localizer
            idx (int): the index of the sample from the dataset
        """
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
    #Histories (filtering)
    ####################################################################
    def history_filters_reset(self):

        self.history_filter_est = [self.filter.x.copy()]
        self.history_filter_g = [0]
        self.history_filter_y = [np.array([0,0])]
        self.history_filter_p = [np.sqrt(np.diag(self.filter.P))]
    
    def history_filters_update_state_history(self):

        self.history_filter_est.append(self.filter.x.copy())
        self.history_filter_p.append(np.sqrt(np.diag(self.filter.P)))
    
    def history_filters_update_from_msmt(self):

        self.history_filter_g.append(self.filter.g)
        self.history_filter_y.append(self.filter.y)
    
    ####################################################################
    #Histories (point cloud stacking)
    ####################################################################
    def history_pc_stacker_reset(self):

        self.history_pc_stacker_point_clouds = []
        self.history_pc_stacker_position_m = []
        self.history_pc_stacker_heading_rad = []

        return

    def history_pc_stacker_update(
            self,
            valid_stacked_point_cloud:np.ndarray,
            icp_position_m:np.ndarray,
            icp_heading_rad:float):
        """Save the most recently computed stacked point cloud and its
        estimated position and heading

        Args:
            valid_stacked_point_cloud (np.ndarray): Nx2 array of valid points
                corresponding to the most recent stacked point cloud after
                ray tracing/multi-path rejection
            icp_position_m (np.ndarray): Nx2 array corresponding to the
                position from the most recent icp estimate
            icp_heading_rad (float): heading corresponding to the icp position
                estimate using the most recent stacked point cloud
        """
        
        self.history_pc_stacker_point_clouds.append(valid_stacked_point_cloud)
        self.history_pc_stacker_position_m.append(icp_position_m)
        self.history_pc_stacker_heading_rad.append(icp_heading_rad)

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
    def filter_predict_from_frame_samples(self, idx = 0):

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
            dt = imu_data[i,0] - self.filter_last_t

            #create inertial
            inertial = Inertial(
                gyro=imu_data[i,3],
                sencode=vel_data[i,1]
            )

            #predict the filter forward
            self.filter.predict(
                dt=dt,
                inertial=inertial,
                check_P=False
            )

            #save last time
            self.filter_last_t = imu_data[i,0]

            #update histories
            self.history_filters_update_state_history()

        return
    
    def filter_perform_update(self,
                              estimated_position_m:np.ndarray,
                              estimated_heading_rad:np.ndarray,
                              t:float):
        
        z = np.array([
            estimated_position_m[0],
            estimated_position_m[1],
            estimated_heading_rad
        ])

        self.filter.update(
            t=t,
            z=z,
            R=self.filter_R,
            msmt_components=self.filter_msmt_component
        )

        #save the histories
        self.history_filters_update_state_history()
        self.history_filters_update_from_msmt()
    
    def velreset(self):
        if self.vel_filtering_enabled:
            self.dynamic_point_cloud_stacker.reset(
                initial_heading_rad=self.filter.x[2],
                initial_pose_m=np.array([
                    self.filter.x[0],
                    self.filter.x[1]
                ]),
                initial_time_s=self.filter_last_t
            )
     
        self.point_cloud_stacker.reset(
            initial_heading_rad=self.filter.x[2],
            initial_pose_m=np.array([
                self.filter.x[0],
                self.filter.x[1]
            ]),
            initial_time_s=self.filter_last_t
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

        #TODO: improve to resent point cloud stacker
        self.velreset()

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

            #perform localization with the EKF

            #predict the states forward
            self.filter_predict_from_frame_samples(idx=i)

            #generate combined point cloud
            radar_points = self.dataset.get_radar_detections(idx=i)


            #filter dynamic objects
            if self.vel_filtering_enabled:
                static_points = self.vel_filtering.get_static_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.filter.x[3],0.0])
                )


                dynamic_points = self.vel_filtering.get_dynamic_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.filter.x[3],0.0])
                )

                #filter out ground detections
                radar_points = self.localizer.remove_sensor_self_detections(static_points[:,:2])
                dynamic_points = self.localizer.remove_sensor_self_detections(dynamic_points[:,:2])

                # radar_points = self.vel_filtering.remove_dynamic_clusters_from_static_detections(
                #     static_detections=static_points,
                #     dynamic_detections=dynamic_points
                # )
                

                self.dynamic_point_cloud_stacker.add_points(
                    current_points=dynamic_points,
                    heading_rad=self.filter.x[2],
                    pose_m= \
                        np.array([
                            self.filter.x[0],
                            self.filter.x[1]
                        ]),
                    current_time_s=self.filter_last_t
                )

            else:
                radar_points = self.localizer.remove_sensor_self_detections(radar_points[:,:2])
            

            self.point_cloud_stacker.add_points(
                current_points=radar_points,
                heading_rad=self.filter.x[2],
                pose_m= \
                    np.array([
                        self.filter.x[0],
                        self.filter.x[1]
                    ]),
                current_time_s=self.filter_last_t
            )

            #check to see if the vehicle has moved a sufficient amount for using
            #a combined point cloud

            #rel distance was o.5
            if (self.point_cloud_stacker.get_rel_distance_m() > 0.75) or \
                (self.point_cloud_stacker.get_rel_heading_deg() > 90) or \
                (self.point_cloud_stacker.get_elapsed_time() > 10):

                #print(self.dataset.get_vehicle_vel_data(i)[0][1])
                #print(val_dist_calc)

                #get the stacked point cloud
                static_pc = self.point_cloud_stacker.get_points()
                dynamic_pc = self.dynamic_point_cloud_stacker.get_points()

                pc = self.vel_filtering.remove_dynamic_clusters_from_static_detections_knn(
                     static_detections=static_pc,
                     dynamic_detections=dynamic_pc
                )

                #remove multipath detections
                pc = self.multipath.remove_multipath(pc)
            
                est_heading_rad,est_pose_m = self.localizer.update_odometry(
                    points=pc,
                    estimated_heading_rad=self.filter.x[2],
                    estimated_pose_m=np.array([self.filter.x[0],self.filter.x[1]])
                )

                if ((est_heading_rad is not None) and
                    (est_pose_m is not None)):

                    # #perform a measurement
                    self.filter_perform_update(
                        estimated_position_m=est_pose_m,
                        estimated_heading_rad=est_heading_rad,
                        t = self.filter_last_t
                    )

                    #save the measurement and point cloud
                    self.history_pc_stacker_update(
                        valid_stacked_point_cloud=pc,
                        icp_position_m=est_pose_m,
                        icp_heading_rad=est_heading_rad
                    )
                
                # reset the point cloud stacker
                self.velreset()
            elif self.point_cloud_stacker.get_elapsed_time() > 10:

                #if the vehicle hasn't moved significantly over the last 5 seconds,
                #go ahead and reset the point cloud stacker to prevent 
                #accumulation of false points
                # reset the point cloud stacker
                self.velreset()

            
            self.history_update_pose(
                position_m=np.array([self.filter.x[0],self.filter.x[1]]),
                heading_rad=self.filter.x[2],
                idx=i
            )

            if movie_generator:
                #plot the current state
                self.plot_compilation(
                    idx = i,
                    axs = movie_generator.axs,
                    show=False
                )

                movie_generator.save_frame(clear_axs=True)
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
        if len(self.history_filter_g) > 0:
            self.plotter_kalman.plot_chi_2_resp(
                g_thresh=self.filter.g_thresh[2],
                g_hist=np.array(self.history_filter_g),
                idx=idx,
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
        dynamic_combined_pc = self.dynamic_point_cloud_stacker.get_point_from_initial_pose()
        if combined_pc.shape[0] > 0 or dynamic_combined_pc.shape[0] > 0:

            self.plotter_localization.plot_dynamic_and_static_detections_on_map(
                static_points=combined_pc,
                dynamic_points=dynamic_combined_pc,
                heading_rad=self.point_cloud_stacker.initial_heading_rad,
                pose_m=self.point_cloud_stacker.initial_pose_m,
                ax=axs[1,2],
                show=False
            ) 
            axs[1,2].set_title("Current Stacked Point Cloud", #{}".format(len(combined_pc))
                               fontsize=self.plotter_localization.font_size_legend)
        
        
        #reset the marker size
        self.plotter_localization.marker_size=10

        if show:
            plt.show()