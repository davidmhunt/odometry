import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from odometry.supportFns import rotation_functions
from sklearn.neighbors import NearestNeighbors

from geometries.pose.pose import Pose
from geometries.pose.orientation import Orientation
from geometries.pose.position import Position
from geometries.transforms.transformation import Transformation

from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.localization._localizer import _Localizer
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.plotting.plotter_localization import PlotterLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.plotting.plotter_pc_grids import PlotterPCGrid
from odometry.analyzers.analyzer import Analyzer
from odometry.estimators.estimators import (
    _ExtendedKalmanFilter,
    KalmanXYPhiSpeedGyroEncoder,
    Inertial)
from odometry.point_cloud_processing.temporal_pc_stacker import temporalPcStacker
from odometry.point_cloud_processing.multipath import MultiPath
from odometry.point_cloud_processing.vel_filtering import VelFiltering
from odometry.plotting.movies import MovieGenerator

class _TestBench:

    def __init__(self,
                 gt_localizer:icp2DLocalization,
                 map_handler:MapHandler,
                 dataset:CpslDS,
                 localizer:_Localizer=None,
                 use_filters:bool = True,
                 use_vehicle_odom:bool = False) -> None:
        
        #initialize the localizer
        self.localizer:_Localizer = localizer
        self.gt_localizer:icp2DLocalization = gt_localizer

        #vehicle_movement_flag
        self.vehicle_moving = False

        #last pose and heading
        self.latest_pose_m:np.ndarray = None
        self.latest_heading_rad:float = None

        #filter parameters
        self.use_filters = use_filters
        self.filter:KalmanXYPhiSpeedGyroEncoder = None
        self.inertial_integrator:KalmanXYPhiSpeedGyroEncoder = None
        self.filter_gt:KalmanXYPhiSpeedGyroEncoder = None #gt filter
        self.filter_R:np.ndarray = None
        self.filter_msmt_component:list = None
        
        #filter time tracking
        self.filter_last_t:float = None
        self.filter_next_vel_data:np.ndarray = None

        #vehicle odometry usage flag
        self.use_vehicle_odom = use_vehicle_odom
        self.previous_vehicle_odom_pose:Pose = Pose()

        #load the datasets
        self.map_handler:MapHandler = map_handler
        self.dataset:CpslDS = dataset

        #initialize a plotter
        self.plotter_localization = PlotterLocalization(dataset,map_handler)
        self.plotter_pc_grid = PlotterPCGrid(dataset,map_handler)
        self.plotter_kalman = PlotterKalman()
        #initialize an analyzer class
        self.analyzer = Analyzer()

        #localization histories
        self.history_position_m = None
        self.history_heading_deg = None
        self.history_position_m_gt = None
        self.history_heading_deg_gt = None
        self.history_localizers_reset()

        #combined point cloud processing history
        self.history_pc_processor_point_cloud:list = None
        self.history_pc_processor_position_m:list = None
        self.history_pc_processor_heading_rad:list = None
        self.history_pc_processor_reset()

        #point cloud quality history
        self.pc_quality_dist_thresh_m = 0.5
        self.history_pc_quality_distances:list = None
        self.history_pc_quality_num_quality_points:list = None
        self.pc_quality_clusterer:NearestNeighbors = None
        self.history_pc_quality_reset()

        #timing history
        self.history_update_periods:list = None
        self.history_update_actitve_compute_times:list = None
        self.history_current_update_period_time:float = 0.0
        self.history_current_active_compute_time:float = 0.0
        self.history_timing_reset()

        #kalman filter histories
        self.history_filter_est = None
        self.history_filter_g = None
        self.history_filter_y = None
        self.history_filter_p = None

        #point cloud processing
        self.vel_filtering:VelFiltering = VelFiltering(
            v_thresh=0.05,
            min_static_rejection_radius=0.25,
            dynamic_cluster_eps=0.5,
            dynamic_cluster_min_samples=15
        )

        #TODO: Child add point cloud processing abilities

        return
    
    ####################################################################
    #Initializing localizers and filters
    #################################################################### 
    
    def init_localization(self,
                          est_start_heading_rad,
                          est_start_pose_m,
                          show = False,
                          gyro_bias=-0.0024): #gyro bias for radnav dataset
        

        if self.gt_localizer:
            #load the map points into the localizers
            self.gt_localizer.load_map_point_cloud(
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

        if self.localizer:

            #load map points into the localizer
            self.localizer.load_map_point_cloud(
                map_points=self.map_handler.map_points
            )

            #get the first points in the localizer point cloud
            init_points = self.dataset.get_radar_data(idx=0)
            init_points = init_points[:,:2]

            if not self.gt_localizer:

                new_heading_rad,new_pose_m = self.localizer.update_odometry(
                        points=init_points,
                        estimated_heading_rad=est_start_heading_rad,
                        estimated_pose_m=est_start_pose_m
                    )
            self.localizer.reset_odometry(
                pose=new_pose_m,
                heading_rad = new_heading_rad
            )
        

            if new_heading_rad is None:
                new_heading_rad = est_start_heading_rad
                new_pose_m = est_start_pose_m
                print("radar icp failed to find initial location, using est start pose")      
        

        if show and self.gt_localizer:
            self.plotter_localization.plot_detections_on_map(
                current_points=init_gt_points,
                heading_rad=new_heading_rad,
                pose_m=new_pose_m,
                show=show
            )

            #reset the last heading and pose
        self.latest_pose_m = new_pose_m
        self.latest_heading_rad = new_heading_rad

        if self.use_filters:

            self.init_filter(
                est_start_heading_rad=new_heading_rad,
                est_start_position_m=new_pose_m,
                start_time_s = self.get_dataset_start_time(idx=0),
                gyro_bias=gyro_bias
            )
        
        if self.use_vehicle_odom:
            self.init_vehicle_odometry()
        
        return new_heading_rad,new_pose_m
    
    def init_filter(self,
                    est_start_heading_rad:float,
                    est_start_position_m:np.ndarray,
                    start_time_s:float,
                    gyro_bias:float = 0.0):
        
        #declare initial state [x,y,phi,speed,gyro bias, encoder bias]
        x0 = np.array([
            est_start_position_m[0],
            est_start_position_m[1],
            est_start_heading_rad,
            0,
            gyro_bias,
            0
        ])

        #declare initial state covariance matrix originally [5,5,0.1,1,1e-2,1e-2])
        P0 = np.diag([5,5,0.1,1,1e-7,1e-2])

        #filter for gt
        self.filter_gt = KalmanXYPhiSpeedGyroEncoder(
            t0=start_time_s,
            x0 = x0.copy(),
            P0 = P0,
            chi2_pct=0.95, #originally 0.95
            do_chi2=True
        )

        #inertial integrator
        self.inertial_integrator = KalmanXYPhiSpeedGyroEncoder(
            t0=start_time_s,
            x0 = x0.copy(),
            P0 = P0,
            chi2_pct=0.95, #originally 0.95
            do_chi2=True
        )

        #filter under test
        self.filter = KalmanXYPhiSpeedGyroEncoder(
            t0=start_time_s,
            x0 = x0.copy(),
            P0 = P0,
            chi2_pct=0.95, #originally 0.95
            do_chi2=True
        )

        #define the observation noise
        # self.filter_R = np.diag([1.0,1.0,1.0]) ** 2 #original values
        self.filter_R = np.diag([0.5,0.5,0.25]) ** 2
        self.filter_msmt_component = ["x","y","phi"]

        #reset filter histories
        self.history_filters_reset()

        #reset filter time
        self.filter_last_t = \
            self.dataset.get_imu_full_data(idx=0)[0,0]
            
    ####################################################################
    #Histories (localizers)
    #################################################################### 

    def history_localizers_reset(self):

        n = self.dataset.num_frames

        #reset the pose histories
        self.history_position_m = np.zeros(shape=(n,2),dtype=np.double)
        self.history_position_m_gt = np.zeros(shape=(n,2),dtype=np.double)

        #reset the orientation histories
        self.history_heading_deg = np.zeros(shape=(n),dtype=np.double)
        self.history_heading_deg_gt = np.zeros(shape=(n),dtype=np.double)
    
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
    def history_pc_processor_reset(self):

        self.history_pc_processor_point_cloud = []
        self.history_pc_processor_position_m = []
        self.history_pc_processor_heading_rad = []

        return

    def history_pc_processor_update(
            self,
            point_cloud:np.ndarray,
            position_m:np.ndarray,
            heading_rad:float):
        """Save the most recently computed stacked point cloud and its
        estimated position and heading

        Args:
            point_cloud (np.ndarray): Nx2 array of points
                corresponding to the most recent processed point cloud
            position_m (np.ndarray): Nx2 array corresponding to the
                position from the most recent estimate
            heading_rad (float): heading corresponding to the position
                estimate using the most recent stacked point cloud
        """
        
        self.history_pc_processor_point_cloud.append(point_cloud)
        self.history_pc_processor_position_m.append(position_m)
        self.history_pc_processor_heading_rad.append(heading_rad)

    ####################################################################
    #Histories (point cloud quality)
    ####################################################################
    def history_pc_quality_reset(self):

        self.history_pc_quality_num_quality_points = []
        self.history_pc_quality_distances = []
        self.pc_quality_clusterer = NearestNeighbors(
            n_neighbors=1,
            algorithm='kd_tree'
        ).fit(
            self.map_handler.map_points
        )

        return

    def history_pc_quality_update(
            self,
            point_cloud:np.ndarray,
            gt_pc:np.ndarray,
            gt_position_m:np.ndarray,
            gt_heading_rad:float):
        """Save the most recently computed stacked point cloud and its
        estimated position and heading

        Args:
            point_cloud (np.ndarray): Nx2 array of corresponding
                to the most recent sensed point cloud in the sensor
                frame
            gt_pc (np.ndarray): Nx2 array of point corresponding
                to the most recent ground truth point cloud in the sensor frame
            gt_position_m (np.ndarray): Nx2 array corresponding to the
                position of the agent in the map
            gt_heading_rad (float): heading corresponding to the
                orientation of the agent in the map
        """
        
        #align the points with the map
        if point_cloud.shape[0] > 0:

            #TODO: Check if this is needed (originally commented out)
            # aligned_points = rotation_functions.apply_rot_trans(
            #     points=point_cloud[:,0:2],
            #     rot_angle_rad=gt_heading_rad,
            #     trans=gt_position_m
            # )

            #fit to the current gt point cloud
            self.pc_quality_clusterer = NearestNeighbors(
                n_neighbors=1,
                algorithm='kd_tree'
            ).fit(
                gt_pc[:,0:2]
            )

            #compute the distances
            distances,_ = self.pc_quality_clusterer.kneighbors(point_cloud)

            self.history_pc_quality_distances.append(distances[:,0])
            self.history_pc_quality_num_quality_points.append(
                np.sum(distances[:,0] < self.pc_quality_dist_thresh_m)
            )
    
    ####################################################################
    #Histories (timing measurement)
    ####################################################################
    def history_timing_reset(self):
        self.history_update_periods = []
        self.history_update_actitve_compute_times = []
        self.history_current_update_period_time:float = 0.0
        self.history_current_active_compute_time:float = 0.0
    
    def history_timing_save_compute_time(self):

        #save the compute times
        self.history_update_periods.append(
            self.history_current_update_period_time
        )
        self.history_update_actitve_compute_times.append(
            self.history_current_active_compute_time
        )

        #reset the tracking variables
        self.history_current_update_period_time:float = 0.0
        self.history_current_active_compute_time:float = 0.0

    ####################################################################
    #Handling Dataset Time
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
                self.dataset.vehicle_vel_enabled or
                self.dataset.vehicle_odom_enabled),\
                "cannot get time as IMU_full nor \
                vehicle_vel datasets not found"
        
        start_time = 0.0

        if self.dataset.imu_full_enabled:
            data = self.dataset.get_imu_full_data(idx=idx)
            start_time = max(data[0,0],start_time)
        
        if self.dataset.vehicle_vel_enabled:
            data = self.dataset.get_vehicle_vel_data(idx=idx)
            start_time = max(data[0,0],start_time)

        if self.dataset.vehicle_odom_enabled:
            data = self.dataset.get_vehicle_odom_data(idx=idx)
            start_time = max(data[0,0],start_time)
        
        return start_time

    ####################################################################
    #Filter Predictions and Updates
    ####################################################################
    def vehicle_vel_check_for_movement(self,vel_data:np.ndarray):

        if vel_data[1] == 0.0 and vel_data[2] == 0.0:

            self.vehicle_moving = False
        else:
            self.vehicle_moving = True
    
    def filters_predict_from_frame_samples(self, idx = 0, gt_enabled=False):

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

            #predict the localization filter forward
            self.inertial_integrator.predict(
                dt=dt,
                inertial=inertial,
                check_P=False
            )

            self.filter.predict(
                dt=dt,
                inertial=inertial,
                check_P=False
            )

            if gt_enabled and (self.gt_localizer is not None):
                self.filter_gt.predict(
                    dt=dt,
                    inertial=inertial,
                    check_P=False
                )

            #save last time
            self.filter_last_t = imu_data[i,0]

            #update histories
            self.history_filters_update_state_history()

            #update vehicle moving flag
            self.vehicle_vel_check_for_movement(vel_data[i])

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

    def filter_gt_perform_update(self,
                              estimated_position_m:np.ndarray,
                              estimated_heading_rad:np.ndarray,
                              t:float):
        
        z = np.array([
            estimated_position_m[0],
            estimated_position_m[1],
            estimated_heading_rad
        ])

        self.filter_gt.update(
            t=t,
            z=z,
            R=self.filter_R,
            msmt_components=self.filter_msmt_component
        )

    ####################################################################
    #Vehicle odom updates
    ####################################################################
    def init_vehicle_odometry(
            self
    ):
        
        #get the initial odometry point
        #indexed by [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz]
        initial_odom_data = self.dataset.get_vehicle_odom_data(idx=0)[-1,1:8]
        self.previous_vehicle_odom_pose = Pose(
            position=Position(
                x=initial_odom_data[0],
                y=initial_odom_data[1],
                z=initial_odom_data[2]
            ),
            orientation=Orientation(
                qw=initial_odom_data[3],
                qx=initial_odom_data[4],
                qy=initial_odom_data[5],
                qz=initial_odom_data[6]
            )
        )
        # self.previous_vehicle_odom_pose = self.previous_vehicle_odom_pose.flu_from_ned()
    
    def get_odom_transformation(
            self,
            current_idx:int
    )->tuple:
        """Get the odometry transformation between the previous and current odometry frames.
        Note: the translation/rotation is converted to be in the world coordinate frame

        Args:
            current_idx (int): _description_

        Returns:
            tuple: _description_
        """

        if current_idx == 269:
            pass

        #get the current odometry pose
        #indexed by [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz]
        current_odom_data = self.dataset.get_vehicle_odom_data(idx=current_idx)[-1,1:8]
        current_odom_pose = Pose(
            position=Position(
                x=current_odom_data[0],
                y=current_odom_data[1],
                z=current_odom_data[2]
            ),
            orientation=Orientation(
                qw=current_odom_data[3],
                qx=current_odom_data[4],
                qy=current_odom_data[5],
                qz=current_odom_data[6]
            )
        )
        # current_odom_pose = current_odom_pose.flu_from_ned()

        #compute the transformation from the previous to current odom frame
        transformation = Transformation.from_orig_to_new(
            original_pose=self.previous_vehicle_odom_pose,
            new_pose=current_odom_pose
        )
        
        #get the rotation adjustment
        rotation = transformation.rotation
        rotation = Orientation(
            qx=rotation[0],
            qy=rotation[1],
            qz=rotation[2],
            qw=rotation[3]
        )
        heading_change_rad = rotation.to_euler(degrees=False)[2]


        #get the translation - need to move from odom -> body -> world
        translation = transformation.translation

        #compute the appropriate transforms to adjust the coordinate frame coorectly
        trans_odom_to_body = Transformation.from_orig_to_new(
            original_pose=Pose(),
            new_pose=Pose(
                orientation=self.previous_vehicle_odom_pose.orientation
            )
        )
        trans_body_to_map = Transformation.from_orig_to_new(
            original_pose=Pose(),
            new_pose=Pose(
                orientation=Orientation.from_euler(
                    yaw=self.latest_heading_rad,
                    degrees=False
                )
            )
        )
        translation = trans_odom_to_body.apply_transformation(translation)
        translation = trans_body_to_map.apply_transformation(translation)

        #update the preious pose
        self.previous_vehicle_odom_pose = current_odom_pose

        return translation[0:2],heading_change_rad



    ####################################################################
    #Processing point clouds
    ####################################################################
    def process_point_cloud(
            self,
            point_cloud_raw:np.ndarray,
            static_points:np.ndarray,
            dynamic_points:np.ndarray,
            current_pose:Pose,
            gt_points:np.ndarray=np.empty(shape=(0,2))) -> np.ndarray:
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
            gt_points (np.ndarray,optional): [x,y] point cloud corresponding to 
                the ground truth detections. Can be used for evaluation of generated
                point cloud
                Defaults to np.empty(shape=(0,2)).

        Returns:
            np.ndarray: [x,y,z] point cloud to be used for down stream localization
                tasks. Returns empty array if no points available or if no point 
                cloud ready to be used
        """
        #TODO: Implemented by child
        return np.empty(shape=(0,3))

    ####################################################################
    #Running localization for the dataset
    ####################################################################

    def run(
            self,
            max_frame=-1,
            gt_enabled=True,
            movie_generator:MovieGenerator = None):
        """Run the test bench

        Args:
            max_frame (int, optional): The frame to run the test bench up to.
                -1 indicates to run the entire dataset. Defaults to -1.
            gt_enabled (bool, optional): On True, additionally computes
                ground truth trajectories as well. Defaults to True.
            movie_generator (MovieGenerator, optional): When provided with a 
                MovieGenerator, additionally generates a movie. Defaults to None.
        """
        if max_frame == -1:
            max_frame = self.dataset.num_frames

        for i in tqdm(range(max_frame)):

            #start time tracking
            start_time = time.time()

            if self.use_filters:

                #predict the states forward
                self.filters_predict_from_frame_samples(
                    idx=i,
                    gt_enabled=gt_enabled)
            
            elif self.use_vehicle_odom:

                translation,heading_update = self.get_odom_transformation(current_idx=i)

                #TODO: This is currently broken, but I'm not sure why
                self.latest_pose_m -= translation
                self.latest_heading_rad += heading_update

            #process lidar ground truth
            if gt_enabled and (self.gt_localizer is not None):
                # update the lidar ground truth
                gt_points = self.dataset.get_lidar_point_cloud_raw(idx=i)

                #filter out ground, set z coordinate to 0 for remaining points
                valid_points = gt_points[:,2] > -0.2 #filter out ground
                valid_points = valid_points & (gt_points[:,2] < 0.1) #higher elevation points
                gt_points = gt_points[valid_points,:3]
                gt_points[:,2] = 0.0
                
                if self.use_filters:
                    new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
                        points=gt_points[:,0:2],
                        estimated_heading_rad=self.filter_gt.x[2],
                        estimated_pose_m=np.array(
                            [self.filter_gt.x[0],self.filter_gt.x[1]])
                    )

                    #perform a measurement
                    self.filter_gt_perform_update(
                        estimated_position_m=new_pose_m,
                        estimated_heading_rad=new_heading_rad,
                        t = self.filter_last_t
                    )

                    self.history_update_pose_gt(
                        position_m=np.array(
                            [self.filter_gt.x[0],self.filter_gt.x[1]]
                        ),
                        heading_rad=self.filter_gt.x[2],
                        idx = i
                    )
                else:
                    new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
                        points=gt_points[:,0:2],
                        estimated_heading_rad=self.gt_localizer.current_heading_rad,
                        estimated_pose_m=self.gt_localizer.current_pose_m.copy()
                    )

                    self.history_update_pose_gt(
                        position_m=self.gt_localizer.current_pose_m.copy(),
                        heading_rad=self.gt_localizer.current_heading_rad,
                        idx = i
                    )

                
            else:
                gt_points = np.empty(shape=(0,3))

            #process radar detections
            if self.localizer and self.vehicle_moving:
                
                #get the combined radar point cloud [x,y,z,vel]
                radar_points = self.dataset.get_radar_data(idx=i)

                static_points = self.vel_filtering.get_static_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.filter.x[3],0.0])
                )

                dynamic_points = self.vel_filtering.get_dynamic_detections(
                    detections=radar_points,
                    ego_vel=np.array([self.filter.x[3],0.0])
                )

                #get the current pose
                current_pose = Pose(
                    position=Position(
                        x = self.inertial_integrator.x[0],
                        y = self.inertial_integrator.x[1]
                    ),
                    orientation=Orientation.from_euler(
                        yaw=self.inertial_integrator.x[2],
                        degrees=False
                    )
                )

                #process the point cloud
                pc = self.process_point_cloud(
                    point_cloud_raw=radar_points,
                    static_points=static_points,
                    dynamic_points=dynamic_points,
                    current_pose=current_pose,
                    gt_points=gt_points
                )

                if pc.shape[0] > 0:
                   
                    est_heading_rad,est_pose_m = self.localizer.update_odometry(
                        points=pc[:,0:2],
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
                        self.history_pc_processor_update(
                            point_cloud=pc[:,0:2],
                            position_m=est_pose_m,
                            heading_rad=est_heading_rad
                        )

                        if gt_enabled:
                            self.history_pc_quality_update(
                                point_cloud=pc[:,0:2],
                                gt_pc=gt_points[:,0:2],
                                gt_position_m=self.filter_gt.x[0:2],
                                gt_heading_rad=self.filter_gt.x[2]
                            )
                #update the time tracking
                stop_time = time.time()
                self.history_current_update_period_time += (1/20.0)
                self.history_current_active_compute_time += \
                    (stop_time - start_time)
                self.history_timing_save_compute_time()

            if self.use_filters:    
                #save pose history
                self.latest_pose_m = self.filter.x[0:2]
                self.latest_heading_rad = self.filter.x[2]
                                
            self.history_update_pose(
                position_m=self.latest_pose_m,
                heading_rad=self.latest_heading_rad,
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
    def analyze(self,
                save_folder_path:str="Results",
                file_name:str="summary",
                export_to_csv=False):

        self.analyzer.show_summary_statistics(
            self.history_position_m,
            self.history_position_m_gt,
            self.history_heading_deg,
            self.history_heading_deg_gt,
            self.history_pc_quality_distances,
            self.history_pc_quality_num_quality_points
        )

        if export_to_csv:
            self.analyzer.record_error_statistics(
                self.history_position_m,
                self.history_position_m_gt,
                self.history_heading_deg,
                self.history_heading_deg_gt,
                self.history_pc_quality_distances,
                self.history_pc_quality_num_quality_points,
                save_folder=save_folder_path,
                file_name=file_name
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
        
        pass