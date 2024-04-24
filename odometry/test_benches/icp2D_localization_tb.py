import numpy as np
from tqdm import tqdm
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.datasets.radnav_ds import radnavDS
from odometry.datasets.map_handler import MapHandler
from odometry.plotting.plotter import Plotter
from odometry.analyzers.analyzer import Analyzer

class icp2DLocalizationTB:

    def __init__(self,
                 localizer:icp2DLocalization,
                 gt_localizer:icp2DLocalization,
                 map_handler:MapHandler,
                 dataset:radnavDS) -> None:
        
        #initialize the localizer
        self.localizer:icp2DLocalization = localizer
        self.gt_localizer:icp2DLocalization = gt_localizer

        #load the datasets
        self.map_handler:MapHandler = map_handler
        self.dataset:radnavDS = dataset

        #initialize a plotter
        self.plotter = Plotter(dataset,map_handler)

        #initialize an analyzer class
        self.analyzer = Analyzer()

        #histories
        self.history_position_m = None
        self.history_heading_deg = None
        self.history_position_m_gt = None
        self.history_heading_deg_gt = None
        self.history_reset()

    
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

        #get the first points in the point cloud
        init_gt_points = self.dataset.get_lidar_point_cloud(idx=0)

        new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
            points=init_gt_points,
            estimated_heading_rad=est_start_heading_rad,
            estimated_pose_m=est_start_pose_m
        )

        print("icp estimated heading:{} deg, pose:{}".format(
            np.rad2deg(new_heading_rad),new_pose_m))
        
        self.gt_localizer.reset_odometry(
            pose=new_pose_m,
            heading_rad=new_heading_rad
        )


        if show:
            self.plotter.plot_detections_on_map(
                current_points=init_gt_points,
                heading_rad=new_heading_rad,
                pose_m=new_pose_m,
                show=show
            )
    
    ####################################################################
    #Histories
    #################################################################### 

    def history_reset(self):

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
    #Running localization for the dataset
    ####################################################################

    def run(self):
        for i in tqdm(range(self.dataset.num_frames)):

            #update the lidar ground truth
            gt_points = self.dataset.get_lidar_point_cloud(idx=i)

            new_heading_rad,new_pose_m = self.gt_localizer.update_odometry(
                points=gt_points
            )

            self.history_update_pose_gt(
                position_m=new_pose_m,
                heading_rad=new_heading_rad,
                idx = i
            )

            #update the radar ground truth
            radar_points = self.dataset.get_radar_detections(idx=i)

            est_heading_rad,est_pose_m = self.localizer.update_odometry(
                points=radar_points[:,:2],
                estimated_heading_rad=new_heading_rad,
                estimated_pose_m=new_pose_m
            )

            if est_heading_rad is None:
                est_heading_rad = self.localizer.current_heading_rad
            if est_pose_m is None:
                est_pose_m = self.localizer.current_pose_m
            
            self.history_update_pose(
                position_m=est_pose_m,
                heading_rad=est_heading_rad,
                idx=i
            )
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





