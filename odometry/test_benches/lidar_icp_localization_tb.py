import numpy as np
from tqdm import tqdm
from odometry.localization.icp2D_localization import icp2DLocalization
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.plotting.plotter_localization import PlotterLocalization
from odometry.analyzers.analyzer import Analyzer

class lidarICPLocalization:

    def __init__(self,
                 gt_localizer:icp2DLocalization,
                 map_handler:MapHandler,
                 dataset:CpslDS) -> None:
        
        #initialize the localizer
        self.gt_localizer:icp2DLocalization = gt_localizer

        #load the datasets
        self.map_handler:MapHandler = map_handler
        self.dataset:CpslDS = dataset

        #initialize a plotter
        self.plotter = PlotterLocalization(dataset,map_handler)

        #initialize an analyzer class
        self.analyzer = Analyzer()

        #histories
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
        self.history_position_m_gt = np.zeros(shape=(n,2),dtype=np.double)

        #reset the orientation histories
        self.history_heading_deg_gt = [None] * n
    
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
        return




