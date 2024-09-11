import sys

sys.path.append("../")
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#load the necessary odometry modules
from odometry.datasets.map_handler import MapHandler
from odometry.datasets.radnav_ds import radnavDS
from odometry.test_benches.radnav_stacked_pc_tb import RadnavStackedPCTB
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.plotting.movies import MovieGenerator
from odometry.point_cloud_processing.temporal_pc_stacker import temporalPcStacker

#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()
DATASET_PATH=os.getenv("DATASET_DIRECTORY")
MAP_DIRECTORY=os.getenv("MAP_DIRECTORY")

results_parent_folder = "radnav_trajectories_09092024"

datasets_to_test = {
     "WILK_nav":{
          "map":"wilkinson.yaml",
          "datasets":[
            # 'wilk_nav_1',
            # 'wilk_nav_2',
            # 'wilk_nav_3',
            'wilk_nav_4'
            # 'wilk_nav_5',
          ]
     },
     "CPSL_nav":{
         "map":"cpsl_full.yaml",
         "datasets":[
             'cpsl_nav_1',
             'cpsl_nav_2',
             'cpsl_nav_3',
             'cpsl_nav_4'
            #  'cpsl_nav_5'
            ]
     },
     "WILK_BASEMENT_nav":{
         "map":"wilk_basement_revB.yaml",
         "datasets":[
             'wilk_basement_nav_1',
             'wilk_basement_nav_2',
             'wilk_basement_nav_3',
             'wilk_basement_nav_4'
            #  'wilk_basement_nav_5'
            ]
     }
}

def create_dir(path):

        if not os.path.isdir(path):
            os.makedirs(path)
        return

def compute_trajectories(folder_name,file_name,map_file,generate_movie=False):

    #initialize the dataset
    dataset = radnavDS(
        dataset_path=os.path.join(DATASET_PATH,folder_name,file_name),
        radar_folder="radar_combined",
        lidar_folder="lidar",
        camera_folder="camera",
        imu_orientation_folder="imu_data",
        imu_full_folder="imu_data_full",
        vehicle_vel_folder="vehicle_vel"
    )

    #initialize the map handler
    map_handler = MapHandler(
        maps_folder=MAP_DIRECTORY,
        map_file=map_file
    )

    #initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.5,#originally 0.1
        icp_best_points_percentile=80, #originally 65
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=7, #originally 5
        icp_max_iterations=20,
        self_detection_radius_m=0 #originally 1.5
    )

    lidar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.1, #was 0.6, try 0.1
        icp_best_points_percentile=50, #was 50 - try 75
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=10,
        icp_max_iterations=20,
        self_detection_radius_m=1.0 #was 0.25, try 1.0
    )

    #initialize the point cloud stacker
    pc_stacker = temporalPcStacker(
        num_frames_static_history = 4, # originally 4
        num_frames_dynamic_history = 0, # originally 0
        refresh_distance_m = 3, #originally 3
        refresh_rot_deg = 90, #originally 90
        refresh_time_s = 5, #originally 5
        grid_resolution_m = 5e-2, #originally 5e-2
        grid_max_distance_m = 20, #originally 20
        multi_path_clustering_eps = 0.5, #originally 0.5
        multi_path_clustering_min_samples = 15,  #originally 15
        multi_path_num_frames_history=4, # originally 4
        vel_filtering_enabled = True,
        vel_filtering_v_thresh = 0.05, # originally 0.05
        vel_filtering_min_static_rejection_radius = 0.25, #originally 0.25
        vel_filtering_dynamic_cluster_eps = 0.5, #originally 0.5
        vel_filtering_dynamic_cluster_min_samples = 15, #originally 15
        min_detection_radius_m=1.5, #originally 1.5
        max_detection_range_m=20, #originally 20
        gyro_bias=-0.0024 #originally -0.0024
    )

    #initialize the test bench
    test_bench = RadnavStackedPCTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        pc_stacker=pc_stacker
    )

    start_heading = np.deg2rad(0)
    start_pose = np.array([0.00,0.00])

    #initialize the localization
    new_heading_rad,new_pose_m = test_bench.init_localization(
        est_start_heading_rad=start_heading,
        est_start_pose_m=start_pose,
        show=False
    )

    #initialize the filter
    test_bench.init_filter(
        est_start_heading_rad=new_heading_rad,
        est_start_position_m=new_pose_m,
        start_time_s=test_bench.get_dataset_start_time(idx=0),
        gyro_bias=-0.0024
    )

    #initialize the point cloud stacker
    test_bench.init_pc_stacker(
        start_time_s=test_bench.get_dataset_start_time(idx=0)
    )

    if generate_movie:
        #loading directory from .env file
        MOVIE_TEMP_DIRECTORY = os.getenv("MOVIE_TEMP_DIRECTORY")

        #initialize the movie maker
        movie_generator = MovieGenerator(
            temp_dir_path=os.path.join(MOVIE_TEMP_DIRECTORY,results_parent_folder)
        )
        movie_generator.initialize_figure(
            nrows=2,
            ncols=3,
            figsize=(15,10)
        )
    else:
        movie_generator=None
    
    #run the dataset
    end_idx = dataset.num_frames
    test_bench.run(
        max_frame=end_idx,
        gt_enabled=True,
        movie_generator=movie_generator)
    
    if generate_movie:
        movie_folder="{}/Movies".format(results_parent_folder)
        create_dir(movie_folder)
        movie_generator.save_movie(video_file_name="{}/{}.mp4".format(
            movie_folder,file_name),fps=20)
    
    # save the ground truth trajectories
    result_folder="{}/gt_trajectories".format(results_parent_folder)
    create_dir(result_folder)
    np.save("{}/{}.npy".format(result_folder,file_name),
            test_bench.history_position_m_gt)
    
    #save the estimated trajectories
    result_folder="{}/est_trajectories".format(results_parent_folder)
    create_dir(result_folder)
    np.save("{}/{}.npy".format(result_folder,file_name),
            test_bench.history_position_m)


    #save the position history plot for checking
    position_history_folder = \
        "{}/Images/position_history".format(results_parent_folder)
    create_dir(position_history_folder)
    fig, axs = plt.subplots(figsize=(5,5))
    test_bench.plotter_localization.plot_position_history_m(
        test_bench.history_position_m,
        test_bench.history_position_m_gt,
        idx=end_idx-1,
        ax=axs,
        show=False
    )
    fig.savefig("{}/{}.png".format(position_history_folder,file_name))


if __name__ == "__main__":

    for folder_name in datasets_to_test.keys():
         map_name = datasets_to_test[folder_name]["map"]
         for file_name in datasets_to_test[folder_name]["datasets"]:
            print("analyzing: {}".format(file_name))
            compute_trajectories(
                folder_name=folder_name,
                file_name=file_name,
                map_file=map_name,
                generate_movie=False
            )