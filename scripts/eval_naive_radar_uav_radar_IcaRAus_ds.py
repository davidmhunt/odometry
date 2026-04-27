import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

#load the necessary odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.plotting.movies import MovieGenerator
from odometry.test_benches.naive_radar_baseline_tb import NaiveRadarBaselineTB
from odometry.test_benches._test_bench import PredictionSource, GroundTruthSource, OdomCoordinateFrame

#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()

DATASET_PATH = "/data/IcaRAus/datasets/UAV/Radar_datasets"
MAP_DIRECTORY = "/data/IcaRAus/maps"

config_label = "eval_naive_radar_uav_radar_IcaRAus_ds"
results_parent_folder = "{}_eval".format(config_label)

datasets_to_test = {
     "vicon_box":{
          "map":"north_vicon_1.yaml",
          "datasets":[
            "vicon_box_1",
            "vicon_box_2_video",
            "vicon_box_3",
            "vicon_box_4",
            "vicon_box_5"
          ]
     },
     "vicon_cross":{
          "map":"north_vicon_1.yaml",
          "datasets":[
            "vicon_cross_1",
            "vicon_cross_2"
          ]
     },
     "vicon_box_rotate":{
        "map":"north_vicon_1.yaml",
        "datasets":[
            "vicon_box_rotate_1",
        ]
     }
}

def create_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)
        return

def analyze_dataset(
        folder_name,
        file_name,
        map_file,
        generate_movie=False):

    #initialize the dataset
    dataset = CpslDS(
        dataset_path=os.path.join(DATASET_PATH,folder_name,file_name),
        radar_pc_folder="radar_combined_pc",
        vehicle_odom_folder="vehicle_odom",
        vicon_folder="vicon_x500_8"
    )

    #initialize the map handler
    map_handler = MapHandler(
        maps_folder=MAP_DIRECTORY,
        map_file=map_file
    )

    #initialize the localizers (Using same parameters as IcaRAus evaluation)
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=85,
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=7,
        icp_max_iterations=5,
        self_detection_radius_m=0
    )

    lidar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.1,
        icp_best_points_percentile=50,
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=10,
        icp_max_iterations=20,
        self_detection_radius_m=1.0
    )

    #initialize the test bench
    test_bench = NaiveRadarBaselineTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        use_filters=True,
        prediction_source=PredictionSource.VEHICLE_ODOM,
        gt_source=GroundTruthSource.MOTION_CAPTURE,
        odom_frame=OdomCoordinateFrame.NED
    )

    if file_name == "north_1st_4":
        start_heading = np.deg2rad(45)
        start_pose = np.array([1.0, 0.5])
    else:
        start_heading = np.deg2rad(0)
        start_pose = np.array([0.00, 0.00])

    #initialize the localization
    new_heading_rad, new_pose_m = test_bench.init_localization(
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

    if generate_movie:
        #loading directory from .env file
        MOVIE_TEMP_DIRECTORY = os.getenv("MOVIE_TEMP_DIRECTORY")

        #initialize the movie maker
        movie_generator = MovieGenerator(
            temp_dir_path=os.path.join(MOVIE_TEMP_DIRECTORY, results_parent_folder)
        )
        movie_generator.initialize_figure(
            nrows=2,
            ncols=3,
            figsize=(15, 10)
        )
        
        movie_folder = "{}/Movies".format(results_parent_folder)
        create_dir(movie_folder)
        movie_generator.start_movie(
            video_file_name="{}/{}.mp4".format(movie_folder, file_name),
            fps=10
        )
    else:
        movie_generator = None
    
    #run the dataset
    end_idx = dataset.num_frames
    test_bench.run(
        max_frame=end_idx,
        gt_enabled=True,
        movie_generator=movie_generator
    )
    
    if generate_movie:
        movie_generator.save_movie()
    
    #save the analysis
    result_folder = "{}/Results".format(results_parent_folder)
    create_dir(result_folder)
    test_bench.analyze(
        save_folder_path=result_folder,
        file_name=file_name,
        export_to_csv=True
    )

    #save the position history plot for checking
    position_history_folder = "{}/Images/position_history".format(results_parent_folder)
    create_dir(position_history_folder)
    fig, axs = plt.subplots(figsize=(5, 5))
    test_bench.plotter_localization.plot_position_history_m(
        test_bench.history_position_m,
        test_bench.history_position_m_gt,
        history_position_m_inertial=test_bench.history_position_m_inertial,
        idx=end_idx - 1,
        ax=axs,
        show=False
    )
    fig.savefig("{}/{}.png".format(position_history_folder, file_name))
    plt.close(fig)


if __name__ == "__main__":
    for folder_name in datasets_to_test.keys():
         map_name = datasets_to_test[folder_name]["map"]
         for file_name in datasets_to_test[folder_name]["datasets"]:
            print("analyzing: {}".format(file_name))
            
            analyze_dataset(
                folder_name=folder_name,
                file_name=file_name,
                map_file=map_name,
                generate_movie=False
            )
    
    analyzer = Analyzer()
    analyzer.show_cumulative_summary_from_csvs(
        save_folder="{}/Results".format(results_parent_folder)
    )
