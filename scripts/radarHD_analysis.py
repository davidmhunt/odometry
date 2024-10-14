import sys

sys.path.append("../")
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

from mmwave_model_integrator.input_encoders.radarhd_encoder import RadarHDEncoder
from mmwave_model_integrator.model_runner.radarhd_runner import RadarHDRunner
from mmwave_model_integrator.decoders.radarhd_decoder import RadarHDDecoder

#load the necessary odometry modules
from odometry.test_benches.radar_model_ekf_tb import RadarModelEKFTB
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.plotting.movies import MovieGenerator

#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()
DATASET_PATH=os.getenv("MODEL_DATASET_DIRECTORY")
MAP_DIRECTORY=os.getenv("MAP_DIRECTORY")
RADARHD_MODEL_STATE_DICT_PATH=os.getenv("RADARHD_MODEL_STATE_DICT_PATH")
CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")

results_parent_folder = "radarHD09112024"
model_dataset_folder_name = "radarHD_comp_datasets"

datasets_to_test = {
     "WILK":{
          "map":"wilkinson.yaml",
          "datasets":[
               'WILK_1',
               'WILK_2',
               'WILK_3'
          ]
     },
     "CPSL":{
         "map":"cpsl_full.yaml",
         "datasets":[
             'cpsl_RadarHD',
              'CPSL_3',
              'CPSL_1',
              'CPSL_2'
             ]
     },
     "WILK_BASEMENT":{
         "map":"wilk_basement.yaml",
         "datasets":[
             "WILK_BASEMENT_1",
            #  "WILK_BASEMENT_2",
             "WILK_BASEMENT_3"
         ]
     }
}

def create_dir(path):

        if not os.path.isdir(path):
            os.makedirs(path)
        return

def analyze_dataset(folder_name,file_name,map_file,generate_movie=False):

    #initialize the dataset
    dataset = CpslDS(
        dataset_path=os.path.join(DATASET_PATH,model_dataset_folder_name,folder_name,file_name),
        radar_folder="radar_0",
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

    #radar config manager
    cfg_manager = ConfigManager()
    cfg_path = os.path.join(CONFIG_DIRECTORY,"RadarHD.cfg")
    cfg_manager.load_cfg(cfg_path)
    cfg_manager.compute_radar_perforance(profile_idx=0)

    #initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.5,#originally 0.1
        icp_best_points_percentile=80, #originally 65
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=7, #originally 5
        icp_max_iterations=20,
        self_detection_radius_m=1.5 #originally 1.5
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

    #initialize model encoder, runner, and decoder
    encoder = RadarHDEncoder(
        config_manager=cfg_manager,
        range_max=10.8,
        num_range_bins=256,
        mag_threshold=0.05,
        num_az_angle_bins=64
    )

    runner = RadarHDRunner(
        state_dict_path=RADARHD_MODEL_STATE_DICT_PATH,
        cuda_device="cuda:0"
    )

    decoder = RadarHDDecoder(
        max_range_m=10.8,
        num_range_bins=256,
        num_angle_bins=512
    )

    test_bench = RadarModelEKFTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        encoder=encoder,
        runner=runner,
        decoder=decoder
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
    
    #save the analysis
    result_folder="{}/Results".format(results_parent_folder)
    create_dir(result_folder)
    test_bench.analyze(
        save_folder_path=result_folder,
        file_name=file_name,
        export_to_csv=True
    )

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