import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#load the necessary odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.plotting.movies import MovieGenerator
from odometry.test_benches.ragnnarok_point_cloud_integrator_tb import GnnPointCloudIntegratorTB
from odometry.point_cloud_processing._point_cloud_integrator import _PointCloudIntegrator
from odometry.point_cloud_processing.pc_grid.probabilistic_pc_grid import ProbabilisticPCGrid
from odometry.point_cloud_processing.pc_grid.historical_pc_grid import HistoricalPCGrid

from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder


#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()
DATASET_PATH=os.getenv("DATASET_DIRECTORY")
MAP_DIRECTORY=os.getenv("MAP_DIRECTORY")
GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")

config_label = "RaGNNarok_1fp_20fh_0_50_th_5mRng_0_2_res"
results_parent_folder = "{}_train".format(config_label)


datasets_to_test = {
    #  "WILK":{
    #       "map":"wilkinson.yaml",
    #       "datasets":[
    #         #    'WILK_Path_1_With_Dynamic',
    #         #     'WILK_Multipath_Test_4', #test
    #             'WILK_Multipath_Test_5', #train
    #         #     'WILK_Slow_4',
    #         #     'WILK_Path_1_Slow_With_Dynamic_Trickery_1',
    #         #     'WILK_Path_1_Slow_No_Dynamic_1',
    #         #     'WILK_Slow_Walk_Test_1',
    #         #     'WILK_Path_1_With_Dynamic_2',
    #         #     'WILK_Path_1_No_Dynamic',
    #         #     'WILK_Slow_Walk_Test_2',
    #         #     'WILK_Path_1_Slow_Dynamic_1',
    #         #     'WILK_Multipath_Test_1',
    #         #     'WILK_Multipath_Test_3', #train
    #         #     'WILK_Slow_1',
    #         #     'WILK_vel_cfg_1', #test
    #         #     'WILK_Slow_2',
    #         #     'WILK_Path_1_Same_Side_Dynamic_1',
    #         #     'WILK_vel_cfg_2', #train
    #         #     'WILK_Multipath_Test_1_spin_recal',
    #         #     'WILK_Multipath_Test_2', #test
    #         #     'WILK_Slow_3'
    #       ]
    #  },
    #  "CPSL":{
    #      "map":"cpsl_full.yaml",
    #      "datasets":[
    #          'CPSL_Walk_1', #train
    #          'CPSL_Vel_2', #train
    #          'CPSL_NoVel_2',
    #          'CPSL_Walk_2', #train
    #          'CPSL_Vel_1', #train
    #          'CPSL_NoVel_1',
    #         #  'CONFIG_TEST',
    #          'CPSL_Vel_3', #test
    #         #  'CPSL_No_Move',
    #          'CPSL_Lidar_Test',
    #          'CPSL_vel_cfg_1'] #test
    #  },
     "WILK_BASEMENT":{
         "map":"wilk_basement_revB.yaml",
         "datasets":[
            #  'wilk_basement_1', #train
            #  'wilk_basement_2',] #train
             'wilk_basement_3'] #train Rag
     }
}

def create_dir(path):

        if not os.path.isdir(path):
            os.makedirs(path)
        return

def generate_gnn_dataset(
        folder_name,
        file_name,
        map_file,
        generate_movie=False,
        clear_existing_train_data=False):

    #initialize the dataset
    dataset = CpslDS(
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

    #initialize the dataset encoders
    input_encoder = _NodeEncoder()
    gt_encoder = _GTNodeEncoder()

    #initialize the dataset generator
    generated_dataset_path = os.path.join(GENERATED_DATASETS_PATH,"{}_train".format(config_label))
    dataset_generator = _OnlineDatasetGenerator(
        generated_dataset_path=generated_dataset_path,
        input_encoder=input_encoder,
        ground_truth_encoder=gt_encoder,
        generated_file_name="frame",
        input_encoding_folder="nodes",
        ground_truth_encoding_folder="labels",
        clear_existing_data=clear_existing_train_data
    )

    #initialize the localizers
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.5,#originally 0.1
        icp_best_points_percentile=80, #originally 65
        icp_convergence_translation_threshold=1e-3,
        icp_convergence_rotation_threshold=1e-4,
        icp_point_pairs_threshold=7, #originally 5
        icp_max_iterations=5,
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

    #initialize the probabilistic point cloud grid
    probabilistic_pc_grid = ProbabilisticPCGrid(
        grid_resolution_m=0.20,
        grid_max_distance_m=5.0,
        occupancy_threshold=0.20,
        num_frames_history=20
    )

    historical_pc_grid = HistoricalPCGrid(
        grid_resolution_m=0.20,
        grid_max_distance_m=5.0,
        num_frames_persistance=1
    )

    point_cloud_integrator = _PointCloudIntegrator(
        probabilistic_pc_grid=probabilistic_pc_grid,
        historical_pc_grid=historical_pc_grid,
        min_detection_radius=1.0,
        max_detection_radius=5.0,
    )

    #initialize the test bench
    test_bench = GnnPointCloudIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        model_dataset_generator=dataset_generator
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
            nrows=3,
            ncols=3,
            figsize=(15,15)
        )
    else:
        movie_generator=None
    
    #run the dataset
    end_idx = dataset.num_frames
    test_bench.run(
        max_frame=end_idx,
        gt_enabled=True,
        movie_generator=movie_generator,
        generate_dataset=True)
    
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
    clear_existing_train_data = True
    for folder_name in datasets_to_test.keys():
         map_name = datasets_to_test[folder_name]["map"]
         for file_name in datasets_to_test[folder_name]["datasets"]:
            print("analyzing: {}".format(file_name))
            
            generate_gnn_dataset(
                folder_name=folder_name,
                file_name=file_name,
                map_file=map_name,
                generate_movie=False,
                clear_existing_train_data=clear_existing_train_data
            )

            clear_existing_train_data = False
    
    analyzer = Analyzer()
    analyzer.show_cumulative_summary_from_csvs(
        save_folder="{}/Results".format(results_parent_folder)
    )