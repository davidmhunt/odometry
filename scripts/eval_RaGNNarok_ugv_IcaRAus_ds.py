import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

#load the necessary odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.plotting.plotter_kalman import PlotterKalman
from odometry.plotting.movies import MovieGenerator
from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
from odometry.test_benches._test_bench import _TestBench, PredictionSource, GroundTruthSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator_gnn import TemporalDensityPCIntegratorGNN
from odometry.point_cloud_processing.accumulation.integrators.ragnnarok_pc_integrator_gnn import RagnnarokPointCloudIntegratorGNN
from odometry.test_benches.ragnnarok_point_cloud_integrator_tb import RaGNNPointCloudIntegratorTB

from mmwave_model_integrator.torch_training.models.SAGEGnn import SageGNNClassifier



from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy


from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder

from mmwave_model_integrator.config import Config
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.DensifyingDeepDynamicEdgeConvGnn import DensifyingDeepDynamicEdgeConvGnn

#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()

DATASET_PATH = "/data/IcaRAus/datasets/UGV"
MAP_DIRECTORY = "/data/IcaRAus/maps"
GENERATED_DATASETS_PATH = "/data/IcaRAus/generated_datasets"

normalize_frames = True

config_label = "eval_RaGNNarok_ugv_IcaRAus_ds"

#model information
model_config_path = "/home/david/Documents/odometry/submodules/mmwave_model_integrator/configs/RaGNNarok/RaGNNarok_final_IcaRAus_ds.py"
model_state_dict_path = "/home/david/Documents/odometry/submodules/mmwave_model_integrator/scripts/working_dir/RaGNNarok/RaGNNarok_final_IcaRAus_ds.pth"

results_parent_folder = "{}_eval".format(config_label)


datasets_to_test = {
     "WILK":{
          "map":"wilk_map.yaml",
          "datasets":[
            # "IcaRAus_ugv_wilk_1_5m",
            "IcaRAus_ugv_wilk_2_5m",
            "IcaRAus_ugv_wilk_3_5m"
          ]
     },
     "CPSL":{
         "map":"cpsl_map.yaml",
         "datasets":[
            #  'IcaRAus_ugv_cpsl_1_5m',
             'IcaRAus_ugv_cpsl_2_5m',
             'IcaRAus_ugv_cpsl_3_5m',
         ]
     },
     "NORTH_1ST":{
         "map":"north_1st_map.yaml",
         "datasets":[
            #  'north_1st_1', #unmapped area
            #  'north_1st_2', #unmapped area
             'north_1st_3',
             'north_1st_4',
             'north_1st_5'
         ]
     },
     "NORTH_VICON":{
         "map":"north_vicon_1.yaml",
         "datasets":[
            #  'north_vicon_1',
            #  'north_vicon_2',
             'north_vicon_3',
             'north_vicon_4',
             'north_vicon_5'
         ]
     },
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
        lidar_folder="lidar",
        camera_folder="camera",
        vehicle_odom_folder="vehicle_odom"
    )

    #initialize the map handler
    map_handler = MapHandler(
        maps_folder=MAP_DIRECTORY,
        map_file=map_file
    )

    #initialize the dataset encoders
    input_encoder = _NodeEncoder()

    #initialize the localizers (RaGNNarok tuned parameters)
    radar_odometry = icp2DLocalization(
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=60,
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

    #instantiate the model
    config = Config(model_config_path)
    model_cfg = config.model
    model_type = model_cfg.pop('type')
    model = SageGNNClassifier(**model_cfg)

    dataset_cfg = config.trainer["dataset"]
    enable_downsampling = dataset_cfg.get("enable_downsampling", False)

    runner = GNNRunner(
        model=model,
        state_dict_path=model_state_dict_path,
        cuda_device="cuda:0" if torch.cuda.is_available() else "cpu",
        edge_radius=10.0,
        enable_downsampling=enable_downsampling,
        use_sigmoid=False,
        print_stats=False
    )

    point_cloud_integrator = RagnnarokPointCloudIntegratorGNN(
        gnn_runner=runner,
        grid_resolution_m_prob=0.1,
        grid_max_distance_m_prob=5.0,
        num_frames_history_prob=20,
        grid_resolution_m_hist=0.2,
        grid_max_distance_m_hist=5.0,
        num_frames_history_hist=10,
        min_detection_radius=1.0,
        max_detection_radius=5.0,
        gt_distance_threshold_m_prob=0.2
    )

    #initialize the test bench
    test_bench = RaGNNPointCloudIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        model_dataset_generator=None,
        use_filters=True,
        prediction_source=PredictionSource.VEHICLE_ODOM,
        gt_source=GroundTruthSource.LIDAR,
        odom_frame=OdomCoordinateFrame.FLU
    )

    
    if file_name== "north_1st_4":
        start_heading = np.deg2rad(45)
        start_pose = np.array([1.0,0.5])
    else:
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
        
        movie_folder="{}/Movies".format(results_parent_folder)
        create_dir(movie_folder)
        movie_generator.start_movie(
            video_file_name="{}/{}.mp4".format(movie_folder,file_name),
            fps=10
        )
    else:
        movie_generator=None
    
    #run the dataset
    end_idx = dataset.num_frames
    test_bench.run(
        max_frame=end_idx,
        gt_enabled=True,
        movie_generator=movie_generator,
        generate_dataset=False,
        normalize_frames=normalize_frames)
    
    if generate_movie:
        movie_generator.save_movie()
    
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
        history_position_m_inertial=test_bench.history_position_m_inertial,
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