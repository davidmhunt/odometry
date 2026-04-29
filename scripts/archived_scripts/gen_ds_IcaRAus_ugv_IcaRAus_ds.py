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
from odometry.test_benches.temporal_density_pc_integrator_tb import TemporalDensityPCIntegratorTB
from odometry.test_benches._test_bench import _TestBench, PredictionSource, GroundTruthSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators.temporal_density_pc_integrator import TemporalDensityPCIntegrator


from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy


from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder

#analyzer
from odometry.analyzers.analyzer import Analyzer

from dotenv import load_dotenv
import os
import yaml

#loading enviroment variables
load_dotenv()
# DATASET_PATH=os.getenv("DATASET_DIRECTORY")
# MAP_DIRECTORY=os.getenv("MAP_DIRECTORY")
# GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")

DATASET_PATH = "/data/IcaRAus/datasets/UGV"
MAP_DIRECTORY = "/data/IcaRAus/maps"
GENERATED_DATASETS_PATH = "/data/IcaRAus/generated_datasets"

def create_dir(path):

        if not os.path.isdir(path):
            os.makedirs(path)
        return

def generate_gnn_dataset(
        folder_name,
        file_name,
        map_file,
        config_label,
        results_parent_folder,
        num_frames_history=50,
        normalize_frames=True,
        generate_movie=False,
        clear_existing_train_data=False):

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
        icp_matching_distance_threshold=0.25,
        icp_best_points_percentile=80,
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
        icp_max_iterations=5,
        self_detection_radius_m=1.0
    )

    #initialize the probabilistic point cloud grid
    point_cloud_integrator = TemporalDensityPCIntegrator(
            gt_distance_threshold_m=0.4,
            num_frames_history_gt=1,
            valid_fovs_deg=[(-70,70),(110,-110)],
            num_frames_history=num_frames_history,
            min_detection_radius=1.0,
            max_detection_radius=8.0,
            grid_resolution_m=0.1,
            subsample_percentage=1.0,
            gt_point_labeling_strategy=GtPointLabelingStrategy.USE_VALID_POINTS_FOR_GT_CLASSIFICATION,
            gt_occlusion_aware_clustering=OcclusionAwareClustering(
                clustering_eps=0.5,
                clustering_min_samples=12,
                angle_res_rad=0.017,
                occlusion_threshold=0.7,
                subsample_percentage=1.0,
                remove_occluded=True,
                filter_method='ray_trace'
            ),
            occlusion_aware_clustering=OcclusionAwareClustering(
                clustering_eps=0.25,
                clustering_min_samples=10,
                angle_res_rad=0.017,
                occlusion_threshold=0.9,
                subsample_percentage=0.20,
                remove_occluded=False,
                filter_method='ray_trace' #ray_trace or overlap
            )
        )

    #initialize the test bench
    test_bench = TemporalDensityPCIntegratorTB(
        localizer=radar_odometry,
        gt_localizer=lidar_odometry,
        map_handler=map_handler,
        dataset=dataset,
        point_cloud_integrator=point_cloud_integrator,
        dynamic_point_cloud_integrator=None,
        model_dataset_generator=dataset_generator,
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
        gyro_bias=-0.0024 #irrelevant here as using odom samples
    )

    if generate_movie: #generate_movie:
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
        generate_dataset=True,
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
        idx=end_idx-1,
        ax=axs,
        show=False
    )
    fig.savefig("{}/{}.png".format(position_history_folder,file_name))


if __name__ == "__main__":

    #dataset parameters
    normalize_frames = True
    num_frames_history = 50

    config_dir = os.path.join(os.path.dirname(__file__), "dataset_configs")
    config_filenames = ["IcaRAus_ugv_train_0_35_eps.yaml"]
    # config_filenames = [
    #     "IcaRAus_ugv_train_f1.yaml",
    #     "IcaRAus_ugv_train_f2.yaml",
    #     "IcaRAus_ugv_train_f3.yaml",
    #     "IcaRAus_ugv_train_f4.yaml",
    # ]

    for config_file in config_filenames:
        config_path = os.path.join(config_dir, config_file)
        print("processing config: {}".format(config_file))

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        config_label = config["config_label"]
        datasets_to_test = config["datasets_to_test"]
        results_parent_folder = "{}_train".format(config_label)

        clear_existing_train_data = True
        for folder_name in datasets_to_test.keys():
            map_name = datasets_to_test[folder_name]["map"]
            for file_name in datasets_to_test[folder_name]["datasets"]:
                print("analyzing: {}".format(file_name))
                
                generate_gnn_dataset(
                    folder_name=folder_name,
                    file_name=file_name,
                    map_file=map_name,
                    config_label=config_label,
                    results_parent_folder=results_parent_folder,
                    num_frames_history=num_frames_history,
                    normalize_frames=normalize_frames,
                    generate_movie=False,
                    clear_existing_train_data=clear_existing_train_data
                )

                clear_existing_train_data = False    

        analyzer = Analyzer()
        analyzer.show_cumulative_summary_from_csvs(
            save_folder="{}/Results".format(results_parent_folder)
        )