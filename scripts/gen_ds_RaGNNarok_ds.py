import sys
import os
import yaml
import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# load the necessary odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.analyzers.analyzer import Analyzer
from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder

# loading environment variables
load_dotenv()

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Base paths (can be overridden by config)
DEFAULT_MAP_DIRECTORY = "/data/RaGNNarok/ugv_datasets/maps/"
DEFAULT_GENERATED_DATASETS_PATH = "/data/RaGNNarok/generated_datasets/"

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def generate_ds_fold(config_file, num_frames_history=50, normalize_frames=True):
    """Generates a dataset fold based on a configuration file.

    Args:
        config_file (str): Filename of the dataset configuration (e.g., 'RaGNNarok_ugv_train_f1.yaml').
        num_frames_history (int): Number of frames for accumulation.
        normalize_frames (bool): Whether to normalize frames.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "dataset_configs")
    config_path = os.path.join(config_dir, config_file)
    
    print(f"Processing config: {config_file}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config_label = config["config_label"]
    datasets_to_test = config["datasets_to_test"]
    results_parent_folder = f"{config_label}_train"
    
    # Initialize the dataset generator
    generated_datasets_path = config.get("generated_datasets_path", DEFAULT_GENERATED_DATASETS_PATH)
    generated_dataset_path = os.path.join(generated_datasets_path, f"{config_label}_train")
    
    input_encoder = _NodeEncoder()
    gt_encoder = _GTNodeEncoder()
    
    clear_existing_train_data = True
    
    # Load factories
    sys.path.append(os.path.join(os.path.dirname(__file__), "eval_configs"))
    
    for config_key in datasets_to_test.keys():
        folder_config = datasets_to_test[config_key]
        folder_name = folder_config.get("folder_name", config_key)
        map_name = folder_config["map"]
        dataset_path = folder_config["dataset_path"]
        
        # Determine which ds_gen module to use
        module_name = folder_config.get("test_bench_module")
        if module_name is None:
            raise ValueError(f"test_bench_module not specified for {config_key}")
        
        ds_gen_module = importlib.import_module(module_name)
        
        # Initialize Map Handler
        map_directory = config.get("map_directory", DEFAULT_MAP_DIRECTORY)
        map_handler = MapHandler(
            maps_folder=map_directory,
            map_file=map_name
        )

        for file_name in folder_config["datasets"]:
            print(f"Analyzing: {file_name} in {folder_name} (Module: {module_name})")
            
            # Initialize Dataset
            ds_params = {
                "dataset_path": os.path.join(dataset_path, folder_name, file_name),
                "radar_pc_folder": folder_config.get("radar_pc_folder", "radar_combined"),
                "vehicle_odom_folder": folder_config.get("vehicle_odom_folder", "vehicle_odom")
            }
            
            # Pass all other platform-specific folders if present in the config
            optional_keys = [
                "lidar_folder", "camera_folder", "vicon_folder", 
                "imu_orientation_folder", "imu_full_folder", "vehicle_vel_folder"
            ]
            for key in optional_keys:
                if key in folder_config:
                    ds_params[key] = folder_config[key]
            
            dataset = CpslDS(**ds_params)
            
            # Initialize Dataset Generator for this specific run
            dataset_generator = _OnlineDatasetGenerator(
                generated_dataset_path=generated_dataset_path,
                input_encoder=input_encoder,
                ground_truth_encoder=gt_encoder,
                generated_file_name="frame",
                input_encoding_folder="nodes",
                ground_truth_encoding_folder="labels",
                clear_existing_data=clear_existing_train_data
            )
            
            # Use Factory to get Test Bench
            test_bench = ds_gen_module.get_test_bench(
                dataset=dataset,
                map_handler=map_handler,
                model_dataset_generator=dataset_generator,
                num_frames_history=num_frames_history,
                normalize_frames=normalize_frames
            )
            
            # Start Heading/Pose logic
            start_heading = np.deg2rad(0)
            start_pose = np.array([0.00, 0.00])
            
            # Initialize localization (and filter)
            test_bench.init_localization(
                est_start_heading_rad=start_heading,
                est_start_pose_m=start_pose,
                show=False,
                gyro_bias=-0.0024
            )
            
            # Run Test Bench
            test_bench.run(
                max_frame=dataset.num_frames,
                gt_enabled=True,
                generate_dataset=True,
                normalize_frames=normalize_frames
            )
            
            # Save analysis results
            res_folder = os.path.join(results_parent_folder, "Results")
            create_dir(res_folder)
            test_bench.analyze(
                save_folder_path=res_folder,
                file_name=file_name,
                export_to_csv=True
            )
            
            # Save position history plot
            img_folder = os.path.join(results_parent_folder, "Images", "position_history")
            create_dir(img_folder)
            fig, axs = plt.subplots(figsize=(5, 5))
            test_bench.plotter_localization.plot_position_history_m(
                test_bench.history_position_m,
                test_bench.history_position_m_gt,
                test_bench.history_position_m_inertial,
                idx=dataset.num_frames - 1,
                ax=axs,
                show=False
            )
            fig.savefig(os.path.join(img_folder, f"{file_name}.png"))
            plt.close(fig)
            
            clear_existing_train_data = False

    # Cumulative summary
    analyzer = Analyzer()
    analyzer.show_cumulative_summary_from_csvs(
        save_folder=os.path.join(results_parent_folder, "Results")
    )

if __name__ == "__main__":
    # Parameters
    normalize_frames = True
    num_frames_history = 50
    
    # List of folds to process
    fold_configs = [
        "RaGNNarok_ds_train_f1.yaml",
        "RaGNNarok_ds_train_f2.yaml",
        "RaGNNarok_ds_train_f3.yaml",
    ]
    
    for config in fold_configs:
        print(f"\n{'='*60}\nStarting Fold: {config}\n{'='*60}")
        generate_ds_fold(
            config_file=config,
            num_frames_history=num_frames_history,
            normalize_frames=normalize_frames
        )
