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

# loading environment variables
load_dotenv()

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Base paths
DEFAULT_MAP_DIRECTORY = "/data/RaGNNarok/ugv_datasets/maps/"

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def run_evaluation(base_eval_mode, model_label, dataset_config_file, results_base_dir):
    """Runs evaluation for a specific mode and dataset fold.

    Args:
        base_eval_mode (str): Base evaluation mode (e.g., 'icaraus_gnn', 'ragnnarok_gnn').
        model_label (str): Label for the model being tested.
        dataset_config_file (str): Filename of the dataset configuration YAML.
        results_base_dir (str): Base directory to save evaluation results.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "dataset_configs")
    config_path = os.path.join(config_dir, dataset_config_file)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config_label = config["config_label"]
    datasets_to_test = config["datasets_to_test"]
    
    # Platform resolution
    # In RaGNNarok datasets, they are all UGV, but we use 'platform' metadata for factory selection
    platform = config.get("platform", "ugv")
    eval_module_name = f"{base_eval_mode}_{platform}"
    
    # Handle cross-dataset evaluation naming (e.g., icaraus_gnn on ragnnarok datasets)
    if base_eval_mode == "icaraus_gnn" and "RaGNNarok" in config_label:
        eval_module_name = f"icaraus_gnn_ragnnarok_{platform}"
    
    # Results directory structure
    results_parent_folder = os.path.join(results_base_dir, f"{config_label}_{model_label if model_label else base_eval_mode}")
    create_dir(results_parent_folder)
    
    # Load evaluation module
    sys.path.append(os.path.join(os.path.dirname(__file__), "eval_configs"))
    eval_module = importlib.import_module(eval_module_name)
    
    # Load model info if applicable
    model_info = None
    if model_label:
        model_info_path = os.path.join(
            os.path.dirname(__file__), 
            "eval_configs", "model_info", 
            f"{base_eval_mode}_{'RaGNNarok_ds' if 'RaGNNarok' in config_label else 'IcaRAus_ds'}_{config_label.split('_')[-1]}.yaml"
        )
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                model_info = yaml.safe_load(f)
        else:
            print(f"Warning: Model info not found at {model_info_path}")

    for config_key in datasets_to_test.keys():
        folder_config = datasets_to_test[config_key]
        folder_name = folder_config.get("folder_name", config_key)
        map_name = folder_config["map"]
        dataset_path = folder_config["dataset_path"]
        
        # Initialize Map Handler
        map_directory = config.get("map_directory", DEFAULT_MAP_DIRECTORY)
        map_handler = MapHandler(
            maps_folder=map_directory,
            map_file=map_name
        )

        for file_name in folder_config["datasets"]:
            print(f"Evaluating: {file_name} (Mode: {eval_module_name})")
            
            # Initialize Dataset
            ds_params = {
                "dataset_path": os.path.join(dataset_path, folder_name, file_name),
                "radar_pc_folder": folder_config.get("radar_pc_folder", "radar_combined"),
                "vehicle_odom_folder": folder_config.get("vehicle_odom_folder", "vehicle_odom")
            }
            
            # Optional platform folders
            optional_keys = [
                "lidar_folder", "camera_folder", "vicon_folder", 
                "imu_orientation_folder", "imu_full_folder", "vehicle_vel_folder"
            ]
            for key in optional_keys:
                if key in folder_config:
                    ds_params[key] = folder_config[key]
            
            dataset = CpslDS(**ds_params)
            
            # Use Factory to get fully initialized Test Bench
            test_bench = eval_module.get_test_bench(
                dataset=dataset,
                map_handler=map_handler,
                model_info=model_info,
                num_frames_history=50,
                normalize_frames=True
            )
            
            # Initialize localization
            test_bench.init_localization(
                est_start_heading_rad=np.deg2rad(0),
                est_start_pose_m=np.array([0.00, 0.00]),
                show=False,
                gyro_bias=-0.0024
            )
            
            # Run Test Bench
            test_bench.run(
                max_frame=dataset.num_frames,
                gt_enabled=True,
                generate_dataset=False
            )
            
            # Save analysis results
            results_folder = os.path.join(results_parent_folder, "Results")
            create_dir(results_folder)
            test_bench.analyze(
                save_folder_path=results_folder,
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

    # Cumulative summary
    analyzer = Analyzer()
    analyzer.show_cumulative_summary_from_csvs(
        save_folder=os.path.join(results_parent_folder, "Results")
    )

if __name__ == "__main__":
    # Parameters
    results_base_dir = "/data/RaGNNarok/evaluation_results"
    
    # List of folds to process
    fold_configs = [
        "RaGNNarok_ugv_test_f1.yaml",
        "RaGNNarok_ugv_test_f2.yaml",
        "RaGNNarok_ugv_test_f3.yaml",
    ]
    
    # Modes to run
    eval_modes = [
        {"base_mode": "ragnnarok_gnn", "model_label": "RaGNNarok_gnn"},
        {"base_mode": "icaraus_gnn", "model_label": "icaraus_gnn"},
        {"base_mode": "naive_integrator", "model_label": None},
        {"base_mode": "naive_radar", "model_label": None},
    ]
    
    for config in fold_configs:
        print(f"\n{'='*60}\nStarting Fold: {config}\n{'='*60}")
        for mode in eval_modes:
            print(f"\n--- Running Mode: {mode['base_mode']} (Model: {mode['model_label']}) ---")
            run_evaluation(
                base_eval_mode=mode["base_mode"],
                model_label=mode["model_label"],
                dataset_config_file=config,
                results_base_dir=results_base_dir
            )
