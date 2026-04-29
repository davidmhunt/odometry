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
from odometry.plotting.movies import MovieGenerator
from odometry.analyzers.analyzer import Analyzer

# loading environment variables
load_dotenv()

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Utility to resolve paths relative to the repository root
def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def run_evaluation(base_eval_mode, model_label, dataset_config_file, results_base_dir):
    """Orchestrates the evaluation of a specific mode on a given dataset fold.

    Args:
        base_eval_mode (str): The name of the evaluation mode (e.g., 'icaraus_gnn', 'ragnnarok_gnn').
        model_label (str): The label for model_info lookup (e.g., 'icaraus_gnn', 'RaGNNarok_gnn').
        dataset_config_file (str): Filename of the dataset configuration (e.g., 'RaGNNarok_ugv_test_f1.yaml').
        results_base_dir (str): Root directory to save evaluation results.
    """
    
    # 1. Load Dataset Config
    config_dir = os.path.join(os.path.dirname(__file__), "dataset_configs")
    dataset_config_path = os.path.join(config_dir, dataset_config_file)
    
    with open(dataset_config_path, "r") as f:
        ds_config = yaml.safe_load(f)
    
    config_label = ds_config["config_label"]
    datasets_to_test = ds_config["datasets_to_test"]
    
    # Extract fold indicator (e.g. f1)
    fold = None
    for f_idx in range(1, 4): # RaGNNarok has 3 folds
        if f"_f{f_idx}" in dataset_config_file:
            fold = f"f{f_idx}"
            break
    
    # 2. Iterate through dataset groups
    for group_name in datasets_to_test.keys():
        group_config = datasets_to_test[group_name]
        platform = group_config.get("platform", "ugv")
        datasets_path = group_config["dataset_path"]
        
        # Override MAP_DIRECTORY for RaGNNarok evaluation
        maps_path = group_config.get("maps_path", ds_config.get("map_directory", "/data/RaGNNarok/ugv_datasets/maps/"))
        map_file = group_config["map"]
        
        # Determine Factory Module Name
        if base_eval_mode == "icaraus_gnn":
            eval_module_name = f"icaraus_gnn_ragnnarok_{platform}"
        elif base_eval_mode == "ragnnarok_gnn":
            eval_module_name = f"ragnnarok_gnn_{platform}"
        elif base_eval_mode == "naive_integrator":
            eval_module_name = f"naive_integrator_ragnnarok_{platform}"
        else:
            eval_module_name = f"{base_eval_mode}_{platform}"
            
        # 3. Load Model Info (if applicable)
        model_info = None
        if base_eval_mode in ["icaraus_gnn", "ragnnarok_gnn"]:
            if fold:
                # Use model_label for lookup to match standardized naming
                model_info_file = f"{model_label}_{'RaGNNarok_ds' if 'RaGNNarok' in config_label else 'IcaRAus_ds'}_{fold}.yaml"
                model_info_path = os.path.join(os.path.dirname(__file__), "eval_configs", "model_info", model_info_file)
                
                if os.path.exists(model_info_path):
                    with open(model_info_path, "r") as f:
                        model_info_raw = yaml.safe_load(f)
                    model_info = {
                        "model_config_path": resolve_path(model_info_raw.get("model_config_path")),
                        "model_state_dict_path": resolve_path(model_info_raw.get("model_state_dict_path"))
                    }
                else:
                    print(f"Warning: model_info file not found at {model_info_path}")
        
        # 4. Load Evaluation Factory
        sys.path.append(os.path.join(os.path.dirname(__file__), "eval_configs"))
        try:
            eval_module = importlib.import_module(eval_module_name)
        except ImportError:
            print(f"Error: Could not import evaluation module {eval_module_name}")
            continue
            
        results_folder_label = f"{config_label}_{base_eval_mode}"
        if model_label:
            results_folder_label += f"_{model_label}"
            
        results_parent_folder = os.path.join(results_base_dir, f"{results_folder_label}_eval")
        create_dir(results_parent_folder)

        for file_name in group_config["datasets"]:
            print(f"\nAnalyzing: {file_name} in {group_name} ({platform}) using {eval_module_name}")

            # Initialize Dataset
            folder_name = group_config.get("folder_name", group_name)
            dataset = CpslDS(
                dataset_path=os.path.join(datasets_path, folder_name, file_name),
                radar_pc_folder=group_config.get("radar_pc_folder", "radar_combined"),
                lidar_folder=group_config.get("lidar_folder", "lidar"),
                camera_folder=group_config.get("camera_folder", "camera"),
                vehicle_odom_folder=group_config.get("vehicle_odom_folder", "vehicle_odom"),
                vicon_folder=group_config.get("vicon_folder", None),
                imu_orientation_folder=group_config.get("imu_orientation_folder", "imu_data"),
                imu_full_folder=group_config.get("imu_full_folder", "imu_data_full"),
                vehicle_vel_folder=group_config.get("vehicle_vel_folder", "vehicle_vel")
            )

            # Initialize Map
            map_handler = MapHandler(
                maps_folder=maps_path,
                map_file=map_file
            )

            # Use Factory to get fully initialized Test Bench
            test_bench = eval_module.get_test_bench(
                dataset=dataset,
                map_handler=map_handler,
                model_info=model_info,
                num_frames_history=50,
                normalize_frames=True
            )

            # Initialize localization (RaGNNarok specific gyro bias)
            test_bench.init_localization(
                est_start_heading_rad=np.deg2rad(0),
                est_start_pose_m=np.array([0.00, 0.00]),
                show=False,
                gyro_bias=-0.0024
            )

            # Run TB
            test_bench.run(
                max_frame=dataset.num_frames,
                gt_enabled=True,
                generate_dataset=False
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
            
            # Check if history_position_m_inertial exists (RaGNNarok TBs typically have it)
            if hasattr(test_bench, 'history_position_m_inertial'):
                test_bench.plotter_localization.plot_position_history_m(
                    test_bench.history_position_m,
                    test_bench.history_position_m_gt,
                    test_bench.history_position_m_inertial,
                    idx=dataset.num_frames - 1,
                    ax=axs,
                    show=False
                )
            else:
                test_bench.plotter_localization.plot_position_history_m(
                    test_bench.history_position_m,
                    test_bench.history_position_m_gt,
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
