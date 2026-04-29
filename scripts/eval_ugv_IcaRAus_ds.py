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
from odometry.test_benches._test_bench import OdomCoordinateFrame, PredictionSource

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

def run_evaluation(eval_mode, dataset_config_file, datasets_path, maps_path, results_base_dir):
    """Orchestrates the evaluation of a specific mode on a given dataset fold.

    Args:
        eval_mode (str): The name of the evaluation configuration module (e.g., 'icaraus_gnn').
        dataset_config_file (str): Filename of the dataset configuration (e.g., 'IcaRAus_ugv_test_f1.yaml').
        datasets_path (str): Base path to the source datasets.
        maps_path (str): Base path to the map files.
        results_base_dir (str): Root directory to save evaluation results.
    """
    
    # 1. Load Dataset Config
    config_dir = os.path.join(os.path.dirname(__file__), "dataset_configs")
    dataset_config_path = os.path.join(config_dir, dataset_config_file)
    
    with open(dataset_config_path, "r") as f:
        ds_config = yaml.safe_load(f)
    
    config_label = ds_config["config_label"]
    datasets_to_test = ds_config["datasets_to_test"]
    
    # 2. Load Model Info (if applicable)
    model_info = None
    # We attempt to find a model_info file that matches the fold in the dataset_config name
    # e.g., if ds_config is 'IcaRAus_ugv_test_f1.yaml', look for 'icaraus_gnn_f1.yaml'
    if eval_mode in ["icaraus_gnn", "ragnnarok_gnn"]:
        # Extract fold indicator (e.g. f1)
        fold = None
        if "_f1" in dataset_config_file: fold = "f1"
        elif "_f2" in dataset_config_file: fold = "f2"
        elif "_f3" in dataset_config_file: fold = "f3"
        elif "_f4" in dataset_config_file: fold = "f4"
        
        if fold:
            model_info_file = f"{eval_mode}_{fold}.yaml"
            model_info_path = os.path.join(os.path.dirname(__file__), "eval_configs", "model_info", model_info_file)
            
            if os.path.exists(model_info_path):
                with open(model_info_path, "r") as f:
                    model_info_raw = yaml.safe_load(f)
                    # Resolve paths inside model_info
                    model_info = {
                        "model_config_path": resolve_path(model_info_raw.get("model_config_path")),
                        "model_state_dict_path": resolve_path(model_info_raw.get("model_state_dict_path"))
                    }
            else:
                print(f"Warning: model_info file not found at {model_info_path}")
    
    # 3. Load Evaluation Mode components via Factory
    sys.path.append(os.path.join(os.path.dirname(__file__), "eval_configs"))
    eval_module = importlib.import_module(eval_mode)
    
    results_parent_folder = os.path.join(results_base_dir, f"{config_label}_{eval_mode}_eval")
    create_dir(results_parent_folder)

    for folder_name in datasets_to_test.keys():
        map_file = datasets_to_test[folder_name]["map"]
        for file_name in datasets_to_test[folder_name]["datasets"]:
            print(f"Analyzing: {file_name} in {folder_name} using {eval_mode}")

            # Initialize Dataset
            if "RaGNNarok" in dataset_config_file:
                dataset = CpslDS(
                    dataset_path=os.path.join(datasets_path, folder_name, file_name),
                    radar_pc_folder="radar_combined",
                    lidar_folder="lidar",
                    camera_folder="camera",
                    imu_orientation_folder="imu_data",
                    imu_full_folder="imu_data_full",
                    vehicle_vel_folder="vehicle_vel"
                )
            else: # IcaRAus
                dataset = CpslDS(
                    dataset_path=os.path.join(datasets_path, folder_name, file_name),
                    radar_pc_folder="radar_combined_pc",
                    lidar_folder="lidar",
                    camera_folder="camera",
                    vehicle_odom_folder="vehicle_odom"
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
                model_info=model_info
            )

            # Start Heading/Pose logic
            if file_name == "north_1st_4":
                start_heading = np.deg2rad(45)
                start_pose = np.array([1.0, 0.5])
            else:
                start_heading = np.deg2rad(0)
                start_pose = np.array([0.00, 0.00])

            test_bench.init_localization(
                est_start_heading_rad=start_heading,
                est_start_pose_m=start_pose,
                show=False
            )

            # Run TB
            test_bench.run(
                max_frame=dataset.num_frames,
                gt_enabled=True,
                generate_dataset=False,
                normalize_frames=True
            )

            # Save results
            res_folder = os.path.join(results_parent_folder, "Results")
            create_dir(res_folder)
            test_bench.analyze(
                save_folder_path=res_folder,
                file_name=file_name,
                export_to_csv=True
            )

            # Save plots
            img_folder = os.path.join(results_parent_folder, "Images", "position_history")
            create_dir(img_folder)
            fig, axs = plt.subplots(figsize=(5, 5))
            test_bench.plotter_localization.plot_position_history_m(
                test_bench.history_position_m,
                test_bench.history_position_m_gt,
                history_position_m_inertial=test_bench.history_position_m_inertial,
                idx=dataset.num_frames - 1,
                ax=axs,
                show=False
            )
            fig.savefig(os.path.join(img_folder, f"{file_name}.png"))
            plt.close(fig)

    # Final Summary for this task
    analyzer = Analyzer()
    analyzer.show_cumulative_summary_from_csvs(
        save_folder=os.path.join(results_parent_folder, "Results")
    )

if __name__ == "__main__":
    
    # Define tasks to run
    # Format: (eval_mode, dataset_config_file)
    tasks = [
        # IcaRAus Evaluations (4 Folds)
        ("icaraus_gnn_ugv", "IcaRAus_ugv_test_f1.yaml"),
        ("icaraus_gnn_ugv", "IcaRAus_ugv_test_f2.yaml"),
        ("icaraus_gnn_ugv", "IcaRAus_ugv_test_f3.yaml"),
        ("icaraus_gnn_ugv", "IcaRAus_ugv_test_f4.yaml"),
        
        #TODO: make on all IcaRAus_ds folds
        # Baselines on IcaRAus Fold 1
        ("naive_radar", "IcaRAus_ugv_test_f1.yaml"),
        ("naive_integrator", "IcaRAus_ugv_test_f1.yaml"),
        
        #TODO: should be on the IcaRAus_ds_folds, not RaGNNarok ds folds (i.e. RaGNNarok model trained on IcaRAus folds)
        # RaGNNarok Evaluations (3 Folds)
        ("ragnnarok_gnn", "RaGNNarok_ugv_test_f1.yaml"),
        ("ragnnarok_gnn", "RaGNNarok_ugv_test_f2.yaml"),
        ("ragnnarok_gnn", "RaGNNarok_ugv_test_f3.yaml"),
        
        #TODO: make on all IcaRAus_ds_folds
        # Baselines on RaGNNarok Fold 1
        ("naive_radar", "RaGNNarok_ugv_test_f1.yaml"),
        ("naive_integrator", "RaGNNarok_ugv_test_f1.yaml"),
    ]
    
    # Base paths
    icaraus_datasets_path = "/data/IcaRAus/datasets/UGV"
    icaraus_maps_path = "/data/IcaRAus/maps"
    ragnnarok_datasets_path = "/data/RaGNNarok/ugv_datasets/"
    ragnnarok_maps_path = "/data/RaGNNarok/ugv_datasets/maps/"
    
    results_base_dir = os.path.join(repo_root, "scripts", "evaluation_results")

    for eval_mode, ds_config in tasks:
        print(f"\n{'='*60}\nStarting Task: {eval_mode} on {ds_config}\n{'='*60}")
        
        if "RaGNNarok" in ds_config:
            ds_path = ragnnarok_datasets_path
            m_path = ragnnarok_maps_path
        else:
            ds_path = icaraus_datasets_path
            m_path = icaraus_maps_path
            
        run_evaluation(eval_mode, ds_config, ds_path, m_path, results_base_dir)
