import os
import sys
import yaml
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the repository root and eval_configs to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

eval_configs_dir = os.path.join(repo_root, "scripts", "eval_configs")
if eval_configs_dir not in sys.path:
    sys.path.append(eval_configs_dir)

from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler

def resolve_path(path):
    """Resolves a path relative to the repository root.

    Args:
        path (str): The path to resolve.

    Returns:
        str: The absolute path.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)

def load_eval_module(file_path):
    """Loads a python module from a file path.

    Args:
        file_path (str): Path to the python file.

    Returns:
        module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location("eval_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def generate_pc_figures(config_path):
    """Generates point cloud comparison figures based on a YAML config.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_label = config["config_label"]
    ds_cfg = config["dataset"]
    frame_indices = config["frame_indices"]
    eval_modes = config["eval_modes"]

    # Setup output directory
    output_base_dir = os.path.join(repo_root, "scripts", "results", "pc_comparison", config_label, ds_cfg["name"])
    os.makedirs(output_base_dir, exist_ok=True)

    # Initialize Dataset
    dataset = CpslDS(
        dataset_path=os.path.join(ds_cfg["dataset_path"], ds_cfg.get("folder_name", ds_cfg.get("group_name", "")), ds_cfg["name"]),
        radar_pc_folder=ds_cfg.get("radar_pc_folder", "radar_combined_pc"),
        lidar_folder=ds_cfg.get("lidar_folder", "lidar"),
        camera_folder=ds_cfg.get("camera_folder", "camera"),
        vehicle_odom_folder=ds_cfg.get("vehicle_odom_folder", "vehicle_odom"),
        vicon_folder=ds_cfg.get("vicon_folder", None),
        imu_orientation_folder=ds_cfg.get("imu_orientation_folder", "imu_data"),
        imu_full_folder=ds_cfg.get("imu_full_folder", "imu_data_full"),
        vehicle_vel_folder=ds_cfg.get("vehicle_vel_folder", "vehicle_vel")
    )

    # Initialize Map
    map_handler = MapHandler(
        maps_folder=ds_cfg["maps_path"],
        map_file=ds_cfg["map"]
    )

    trajectories = {}
    gt_trajectory = None

    for mode in eval_modes:
        eval_script_name = mode["eval_config"]
        model_info_file = mode["model_info"]

        print(f"\nEvaluating: {eval_script_name}")

        # Load Eval Module
        eval_script_path = os.path.join(eval_configs_dir, eval_script_name)
        eval_module = load_eval_module(eval_script_path)

        # Load Model Info
        model_info = None
        if model_info_file:
            model_info_path = os.path.join(eval_configs_dir, "model_info", model_info_file)
            with open(model_info_path, "r") as f:
                model_info_raw = yaml.safe_load(f)
            model_info = {
                "model_config_path": resolve_path(model_info_raw.get("model_config_path")),
                "model_state_dict_path": resolve_path(model_info_raw.get("model_state_dict_path"))
            }

        # Initialize Test Bench
        test_bench = eval_module.get_test_bench(
            dataset=dataset,
            map_handler=map_handler,
            model_info=model_info,
            num_frames_history=50,
            normalize_frames=True
        )

        # Initial Pose (Simplified logic from eval scripts)
        start_heading = np.deg2rad(0)
        start_pose = np.array([0.00, 0.00])
        
        # Check if it's the specific dataset mentioned in original scripts
        if ds_cfg["name"] == "north_1st_4":
            start_heading = np.deg2rad(45)
            start_pose = np.array([1.0, 0.5])

        # Initialize localization with optional gyro_bias
        init_kwargs = {
            "est_start_heading_rad": start_heading,
            "est_start_pose_m": start_pose,
            "show": False
        }
        
        gyro_bias = ds_cfg.get("gyro_bias")
        if gyro_bias is not None:
            init_kwargs["gyro_bias"] = gyro_bias
            
        test_bench.init_localization(**init_kwargs)

        # Setup Figure for 5 Point Clouds
        fig, axes = plt.subplots(1, len(frame_indices), figsize=(3 * len(frame_indices), 3))
        if len(frame_indices) == 1:
            axes = [axes]

        current_frame = 0
        for i, target_idx in enumerate(frame_indices):
            print(f"Running to frame {target_idx}...")
            test_bench.run(
                start_frame=current_frame,
                max_frame=target_idx + 1,
                gt_enabled=True,
                generate_dataset=False
            )
            current_frame = target_idx + 1

            # Get last PC and pose
            if len(test_bench.history_pc_processor_point_cloud) > 0:
                pc = test_bench.history_pc_processor_point_cloud[-1]
                pos = test_bench.history_pc_processor_position_m[-1]
                heading = test_bench.history_pc_processor_heading_rad[-1]
                
                test_bench.plotter_localization.plot_detections_on_map(
                    current_points=pc,
                    heading_rad=heading,
                    pose_m=pos,
                    ax=axes[i],
                    show=False
                )
            axes[i].set_title(f"Frame {target_idx}", fontsize=10)
            axes[i].tick_params(axis='both', which='major', labelsize=8)

        # Save PC comparison figure
        identifier = os.path.splitext(eval_script_name)[0]
        pc_fig_path = os.path.join(output_base_dir, f"{identifier}_pc_comparison.png")
        fig.tight_layout()
        fig.savefig(pc_fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved PC comparison to {pc_fig_path}")

        # Finish running the dataset to get full trajectory if not already finished
        if current_frame < dataset.num_frames:
            print(f"Finishing remaining {dataset.num_frames - current_frame} frames for trajectory...")
            test_bench.run(
                start_frame=current_frame,
                max_frame=dataset.num_frames,
                gt_enabled=True,
                generate_dataset=False
            )
        
        trajectories[identifier] = test_bench.history_position_m.copy()
        if gt_trajectory is None:
            gt_trajectory = test_bench.history_position_m_gt.copy()

    # Final Trajectory Comparison
    print("\nGenerating trajectory comparison...")
    fig_traj, ax_traj = plt.subplots(figsize=(5, 5))
    
    # Plot Map
    map_points = map_handler.map_points
    ax_traj.scatter(map_points[:, 0], map_points[:, 1], s=0.5, color='blue', alpha=0.3, label='Map')

    # Plot GT
    if gt_trajectory is not None:
        ax_traj.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'k--', label='Ground Truth', linewidth=1.5)

    # Plot all methods
    for label, traj in trajectories.items():
        ax_traj.plot(traj[:, 0], traj[:, 1], label=label, linewidth=1)

    ax_traj.set_title(f"Trajectory Comparison: {ds_cfg['name']}", fontsize=12)
    ax_traj.set_xlabel("X (m)", fontsize=10)
    ax_traj.set_ylabel("Y (m)", fontsize=10)
    ax_traj.legend(fontsize=8, loc='best')
    ax_traj.axis('equal')
    ax_traj.grid(True, linestyle='--', alpha=0.7)
    ax_traj.tick_params(axis='both', which='major', labelsize=8)

    fig_traj.tight_layout()
    traj_fig_path = os.path.join(output_base_dir, "trajectory_comparison.png")
    fig_traj.savefig(traj_fig_path, dpi=300)
    plt.close(fig_traj)
    print(f"Saved trajectory comparison to {traj_fig_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate point cloud and trajectory comparison figures.")
    parser.add_argument("--config", type=str, required=True, help="Name of the config file in scripts/pc_comparison_configs/")
    args = parser.parse_args()

    config_file = args.config
    if not config_file.endswith(".yaml"):
        config_file += ".yaml"
    
    config_path = os.path.join(repo_root, "scripts", "pc_comparison_configs", config_file)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    generate_pc_figures(config_path)
