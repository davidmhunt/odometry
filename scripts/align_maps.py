import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.neighbors import NearestNeighbors

from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D import icp2D
from odometry.supportFns import rotation_functions

def parse_args():
    """Parse command line arguments for map alignment.

    Returns:
        argparse.Namespace: Parsed command line arguments including the configuration file name.
    """
    parser = argparse.ArgumentParser(description="Align radar and lidar maps using ICP and compute similarity metrics based on a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the configuration .yaml file in scripts/map_comparison_configs/"
    )
    return parser.parse_args()

def compute_nearest_neighbor_distances(source_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """Compute the distance from each point in the source cloud to its nearest neighbor in the reference cloud.

    Args:
        source_points (np.ndarray): Nx2 array of points to find neighbors for.
        reference_points (np.ndarray): Mx2 array of reference points.

    Returns:
        np.ndarray: Array of distances to the nearest neighbors in the reference cloud.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(reference_points)
    distances, _ = nn.kneighbors(source_points)
    return distances.flatten()

def main():
    """Main execution function to load configuration, align maps, and compute metrics."""
    args = parse_args()
    
    # Locate and load the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "map_comparison_configs", args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    lidar_map_path = config.get("lidar_map_yaml")
    radar_map_path = config.get("radar_map_yaml")
    initial_guess = config.get("initial_guess", {"x": 0.0, "y": 0.0, "heading_offset_degrees": 0.0})
    
    # Initialize MapHandlers and load points
    print(f"Loading lidar map: {lidar_map_path}")
    lidar_handler = MapHandler(os.path.dirname(lidar_map_path), os.path.basename(lidar_map_path))
    lidar_points = lidar_handler.map_points

    print(f"Loading radar map: {radar_map_path}")
    radar_handler = MapHandler(os.path.dirname(radar_map_path), os.path.basename(radar_map_path))
    radar_points = radar_handler.map_points

    if lidar_points is None or radar_points is None:
        print("Error: One or both map point clouds could not be loaded. Please check file paths in config.")
        return

    # Prepare initial guess
    init_pose = np.array([initial_guess.get("x", 0.0), initial_guess.get("y", 0.0)])
    init_heading_rad = np.radians(initial_guess.get("heading_offset_degrees", 0.0))

    # Perform ICP alignment
    print(f"Performing ICP alignment with initial guess: Pose={init_pose}, Heading={init_heading_rad:.4f} rad")
    icp_solver = icp2D()
    icp_solver.load_map_point_cloud(lidar_points)
    
    new_heading, new_pose = icp_solver.icp(
        current_points=radar_points,
        reference_points=lidar_points,
        estimated_heading_rad=init_heading_rad,
        estimated_pose_m=init_pose
    )
    
    if new_heading is None:
        print("ICP alignment failed. Proceeding with points at initial guess for metric calculation.")
        aligned_radar_points = rotation_functions.apply_rot_trans(
            points=radar_points,
            rot_angle_rad=init_heading_rad,
            trans=init_pose
        )
        heading_final = init_heading_rad
        pose_final = init_pose
    else:
        print(f"ICP Successful. Final Heading: {new_heading:.4f} rad, Final Pose: {new_pose}")
        aligned_radar_points = rotation_functions.apply_rot_trans(
            points=radar_points,
            rot_angle_rad=new_heading,
            trans=new_pose
        )
        heading_final = new_heading
        pose_final = new_pose

    # Compute similarity metrics
    radar_to_lidar_dist = compute_nearest_neighbor_distances(aligned_radar_points, lidar_points)
    lidar_to_radar_dist = compute_nearest_neighbor_distances(lidar_points, aligned_radar_points)

    # Chamfer: mean of (mean of radar -> lidar) and (mean of lidar -> radar)
    chamfer_dist = 0.5 * (np.mean(radar_to_lidar_dist) + np.mean(lidar_to_radar_dist))
    
    # Hausdorff: max of (max of radar -> lidar) and (max of lidar -> radar)
    hausdorff_dist = max(np.max(radar_to_lidar_dist), np.max(lidar_to_radar_dist))

    # Map scale context
    all_points = np.vstack([lidar_points, aligned_radar_points])
    map_length = np.max(all_points[:, 1]) - np.min(all_points[:, 1])
    map_width = np.max(all_points[:, 0]) - np.min(all_points[:, 0])

    print(f"\nAlignment Results:")
    print(f"Chamfer Distance: {chamfer_dist:.4f} m")
    print(f"Hausdorff Distance: {hausdorff_dist:.4f} m")
    print(f"Map Scale - Length: {map_length:.2f} m, Width: {map_width:.2f} m")

    # Save summary results
    config_name = os.path.splitext(args.config)[0]
    save_dir = os.path.join("map_comparison_results/map_comparison", config_name)
    os.makedirs(save_dir, exist_ok=True)
    
    summary_data = {
        "config": [args.config],
        "lidar_map": [lidar_map_path],
        "radar_map": [radar_map_path],
        "chamfer_distance_m": [chamfer_dist],
        "hausdorff_distance_m": [hausdorff_dist],
        "map_length_m": [map_length],
        "map_width_m": [map_width],
        "final_heading_rad": [heading_final],
        "final_pose_x_m": [pose_final[0]],
        "final_pose_y_m": [pose_final[1]]
    }
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(save_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Plotting
    plt.figure(figsize=(4, 4))
    plt.scatter(
        lidar_points[:, 0], 
        lidar_points[:, 1], 
        s=2, 
        c='blue', 
        alpha=0.6, 
        label='Lidar Map'
    )
    plt.scatter(
        aligned_radar_points[:, 0], 
        aligned_radar_points[:, 1], 
        s=2, 
        c='red', 
        alpha=0.6, 
        label='Radar Map (Aligned)'
    )
    
    plt.legend(markerscale=5, fontsize=10)
    plt.title(
        f"Aligned Maps: {config_name}\nChamfer: {chamfer_dist:.4f}m, Hausdorff: {hausdorff_dist:.4f}m",
        fontsize=12
    )
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "alignment_overlay.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Alignment plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
