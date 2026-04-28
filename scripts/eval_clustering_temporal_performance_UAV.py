import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure the local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environmental variables
load_dotenv()

# Odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB
from odometry.test_benches._test_bench import PredictionSource, GroundTruthSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy

# --- Configuration Parameters ---
SUBSAMPLE_PERCENTAGE = 0.20
NUM_FRAMES_HISTORY = 50
NUM_EVAL_POINTS = 10

# Fixed clustering parameters for temporal evaluation
# CLUSTERING_EPS = 0.25
# CLUSTERING_MIN_SAMPLES = 10

CLUSTERING_EPS = 0.35
CLUSTERING_MIN_SAMPLES = 10

# Dataset and Map configuration
DATASET_PATH = "/data/IcaRAus/datasets/UAV/Flow_datasets"
MAP_DIRECTORY = "/data/IcaRAus/maps"
folder_name = "vicon_diamond"
file_name = "vicon_diamond_1"

# --- Initialization ---
dataset = CpslDS(
    dataset_path=os.path.join(DATASET_PATH, folder_name, file_name),
    radar_pc_folder="radar_combined_pc",
    lidar_folder="lidar",
    camera_folder="camera",
    vehicle_odom_folder="vehicle_odom",
    vicon_folder="vicon_x500_8"
)

map_handler = MapHandler(
    maps_folder=MAP_DIRECTORY,
    map_file="north_vicon_1.yaml"
)

# Radar and Lidar localizers
radar_odometry = icp2DLocalization(
    icp_matching_distance_threshold=0.25,
    icp_best_points_percentile=85, #was 60
    icp_convergence_translation_threshold=1e-3,
    icp_convergence_rotation_threshold=1e-4,
    icp_point_pairs_threshold=7,
    icp_max_iterations=5,
    self_detection_radius_m=0
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

# Integrator setup
point_cloud_integrator = _PointCloudIntegrator(
    gt_distance_threshold_m=0.4,
    num_frames_history_gt=1,
    valid_fovs_deg=[(-70, 70), (110, -110)],
    num_frames_history=NUM_FRAMES_HISTORY,
    min_detection_radius=1.0,
    max_detection_radius=8.0,
    grid_resolution_m=0.05,
    gt_point_labeling_strategy=GtPointLabelingStrategy.USE_GT_POINTS_FOR_GT_CLASSIFICATION,
    # occlusion_aware_clustering=OcclusionAwareClustering(
    #     clustering_eps=CLUSTERING_EPS,
    #     clustering_min_samples=CLUSTERING_MIN_SAMPLES,
    #     angle_res_rad=0.017,
    #     occlusion_threshold=0.9,
    #     subsample_percentage=SUBSAMPLE_PERCENTAGE,
    #     remove_occluded=False
    # )
)

test_bench = PointCloudIntegratorTB(
    localizer=radar_odometry,
    gt_localizer=lidar_odometry,
    map_handler=map_handler,
    dataset=dataset,
    point_cloud_integrator=point_cloud_integrator,
    use_filters=True,
    prediction_source=PredictionSource.VEHICLE_ODOM,
    gt_source=GroundTruthSource.MOTION_CAPTURE,
    odom_frame=OdomCoordinateFrame.NED
)

# Initialize positions
test_bench.init_localization(est_start_heading_rad=0, est_start_pose_m=np.array([0, 0]), show=False)
test_bench.init_filter(
    est_start_heading_rad=0,
    est_start_position_m=np.array([0, 0]),
    start_time_s=test_bench.get_dataset_start_time(idx=0),
    gyro_bias=-0.0024
)

# Calculate target frames (evenly spaced)
# We start from NUM_FRAMES_HISTORY to ensure we have a full buffer for the first eval point
target_frames = np.linspace(NUM_FRAMES_HISTORY, 0.75 * dataset.num_frames - 1, NUM_EVAL_POINTS, dtype=int)

# --- Sequential Evaluation ---
fig, axs = plt.subplots(2, 5, figsize=(30, 12))
axs_flat = axs.flatten()

# Add second figure for centroids
fig_centroids, axs_centroids = plt.subplots(2, 5, figsize=(30, 12))
axs_centroids_flat = axs_centroids.flatten()

clusterer = OcclusionAwareClustering(
    clustering_eps=CLUSTERING_EPS,
    clustering_min_samples=CLUSTERING_MIN_SAMPLES,
    angle_res_rad=0.017,
    occlusion_threshold=0.9,
    subsample_percentage=SUBSAMPLE_PERCENTAGE,
    filter_method="ray_trace",
    remove_occluded=True
)

current_frame = 0
for i, target_frame in enumerate(target_frames):
    print(f"\nProcessing segment {i+1}/{NUM_EVAL_POINTS}: Frame {current_frame} to {target_frame}...")
    
    # Run the test bench up to the target frame
    test_bench.run(start_frame=current_frame, max_frame=target_frame, gt_enabled=True)
    
    # Update current frame for next segment
    current_frame = target_frame
    
    # Extract accumulated points at this checkpoint
    radar_dets = test_bench.point_cloud_integrator.get_raw_point_history()
    current_pose = test_bench.history_position_m_gt[target_frame - 1]
    current_heading = test_bench.history_heading_deg_gt[target_frame - 1]
    
    # Perform clustering
    filtered_pts, labels, _ = clusterer.process(radar_dets)
    
    # Calculate centroids
    unique_labels = np.unique(labels)
    centroids = []
    for l in unique_labels:
        if l != -1:
            centroids.append(np.mean(filtered_pts[labels == l, 0:2], axis=0))
    centroids = np.array(centroids) if len(centroids) > 0 else np.empty((0, 2))

    # Plot results (clusters)
    test_bench.plotter_localization.plot_detection_clusters_on_map(
        current_points=filtered_pts[:, 0:2],
        labels=labels,
        heading_rad=np.deg2rad(current_heading),
        pose_m=current_pose,
        ax=axs_flat[i],
        show=False
    )
    axs_flat[i].set_title(f"Checkpoint {i+1}: Frame {target_frame}")

    # Plot results (centroids)
    original_marker_size = test_bench.plotter_localization.marker_size
    test_bench.plotter_localization.marker_size = 3.0
    test_bench.plotter_localization.plot_detections_on_map(
        current_points=centroids,
        heading_rad=np.deg2rad(current_heading),
        pose_m=current_pose,
        ax=axs_centroids_flat[i],
        show=False
    )
    test_bench.plotter_localization.marker_size = original_marker_size
    axs_centroids_flat[i].set_title(f"Centroids Checkpoint {i+1}: Frame {target_frame}")

fig.suptitle(f"Temporal Clustering Consistency (subsample={SUBSAMPLE_PERCENTAGE}, eps={CLUSTERING_EPS}, min={CLUSTERING_MIN_SAMPLES})", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs("cluster_tuning", exist_ok=True)
fig.savefig("cluster_tuning/temporal_performance_grid_uav.png")

# Save centroids figure
fig_centroids.suptitle(f"Temporal Cluster Centroids (subsample={SUBSAMPLE_PERCENTAGE}, eps={CLUSTERING_EPS}, min={CLUSTERING_MIN_SAMPLES})", fontsize=16)
fig_centroids.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_centroids.savefig("cluster_tuning/temporal_centroids_grid_uav.png")

plt.close('all')
