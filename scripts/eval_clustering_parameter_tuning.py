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
END_IDX = 437

# EPS and Min Samples combinations for tuning (8 combinations for 2x4 grid)
test_cases = [
    # (0.05, 5),  (0.05, 10), 
    (0.1, 5),   (0.1, 10), 
    (0.15, 7),  (0.15, 12), 
    (0.2, 7),   (0.2, 10),
    (0.25,7), (0.25,10),
    # (0.25,12), (0.25,15),
    # (0.3,7), (0.3,10),
    
]

# Dataset and Map configuration
DATASET_PATH = "/data/IcaRAus/datasets/UGV"
MAP_DIRECTORY = "/data/IcaRAus/maps"
folder_name = "WILK"
file_name = "IcaRAus_ugv_wilk_2_5m"

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
    map_file="wilk_map.yaml"
)

# Radar and Lidar localizers (parameters from baseline)
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

# Standard integrator setup
point_cloud_integrator = _PointCloudIntegrator(
    gt_distance_threshold_m=0.4,
    num_frames_history_gt=1,
    valid_fovs_deg=[(-70, 70), (110, -110)],
    num_frames_history=NUM_FRAMES_HISTORY,
    min_detection_radius=1.0,
    max_detection_radius=8.0,
    grid_resolution_m=0.05,
    gt_point_labeling_strategy=GtPointLabelingStrategy.USE_GT_POINTS_FOR_GT_CLASSIFICATION,
    occlusion_aware_clustering=OcclusionAwareClustering(
        clustering_eps=0.15,
        clustering_min_samples=7,
        angle_res_rad=0.017,
        occlusion_threshold=0.9,
        subsample_percentage=SUBSAMPLE_PERCENTAGE,
        remove_occluded=False
    )
)

dynamic_point_cloud_integrator = _PointCloudIntegrator(
    gt_distance_threshold_m=0.4,
    num_frames_history_gt=1,
    valid_fovs_deg=[(-70, 70), (110, -110)],
    num_frames_history=NUM_FRAMES_HISTORY,
    min_detection_radius=1.0,
    max_detection_radius=8.0,
    grid_resolution_m=0.05,
    gt_point_labeling_strategy=GtPointLabelingStrategy.USE_GT_POINTS_FOR_GT_CLASSIFICATION,
    occlusion_aware_clustering=None
)

test_bench = PointCloudIntegratorTB(
    localizer=radar_odometry,
    gt_localizer=lidar_odometry,
    map_handler=map_handler,
    dataset=dataset,
    point_cloud_integrator=point_cloud_integrator,
    dynamic_point_cloud_integrator=dynamic_point_cloud_integrator,
    use_filters=True,
    prediction_source=PredictionSource.VEHICLE_ODOM,
    gt_source=GroundTruthSource.LIDAR,
    odom_frame=OdomCoordinateFrame.FLU
)

# Initialize positions
test_bench.init_localization(est_start_heading_rad=0, est_start_pose_m=np.array([0, 0]), show=False)
test_bench.init_filter(
    est_start_heading_rad=0,
    est_start_position_m=np.array([0, 0]),
    start_time_s=test_bench.get_dataset_start_time(idx=0),
    gyro_bias=-0.0024
)

# --- Data Collection ---
print(f"Accumulating points over {END_IDX} frames...")
test_bench.run(start_frame=0, max_frame=END_IDX, gt_enabled=True)

# Extract final accumulated points in ego frame
radar_dets = test_bench.point_cloud_integrator.get_raw_point_history() # Nx4 [x,y,z,frame]
dynamic_dets = test_bench.dynamic_point_cloud_integrator.get_raw_point_history()
current_pose = test_bench.history_position_m_gt[END_IDX - 1]
current_heading = test_bench.history_heading_deg_gt[END_IDX - 1]

# --- Parameter Sweep & Visualization ---
fig, axs = plt.subplots(2, 4, figsize=(24, 12))
axs_flat = axs.flatten()

print("Iterating through clustering parameter combinations...")
for i, (eps, min_pts) in enumerate(test_cases):
    print(f"Testing combo {i+1}/8: eps={eps}, min_samples={min_pts}")
    
    # Create a fresh clusterer for each test case
    clusterer = OcclusionAwareClustering(
        clustering_eps=eps,
        clustering_min_samples=min_pts,
        angle_res_rad=0.017,
        occlusion_threshold=0.9,
        subsample_percentage=SUBSAMPLE_PERCENTAGE,
        filter_method="ray_trace",
        remove_occluded=True
    )
    
    # Perform clustering
    filtered_pts, labels, _ = clusterer.process(radar_dets)
    
    # Calculate number of clusters (excluding noise -1)
    num_clusters = len(np.unique(labels[labels != -1]))
    
    # Plot results on the specific subplot
    test_bench.plotter_localization.plot_detection_clusters_on_map(
        current_points=filtered_pts[:, 0:2],
        labels=labels,
        heading_rad=np.deg2rad(current_heading),
        pose_m=current_pose,
        ax=axs_flat[i],
        dynamic_points=dynamic_dets[:, 0:2],
        show=False
    )
    axs_flat[i].set_title(f"eps={eps}, min_pts={min_pts}, clusters={num_clusters}")

plt.suptitle(f"Clustering Hyperparameter Sweep (subsample={SUBSAMPLE_PERCENTAGE})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs("cluster_tuning", exist_ok=True)
plt.savefig("cluster_tuning/parameter_tuning_sweep.png")
plt.close()
