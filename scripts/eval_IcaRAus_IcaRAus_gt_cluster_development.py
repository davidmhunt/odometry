
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#load the necessary odometry modules
from cpsl_datasets.cpsl_ds import CpslDS
from cpsl_datasets.map_handler import MapHandler
from odometry.localization.icp2D_localization import icp2DLocalization
from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB
from odometry.test_benches._test_bench import PredictionSource, GroundTruthSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.TwoStreamSpatioTemporalGnn import TwoStreamSpatioTemporalGnn
from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering
from odometry.point_cloud_processing.pc_fov_filter import pcFovFilter
from odometry.point_cloud_processing.accumulation.pc_accumulator import GtPointLabelingStrategy
from scipy.spatial import cKDTree




from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder

from dotenv import load_dotenv
import os


def get_gt_pc(idx:int)-> np.ndarray:

    pc_range_filter = pcRangeFilter(
        min_detection_radius_m=0.5,
        max_detection_radius_m=test_bench.point_cloud_integrator.max_detection_radius
    )

    gt_points = test_bench.dataset.get_lidar_point_cloud_raw(idx)

    gt_points = pc_range_filter.get_points_in_detection_range(
        points=gt_points
    )
    
    #filter out ground, set z coordinate to 0 for remaining points
    valid_points = gt_points[:,2] > -0.1 #filter out ground
    valid_points = valid_points & (gt_points[:,2] < 0.5) #higher elevation points
    gt_points = gt_points[valid_points,:3]
    gt_points[:,2] = 0.0

    return gt_points

#loading enviroment variables
load_dotenv()

DATASET_PATH = "/data/IcaRAus/datasets/UGV"
MAP_DIRECTORY = "/data/IcaRAus/maps"
GENERATED_DATASETS_PATH = "/data/IcaRAus/generated_datasets"
generate_movie = False

# #setup the datasets
folder_name = "WILK" #"WILK", "NORTH_VICON", "NORTH_1ST", "CPSL"
file_name = "IcaRAus_ugv_wilk_2_5m" # "IcaRAus_ugv_wilk_2_5m", "north_vicon_1", "north_1st_1", "IcaRAus_ugv_cpsl_1_5m"


normalize_frames = True
num_frames_history = 50
config_label = "IcaRAus_gnn_{}fh_grids".format(num_frames_history)
results_parent_folder = "{}_train".format(config_label)
print(os.path.join(DATASET_PATH,folder_name,file_name))
dataset = CpslDS(
    dataset_path=os.path.join(DATASET_PATH,folder_name,file_name),
    radar_pc_folder="radar_combined_pc",
    lidar_folder="lidar",
    camera_folder="camera",
    vehicle_odom_folder="vehicle_odom",
    vicon_folder="vicon_x500_8"
)
print(dataset.num_frames)

map_handler = MapHandler(
    maps_folder=MAP_DIRECTORY,
    map_file="wilk_map.yaml"#"cpsl_map.yaml", "wilk_map", "north_vicon_1", "north_1st_map"
)

#initialize the localizers
radar_odometry = icp2DLocalization(
    icp_matching_distance_threshold=0.25,#0.1
    icp_best_points_percentile=60, #80
    icp_convergence_translation_threshold=1e-3,
    icp_convergence_rotation_threshold=1e-4,
    icp_point_pairs_threshold=7, #7
    icp_max_iterations=5, #20
    self_detection_radius_m=0 #originally 1.5
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
#initialize the dataset encoders
input_encoder = _NodeEncoder()
gt_encoder = _GTNodeEncoder()

#initialize the dataset generator
generated_dataset_path = os.path.join(GENERATED_DATASETS_PATH,"{}_train".format(config_label))



#if testing the gnn runner
model = TwoStreamSpatioTemporalGnn(
    hidden_channels=28,
    out_channels=1,
    k=4, #original is 40
    dropout=0.1
)

runner = GNNRunner(
    model= model,
    state_dict_path = "/home/david/Downloads/IcaRAus_TwoStreamSpatioTemporalGnn_IcaRAus_ds_50fh_k4.pth",
    cuda_device="cpu",
    edge_radius=10.0, #unused for this model
    enable_downsampling=True,
    downsample_keep_ratio=0.50,
    downsample_min_points=300,
    use_sigmoid=True
)

point_cloud_integrator = _PointCloudIntegrator(
        gt_distance_threshold_m=0.4,
        num_frames_history_gt=1,
        valid_fovs_deg=[(-70,70),(110,-110)],
        num_frames_history=num_frames_history,
        min_detection_radius=1.0,
        max_detection_radius=8.0,
        grid_resolution_m=0.05,
        gt_point_labeling_strategy=GtPointLabelingStrategy.USE_GT_POINTS_FOR_GT_CLASSIFICATION,
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
            clustering_eps=0.15,
            clustering_min_samples=7,
            angle_res_rad=0.017,
            occlusion_threshold=0.9,
            subsample_percentage=0.40,
            remove_occluded=False,
            filter_method='overlap' #ray_trace or overlap
        )
    )



#dynamic point cloud integrator
dynamic_point_cloud_integrator = _PointCloudIntegrator(
    gt_distance_threshold_m=0.5,
    num_frames_history=num_frames_history,
    min_detection_radius=1.0,
    max_detection_radius=4.0
)

test_bench = PointCloudIntegratorTB(
    localizer=radar_odometry,
    gt_localizer=lidar_odometry,
    map_handler=map_handler,
    dataset=dataset,
    point_cloud_integrator=point_cloud_integrator,
    dynamic_point_cloud_integrator=dynamic_point_cloud_integrator,
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
    gyro_bias=-0.0024 #originally -0.0024,
)

end_idx = 437 #for full dataset, use "dataset.num_frames" #point cloud sample: frame 1340 on Wilk_Multipath_test_5 (960 for pc compare, 980 for block diagram)
save_name = config_label + "radar_combined"
test_bench.run(
    start_frame=0,
    max_frame=end_idx,
    gt_enabled=True, #if not desired set to False
    movie_generator=None,
    generate_dataset=False) #use generator script in scripts folder instead


# Get original data
radar_dets, labels_original = test_bench.point_cloud_integrator.get_nodes(normalize_frames=True) 
print(f"Original radar_dets shape: {radar_dets.shape}")

# Get current position and heading
current_pose = test_bench.history_position_m_gt[end_idx-1]
print(f"Current pose: {current_pose}")
current_heading = test_bench.history_heading_deg_gt[end_idx-1]
print(f"Current heading: {current_heading}")

fig, axs = plt.subplots(2, 4, figsize=(24, 12))

test_bench.plotter_localization.marker_size = 0.5
test_bench.plotter_localization.plot_detections_on_map(
    current_points=radar_dets[:,0:2],
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[0, 0],
    show=False
)

fov_filter = pcFovFilter(
    valid_fovs_deg = [(-70,70),(110,-110)]
)


filtered_pts = fov_filter.get_points_in_fov(radar_dets)

pc_grid = test_bench.point_cloud_integrator.raw_point_history._get_density_grid_from_points(
    points=radar_dets
)

test_bench.plotter_pc_grid.plot_pc_grid(
    grid=pc_grid,
    grid_bins=test_bench.point_cloud_integrator.raw_point_history.grid_bins,
    ax=axs[0, 1],
    show=False
)



# Pass results to your plotting function
test_bench.plotter_localization.plot_detections_on_map(
    current_points=filtered_pts[:,0:2],
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[1, 0],
    show=False
)



# --- Apply KNN Filtering ---
tuning_clustering_class = OcclusionAwareClustering(
    clustering_eps=0.25,
    clustering_min_samples=10,
    angle_res_rad=0.017,
    occlusion_threshold=0.9,
    subsample_percentage=0.2,
    filter_method="ray_trace",
    remove_occluded=True
)

# Fit NearestNeighbors on the x,y coordinates
for i in tqdm(range(10)):
    
    filtered_radar_dets, cluster_labels, vis_ids_radar = tuning_clustering_class.process(radar_dets[:,0:4])


test_bench.plotter_localization.plot_detection_clusters_on_map(
    current_points=filtered_radar_dets[:,0:2],
    labels=cluster_labels,
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[1, 1],
    show=False
)

plt.tight_layout()

# --- Post-Clustering Analysis ---
unique_labels = np.unique(cluster_labels)
centroids = []
if len(unique_labels) > 0:
    for l in unique_labels:
        if l != -1:
            centroids.append(np.mean(filtered_radar_dets[cluster_labels == l, 0:3], axis=0))
centroids = np.array(centroids) if len(centroids) > 0 else np.empty((0, 3))

# Classify centroids by proximity to map points in sensor frame
map_points = test_bench.map_handler.map_points
map_in_sensor = test_bench.plotter_localization._apply_inverse_transform(
    points=map_points,
    rot_angle_rad=np.deg2rad(current_heading),
    trans=current_pose
)

# Ensure points are 3D for the proximity function
def ensure_3d(pts):
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] == 2:
        return np.hstack((pts, np.zeros((pts.shape[0], 1))))
    return pts[:,0:3]

if len(centroids) > 0:
    # Use integrator method to find valid centroids
    valid_centroids_pts = test_bench.point_cloud_integrator.raw_point_history._get_dets_close_to_gt_points(
        dets=ensure_3d(centroids),
        gt_points=ensure_3d(map_in_sensor),
        threshold=0.5
    )

    # Create mask for valid centroids
    if len(valid_centroids_pts) > 0:
        tree = cKDTree(valid_centroids_pts)
        dists, _ = tree.query(centroids[:, 0:3], distance_upper_bound=1e-5)
        is_valid = dists <= 1e-5
    else:
        is_valid = np.zeros(len(centroids), dtype=bool)
else:
    is_valid = np.array([], dtype=bool)

# Plot Centroids on Global Map
original_marker_size = test_bench.plotter_localization.marker_size
test_bench.plotter_localization.marker_size = 3.0
test_bench.plotter_localization.plot_detections_on_map(
    current_points=centroids[:, 0:2] if len(centroids) > 0 else np.empty((0,2)),
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[0, 2],
    show=False
)
test_bench.plotter_localization.marker_size = original_marker_size
axs[0, 2].set_title("Cluster Centroids (Global)")

# Separate valid and invalid cluster labels
labels_no_noise = np.array([l for l in unique_labels if l != -1])
valid_labels = labels_no_noise[is_valid]
invalid_labels = labels_no_noise[~is_valid]

# Plot Valid Clusters (Sensor Frame)
valid_mask = np.isin(cluster_labels, valid_labels)
test_bench.plotter_localization.plot_map_on_detection_clusters(
    current_points=filtered_radar_dets[valid_mask, 0:2],
    labels=cluster_labels[valid_mask],
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[1, 2],
    show=False
)
axs[1, 2].set_title("Valid Clusters (Sensor)")

# Plot Invalid Clusters (Sensor Frame)
invalid_mask = np.isin(cluster_labels, invalid_labels)
test_bench.plotter_localization.plot_map_on_detection_clusters(
    current_points=filtered_radar_dets[invalid_mask, 0:2],
    labels=cluster_labels[invalid_mask],
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[1, 3],
    show=False
)
axs[1, 3].set_title("Invalid Clusters (Sensor)")

# Plot Map in Sensor Field of View
test_bench.plotter_localization.plot_map_on_detections(
    current_points=radar_dets[:, 0:2],
    heading_rad=np.deg2rad(current_heading),
    pose_m=current_pose,
    ax=axs[0, 3],
    show=False
)
axs[0, 3].set_title("Map in Sensor FOV")

os.makedirs("cluster_tuning", exist_ok=True)
plt.savefig("cluster_tuning/cluster_development_grid.png")
plt.close()

# --- Statistical Analysis of Clusters ---

# Initialize statistics dictionary
stats = {
    'Point Count': [],
    'Spatial Spread (RMS)': [],
    'Bounding Box Area': [],
    'Spatial Density': [],
    'PCA Linearity': [],
    'Temporal Duration': [],
    'Temporal Variance': [],
    'Mean Frame': [],
    'Is Valid': []
}

for l in labels_no_noise:
    c_pts = filtered_radar_dets[cluster_labels == l]
    
    # Valid flag
    stats['Is Valid'].append(l in valid_labels)
    
    # 1. Point Count
    pt_count = len(c_pts)
    stats['Point Count'].append(pt_count)
    
    # Spatial computations (XY only)
    c_xy = c_pts[:, 0:2]
    centroid = np.mean(c_xy, axis=0)
    
    # 2. Spatial Spread (RMS)
    rms_spread = np.sqrt(np.mean(np.sum((c_xy - centroid)**2, axis=1)))
    stats['Spatial Spread (RMS)'].append(rms_spread)
    
    # 3. Bounding Box Area
    min_xy = np.min(c_xy, axis=0)
    max_xy = np.max(c_xy, axis=0)
    bb_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
    stats['Bounding Box Area'].append(bb_area)
    
    # 4. Spatial Density
    density = pt_count / (bb_area + 1e-6)
    stats['Spatial Density'].append(density)
    
    # 5. PCA Linearity
    if pt_count > 1:
        cov_mat = np.cov(c_xy, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov_mat)
        # Sort eigenvalues descending
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        linearity = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-6)
        stats['PCA Linearity'].append(linearity)
    else:
        stats['PCA Linearity'].append(0.0)
        
    # Temporal computations (assuming normalized frame index is in column 3)
    frames = c_pts[:, 3]
    
    # 6. Temporal Duration
    duration = np.max(frames) - np.min(frames)
    stats['Temporal Duration'].append(duration)
    
    # 7. Temporal Variance
    if pt_count > 1:
        temp_var = np.var(frames)
        stats['Temporal Variance'].append(temp_var)
    else:
        stats['Temporal Variance'].append(0.0)
        
    # 8. Mean Frame
    stats['Mean Frame'].append(np.mean(frames))

# Convert lists to NumPy arrays
for k in stats:
    stats[k] = np.array(stats[k])

valid_mask_stats = stats['Is Valid']
invalid_mask_stats = ~stats['Is Valid']

# Plot the statistics
stats_to_plot = [
    'Point Count', 'Spatial Spread (RMS)', 'Bounding Box Area', 'Spatial Density',
    'PCA Linearity', 'Temporal Duration', 'Temporal Variance', 'Mean Frame'
]

fig2, axs2 = plt.subplots(2, 4, figsize=(24, 12))
axs2 = axs2.flatten()

for i, stat_name in enumerate(stats_to_plot):
    valid_vals = stats[stat_name][valid_mask_stats]
    invalid_vals = stats[stat_name][invalid_mask_stats]
    
    # Scatter points with jitter for visibility
    if len(valid_vals) > 0:
        axs2[i].scatter(np.random.normal(1, 0.05, size=len(valid_vals)), valid_vals, alpha=0.6, color='green', label='Valid')
    if len(invalid_vals) > 0:
        axs2[i].scatter(np.random.normal(2, 0.05, size=len(invalid_vals)), invalid_vals, alpha=0.6, color='red', label='Invalid')
        
    axs2[i].set_xticks([1, 2])
    axs2[i].set_xticklabels(['Valid', 'Invalid'])
    axs2[i].set_title(stat_name)
    axs2[i].grid(True, linestyle='--', alpha=0.7)

axs2[0].legend()
plt.suptitle("Statistical Differences Between Valid and Invalid Clusters", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cluster_tuning/cluster_development_stats.png")
plt.close()