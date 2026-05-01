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
from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB
from odometry.test_benches._test_bench import _TestBench, PredictionSource, GroundTruthSource, OdomCoordinateFrame
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator import _PointCloudIntegrator
from odometry.point_cloud_processing.accumulation.integrators._pc_integrator_gnn_runner import _PointCloudIntegratorGnnRunner
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.torch_training.models.TwoStreamSpatioTemporalGnn import TwoStreamSpatioTemporalGnn


from mmwave_model_integrator.dataset_generators._online_dataset_generator import _OnlineDatasetGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder

from dotenv import load_dotenv
import os

#loading enviroment variables
load_dotenv()
# DATASET_PATH=os.getenv("DATASET_DIRECTORY")
# MAP_DIRECTORY=os.getenv("MAP_DIRECTORY")
# GENERATED_DATASETS_PATH=os.getenv("GENERATED_DATASETS_PATH")

DATASET_PATH = "/data/IcaRAus/datasets/UAV/Flow_datasets"
# DATASET_PATH = "/data/IcaRAus/datasets/UAV/Radar_datasets"
MAP_DIRECTORY = "/data/IcaRAus/maps"
GENERATED_DATASETS_PATH = "/data/IcaRAus/generated_datasets"

# #setup the datasets
folder_name = "vicon_box"
file_name = "vicon_box_1"

print(os.path.join(DATASET_PATH,folder_name,file_name))

dataset = CpslDS(
    dataset_path=os.path.join(DATASET_PATH,folder_name,file_name), #_spin_recall
    radar_pc_folder="radar_combined_pc",
    camera_folder="camera",
    vehicle_odom_folder="vehicle_odom",
    vicon_folder="vicon_x500_8"
)

map_handler = MapHandler(
    maps_folder=MAP_DIRECTORY,
    map_file="north_vicon_1.yaml"#"cpsl_map.yaml"
)

odom_history = []
vicon_history = []

from scipy.spatial.transform import Rotation

#getting odom datasets (format: [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz])
for idx in range(dataset.num_frames):

    odom_data = dataset.get_vehicle_odom_data(idx=idx)[0,:] #only the the 0th sample
    odom_history.append(odom_data)
    
    #getting the vicon datasets (format: [t_x, t_y, t_z, r_w, r_x, r_y, r_z])
    vicon_data = dataset.get_vicon_data(idx=idx) #only the the 0th sample
    vicon_history.append(vicon_data)

odom_history = np.array(odom_history)
vicon_history = np.array(vicon_history)

#add code to generate plots of the odom and vicon data with the following subplots:
#1. x position
#2. y position
#3. z position
#4. roll angle (degrees)
#5. pitch angle (degrees)
#6. yaw angle (degrees)

# Orientation Conversion
# Odom: [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz] -> indices 4-7 are qw, qx, qy, qz
# Vicon: [t_x, t_y, t_z, r_w, r_x, r_y, r_z] -> indices 3-6 are qw, qx, qy, qz

# Scipy Rotation.from_quat expects [qx, qy, qz, qw]
odom_rot = Rotation.from_quat(odom_history[:, [5, 6, 7, 4]])
vicon_rot = Rotation.from_quat(vicon_history[:, [4, 5, 6, 3]])

# Apply 180 degree rotation about x-axis to Odometry (NED -> FLU)
rot_180_x = Rotation.from_euler('x', 180, degrees=True)
# Pre-multiply changes the World frame (NED->FLU), Post-multiply changes the Body frame (FRD->FLU)
odom_rot = rot_180_x * odom_rot * rot_180_x
# odom_rot = rot_180_x * odom_rot

# Convert Odometry Translation from NED to FLU
odom_pos_flu = odom_history[:, 1:4].copy()
odom_pos_flu[:, 1] = -odom_pos_flu[:, 1]  # Y: East to Left
odom_pos_flu[:, 2] = -odom_pos_flu[:, 2]  # Z: Down to Up

# Calculate initial offset for alignment
odom_rot_0 = odom_rot[0]
vicon_rot_0 = vicon_rot[0]
rot_align = odom_rot_0 * vicon_rot_0.inv()

odom_pos_0 = odom_pos_flu[0]
vicon_pos_0 = vicon_history[0, 0:3]
trans_align = odom_pos_0 - rot_align.apply(vicon_pos_0)

# Apply alignment to all Vicon frames
vicon_rot_aligned = rot_align * vicon_rot
vicon_pos_aligned = rot_align.apply(vicon_history[:, 0:3]) + trans_align

odom_euler = odom_rot.as_euler('xyz', degrees=True)
vicon_euler_aligned = vicon_rot_aligned.as_euler('xyz', degrees=True)

# Unwrapping angles for better visualization
odom_euler = np.rad2deg(np.unwrap(np.deg2rad(odom_euler), axis=0))
vicon_euler_aligned = np.rad2deg(np.unwrap(np.deg2rad(vicon_euler_aligned), axis=0))

# Subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
fig.suptitle(f"Odometry vs MOCAP (Aligned): {file_name}")

t = odom_history[:, 0] - odom_history[0, 0]

# Pos X
axs[0, 0].plot(t, odom_pos_flu[:, 0], label='Odom (GNN)')
axs[0, 0].plot(t, vicon_pos_aligned[:, 0], label='Vicon (Aligned)')
axs[0, 0].set_ylabel('X Position (m)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Pos Y
axs[1, 0].plot(t, odom_pos_flu[:, 1], label='Odom (GNN)')
axs[1, 0].plot(t, vicon_pos_aligned[:, 1], label='Vicon (Aligned)')
axs[1, 0].set_ylabel('Y Position (m)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Pos Z
axs[2, 0].plot(t, odom_pos_flu[:, 2], label='Odom (GNN)')
axs[2, 0].plot(t, vicon_pos_aligned[:, 2], label='Vicon (Aligned)')
axs[2, 0].set_ylabel('Z Position (m)')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Roll
axs[0, 1].plot(t, odom_euler[:, 0], label='Odom (GNN)')
axs[0, 1].plot(t, vicon_euler_aligned[:, 0], label='Vicon (Aligned)')
axs[0, 1].set_ylabel('Roll (deg)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Pitch
axs[1, 1].plot(t, odom_euler[:, 1], label='Odom (GNN)')
axs[1, 1].plot(t, vicon_euler_aligned[:, 1], label='Vicon (Aligned)')
axs[1, 1].set_ylabel('Pitch (deg)')
axs[1, 1].legend()
axs[1, 1].grid(True)


# Yaw
axs[2, 1].plot(t, odom_euler[:, 2], label='Odom (GNN)')
axs[2, 1].plot(t, vicon_euler_aligned[:, 2], label='Vicon (Aligned)')
axs[2, 1].set_ylabel('Yaw (deg)')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].legend()
axs[2, 1].grid(True)

# Save the combined plot to a file
plt.tight_layout()
save_path = f"scripts/{file_name}_odom_vs_mocap.png"
plt.savefig(save_path)
print(f"Saved plot to {save_path}")
plt.show()
