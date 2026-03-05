import numpy as np
import matplotlib.pyplot as plt
from cpsl_datasets.cpsl_ds import CpslDS

def main():
    # Initialize the dataset for the UAV vicon box 1 testing scenario
    dataset = CpslDS(
        dataset_path="/data/IcaRAus/datasets/UAV/Flow_datasets/vicon_box/vicon_box_3",
        vicon_folder="vicon_x500_8",
        camera_folder="",
        radar_folder="AWR1843_0",
        lidar_folder="",
        vehicle_odom_folder="vehicle_odom"
    )

    num_frames = dataset.num_frames

    vicon_x, vicon_y, vicon_z = [], [], []
    odom_x, odom_y, odom_z = [], [], []

    for i in range(num_frames):
        # Fetch Vicon data
        vicon_sample = dataset.get_vicon_data(i)
        if vicon_sample.shape[0] > 0:
            vicon_x.append(vicon_sample[0])
            vicon_y.append(vicon_sample[1])
            vicon_z.append(vicon_sample[2])
        else:
            vicon_x.append(np.nan)
            vicon_y.append(np.nan)
            vicon_z.append(np.nan)
            
        # Fetch Odometry data
        odom_data = dataset.get_vehicle_odom_data(i)
        if odom_data is not None and odom_data.shape[0] > 0:
            # Taking the most recent message in the frame
            odom_sample = odom_data[-1]
            odom_x.append(odom_sample[1])
            odom_y.append(odom_sample[2])
            odom_z.append(odom_sample[3])
        else:
            odom_x.append(np.nan)
            odom_y.append(np.nan)
            odom_z.append(np.nan)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(vicon_x, label='Vicon X', alpha=0.8)
    axs[0].plot(odom_x, label='Odom X', alpha=0.8, linestyle='--')
    axs[0].set_title('X Coordinate Comparison')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel('Meters')

    axs[1].plot(vicon_y, label='Vicon Y', alpha=0.8)
    axs[1].plot(odom_y, label='Odom Y', alpha=0.8, linestyle='--')
    axs[1].set_title('Y Coordinate Comparison')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylabel('Meters')

    axs[2].plot(vicon_z, label='Vicon Z', alpha=0.8)
    axs[2].plot(odom_z, label='Odom Z', alpha=0.8, linestyle='--')
    axs[2].set_title('Z Coordinate Comparison')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlabel('Frames')
    axs[2].set_ylabel('Meters')

    plt.tight_layout()
    output_path = '/home/david/Documents/odometry/vicon_odom_comparison.png'
    plt.savefig(output_path)
    print(f"Plot saved successfully to: {output_path}")

    # Also show the plot if running interactively
    # plt.show()

if __name__ == "__main__":
    main()
