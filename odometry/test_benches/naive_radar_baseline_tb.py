import numpy as np
import matplotlib.pyplot as plt
from odometry.test_benches._test_bench import _TestBench

class NaiveRadarBaselineTB(_TestBench):
    """
    Naive baseline test bench that uses only single-frame radar detections.
    No integration or historical processing is performed beyond dynamic detection removal.
    """

    def process_point_cloud(
            self,
            point_cloud_raw: np.ndarray,
            static_points: np.ndarray,
            dynamic_points: np.ndarray,
            current_pose,
            gt_points: np.ndarray = np.empty(shape=(0, 3))) -> np.ndarray:
        """
        Processes the point cloud by returning only the current static detections.

        Args:
            point_cloud_raw (np.ndarray): Full nx4 radar point cloud [x,y,z,vel].
            static_points (np.ndarray): Static nx4 radar detections [x,y,z,vel].
            dynamic_points (np.ndarray): Dynamic nx4 radar detections [x,y,z,vel].
            current_pose (Pose): Current estimated pose.
            gt_points (np.ndarray, optional): Ground truth detections.

        Returns:
            np.ndarray: [x,y,z] point cloud for localization.
        """
        # Return static detections (excluding velocity)
        if static_points is not None and static_points.shape[0] > 0:
            return static_points[:, 0:3]
        
        return np.empty(shape=(0, 3))

    def plot_compilation(
            self,
            idx: int = -1,
            axs: list[plt.Axes] = [],
            show: bool = False
        ) -> list[plt.Axes]:
        """
        Plot a compilation of results for the naive baseline.

        Args:
            idx (int): Dataset frame index.
            axs (list): Existing axes to plot on (expects a 2x3 grid).
            show (bool): If True, display the plot.

        Returns:
            list: The axes used for plotting.
        """
        if len(axs) == 0:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.subplots_adjust(wspace=0.3, hspace=0.3)

        # --- TOP ROW: Camera & Localization ---
        if self.dataset.camera_enabled:
            axs[0, 0].imshow(self.dataset.get_camera_frame(idx))
            axs[0, 0].set_title("Camera View", fontsize=self.plotter_localization.font_size_title)

        self.plotter_localization.plot_position_history_m(
            history_position_m=self.history_position_m,
            history_position_m_gt=self.history_position_m_gt,
            history_position_m_inertial=self.history_position_m_inertial,
            idx=idx + 1,
            ax=axs[0, 1],
            show=False
        )

        self.plotter_localization.plot_heading_history_deg(
            history_heading_deg=self.history_heading_deg,
            history_heading_deg_gt=self.history_heading_deg_gt,
            history_heading_deg_inertial=self.history_heading_deg_inertial,
            idx=idx + 1,
            ax=axs[0, 2],
            show=False
        )

        # --- BOTTOM ROW: Kalman Filter & Detections ---
        if len(self.history_filter_g) > 0:
            self.plotter_kalman.plot_chi_2_resp(
                g_thresh=self.filter.g_thresh[2],
                g_hist=np.array(self.history_filter_g),
                idx=idx,
                ax=axs[1, 0],
                show=False
            )

        # Configure map plotter defaults for detections
        self.plotter_localization.marker_size = 0.5
        self.plotter_localization.plot_x_max = 7.5
        self.plotter_localization.plot_y_max = 7.5

        # Show the points used for localization in this frame
        if len(self.history_pc_processor_point_cloud) > 0:
            pc = self.history_pc_processor_point_cloud[-1]
            if pc.shape[0] > 0:
                self.plotter_localization.plot_detections_on_map(
                    current_points=pc[:, 0:2],
                    heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                    pose_m=self.history_position_m[idx],
                    ax=axs[1, 1],
                    show=False
                )
                axs[1, 1].set_title(
                    "Static Detections: {}".format(pc.shape[0]),
                    fontsize=self.plotter_localization.font_size_title
                )

        # Show Ground Truth Detections if available
        if self.gt_localizer is not None or self.dataset.vicon_enabled:
             # get_gt_points is not in _TestBench but in subclasses usually.
             # In _TestBench, we have gt_points in run(). 
             # Since _TestBench doesn't save gt_pc history by default, 
             # we'll just skip the GT detection plot for now or implement it if possible.
             # Actually, _TestBench.history_pc_quality_update uses gt_pc.
             pass

        if show:
            plt.show()

        return axs
