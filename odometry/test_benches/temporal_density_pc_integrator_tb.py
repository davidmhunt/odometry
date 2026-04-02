import numpy as np
import matplotlib.pyplot as plt

from odometry.test_benches.point_cloud_integrator_tb import PointCloudIntegratorTB

class TemporalDensityPCIntegratorTB(PointCloudIntegratorTB):
    """
    Test bench specializing in visualizing temporal-density grid data.

    Extends PointCloudIntegratorTB to provide detailed insights into the internal
    log-density and temporal grids maintained by the TemporalDensityPCGrid.
    """

    def plot_compilation(
            self,
            idx=-1,
            axs: plt.Axes = [],
            show=False
        ):
        """
        Plot a compilation of results including specialized grid views in the bottom row.

        Args:
            idx (int): Dataset frame index.
            axs (plt.Axes): Existing axes to plot on.
            show (bool): If True, display the plot.
        """
        if len(axs) == 0:
            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
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

        # --- MIDDLE ROW: Kalman Filter & Points on Map ---
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

        # (Filtered Points in middle-middle)
        detections = self.point_cloud_integrator.get_points()
        if detections.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=detections[:, 0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[1, 1],
                show=False
            )
            axs[1, 1].set_title("Filtered Detections: {}".format(detections.shape[0]), fontsize=self.plotter_localization.font_size_title)

        # (Raw Points in middle-right)
        accumulated_points = self.point_cloud_integrator.get_raw_point_history()
        if accumulated_points.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=accumulated_points[:, 0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[1, 2],
                show=False
            )
            axs[1, 2].set_title(
                "Raw Detections: {}".format(accumulated_points.shape[0]),
                fontsize=self.plotter_localization.font_size_title
            )

        # --- BOTTOM ROW: Density Grid, Temporal Grid, GT Detections ---
        # Assuming the integrator uses TemporalDensityPCGrid
        grid_obj = self.point_cloud_integrator.raw_point_history
        grid_bins = grid_obj.grid_bins

        # axs[2,0]: Density Grid
        self.plotter_pc_grid.plot_pc_grid(
            grid=grid_obj.grid,
            grid_bins=grid_bins,
            ax=axs[2, 0],
            show=False
        )
        axs[2, 0].set_title("Density Grid (Log-Density)", fontsize=self.plotter_localization.font_size_title)

        # axs[2,1]: Temporal Grid
        # Normalize temporal grid [0, 1] using history length
        temporal_grid = grid_obj.temporal_grid
        # temporal_grid = np.log1p(temporal_grid)
        norm_factor = float(self.point_cloud_integrator.num_frames_history)
        temporal_grid_norm = temporal_grid / max(norm_factor, 1.0)

        # norm_factor = float(self.point_cloud_integrator.num_frames_history)
        # temporal_grid_norm = grid_obj.temporal_grid / max(norm_factor, 1.0)
        self.plotter_pc_grid.plot_pc_grid(
            grid=temporal_grid_norm,
            grid_bins=grid_bins,
            ax=axs[2, 1],
            show=False
        )
        axs[2, 1].set_title("Temporal Grid (Recency)", fontsize=self.plotter_localization.font_size_title)

        # axs[2,2]: Ground Truth Detections
        gt_detections = self.point_cloud_integrator.get_gt_points()
        if gt_detections.shape[0] > 0:
            self.plotter_localization.plot_detections_on_map(
                current_points=gt_detections[:, 0:2],
                heading_rad=np.deg2rad(self.history_heading_deg[idx]),
                pose_m=self.history_position_m[idx],
                ax=axs[2, 2],
                show=False
            )
            axs[2, 2].set_title(
                "GT Detections: {}".format(gt_detections.shape[0]),
                fontsize=self.plotter_localization.font_size_title
            )

        if show:
            plt.show()

        return axs
