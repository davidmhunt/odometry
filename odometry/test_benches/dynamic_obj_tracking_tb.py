import numpy as np

import matplotlib.pyplot as plt

from odometry.test_benches._test_bench import _TestBench
from odometry.point_cloud_processing._point_cloud_integrator import _PointCloudIntegrator
from odometry.localization.icp2D_localization import icp2DLocalization

class DynamicObjTrackingTB(_TestBench):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_current_tracks(self):
        return self.dynamic_object_tracker.current_tracks

    def get_track_history(self):
        return self.dynamic_object_tracker.track_history
    
    def get_full_track_history(self):
        return self.dynamic_object_tracker.track_history_full
    
    
    def run(self, max_frame=-1, gt_enabled=True, movie_generator = None,generate_dataset=False):
        """Run the test bench

        Args:
            max_frame (int, optional): The frame to run the test bench up to.
                -1 indicates to run the entire dataset. Defaults to -1.
            gt_enabled (bool, optional): On True, additionally computes
                ground truth trajectories as well. Defaults to True.
            movie_generator (MovieGenerator, optional): When provided with a 
                MovieGenerator, additionally generates a movie. Defaults to None.
        """


        return super().run(max_frame, gt_enabled, movie_generator)


    def plot_compilation(
            self,
            idx=-1,
            axs: plt.Axes = [],
            show=False
        ):

        if len(axs) == 0:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2
            fig.subplots_adjust(wspace=0.3, hspace=0.30)
        
        self.plotter_localization.plot_position_history_m(
            history_position_m=self.history_position_m,
            history_position_m_gt=self.history_position_m_gt,
            idx=idx+1,
            ax=axs[0,1],
            show=False
        )

        if self.dataset.camera_enabled:

            axs[0,0].imshow(
                self.dataset.get_camera_frame(idx)
            )
            axs[0,0].set_title("Camera View")

        self.plotter_localization.plot_position_history_m(
            history_position_m=self.history_position_m,
            history_position_m_gt=self.history_position_m_gt,
            idx=idx+1,
            ax=axs[0, 1],
            show=False
        )
        axs[0, 1].set_title("Position & Heading History")

        ### bottom left ###
        ax = axs[1, 0]
        #get the combined radar point cloud [x,y,z,vel]
        radar_points = self.dataset.get_radar_data(idx=idx)

        static_points = self.vel_filtering.get_static_detections(
            detections=radar_points,
            ego_vel=np.array([self.filter.x[3],0.0])
        )
        
        # dynamic_combined_pc = self.dynamic_object_tracker.history_dynamic_objects[-1]
        # current_centroids = self.dynamic_object_tracker.history_clustered_dynamic_centroids[-1]
        if self.dynamic_object_tracker.history_dynamic_objects :
            dynamic_combined_pc = self.dynamic_object_tracker.history_dynamic_objects[-1]
        else:
            dynamic_combined_pc = np.empty((0, 2))
            
        if self.dynamic_object_tracker.history_clustered_dynamic_centroids:
            current_centroids = self.dynamic_object_tracker.history_clustered_dynamic_centroids[-1]
        else:
            current_centroids = {}
        
        
        centroid_points = np.array(list(current_centroids.values()))

        if static_points.shape[0] > 0 or dynamic_combined_pc.shape[0] > 0:
            self.plotter_localization.plot_dynamic_and_static_detections_and_centroids_on_map(
                static_points=static_points,
                dynamic_points=dynamic_combined_pc,
                dynamic_centroids=centroid_points,
                heading_rad=self.filter.x[2],
                pose_m=self.filter.x[0:2],
                ax=ax,
                show=False
            )
            ax.set_title("Dynamic + Static Detections", fontsize=self.plotter_localization.font_size_title)
        ax.grid(False)


        ### bottom right, history tracks ###
        ax = axs[1, 1]
        if hasattr(self.map_handler, 'map_points'):
            map_points = self.map_handler.map_points
            ax.scatter(
                map_points[:, 0],
                map_points[:, 1],
                label="Map",
                marker=".",
                s=0.5,
                color="blue"
            )
        track_history = self.dynamic_object_tracker.get_full_track_history()

        for track_id, traj in track_history.items():
            if len(traj) < 2:
                continue
            xs = [p[1] for p in traj]
            ys = [p[2] for p in traj]
            ax.plot(xs, ys, label=f'Track {track_id}')
            ax.text(xs[-1], ys[-1], str(track_id), fontsize=8)

        ax.set_title("All History Tracks", fontsize=self.plotter_localization.font_size_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)  


        self.plotter_localization.marker_size = 10

        if show:
            plt.show()
