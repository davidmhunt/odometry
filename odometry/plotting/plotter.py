import matplotlib.pyplot as plt
import numpy as np

from odometry.supportFns import rotation_functions
from odometry.datasets.radnav_ds import radnavDS
from odometry.datasets.map_handler import MapHandler

class Plotter:

    def __init__(self,dataset:radnavDS,map_handler:MapHandler) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20

        #import the dataset
        self.dataset:radnavDS = dataset
        self.map_handler:MapHandler = map_handler

        return
    
    def plot_detections_on_map(
            self,
            current_points:np.ndarray,
            heading_rad,
            pose_m,
            ax:plt.Axes=None,
            show=False
    ):
        """Plots a point cloud onto the known map

        Args:
            current_points (np.ndarray): point cloud in agent frame
            heading_rad (_type_): the heading of the vehicle in the
                global frame
            pose_m (_type_): the position of the vehicle in the 
                global frame
            ax (plt.Axes, optional): A set of axes to plot on. 
                Defaults to None.
            show (bool, optional): on True, shows the plot. 
                Defaults to False.
        """
        
        aligned_points = \
            rotation_functions.apply_rot_trans(
                points=current_points,
                rot_angle_rad=heading_rad,
                trans=pose_m
            )
        
        if not ax:
            fig,ax = plt.subplots()
        
        #plot the map
        map_points = self.map_handler.map_points
        ax.scatter(
            map_points[:,0],
            map_points[:,1],
            label="map",
            marker=".",
            s=0.5,
            color="blue")

        #plot the aligned_detections
        ax.scatter(
            aligned_points[:,0],
            aligned_points[:,1],
            label="detections",
            marker="D",
            color="red")
        
        #plot the pose estimate
        ax.scatter(
            pose_m[0],
            pose_m[1],
            marker="o",
            color="cyan",
            s=15.0,
            label="est position"
        )

        ax.set_title("Point cloud Detections: {}".format(aligned_points.shape[0]),fontsize=self.font_size_title)
        ax.set_xlim(
            pose_m[0] - self.plot_x_max,
            pose_m[0] + self.plot_x_max)
        ax.set_xlabel("X",fontsize=self.font_size_axis_labels)
        ax.set_ylim(
            pose_m[1] - self.plot_y_max,
            pose_m[1] + self.plot_y_max)
        ax.set_ylabel("Y",fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
        ax.grid("True")
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles[1:3], labels[1:3], loc="lower right",fontsize=self.font_size_legend)

        if show:
            plt.show()

        return
    
    def plot_lidar_points_on_map(self,
                                idx,
                                heading_rad,
                                pose_m,
                                ax:plt.Axes=None,
                                show=False):
        """Plots a lidar point cloud onto the known map

        Args:
            heading_rad (_type_): the heading of the vehicle in the
                global frame
            pose_m (_type_): the position of the vehicle in the 
                global frame
            ax (plt.Axes, optional): A set of axes to plot on. 
                Defaults to None.
            show (bool, optional): on True, shows the plot. 
                Defaults to False.
        """

        self.plot_detections_on_map(
            current_points=self.dataset.get_lidar_point_cloud(idx=idx),
            heading_rad=heading_rad,
            pose_m=pose_m,
            ax=ax,
            show=show
        )
        
        return
    
    def plot_radar_points_on_map(self,
                                idx,
                                heading_rad,
                                pose_m,
                                ax:plt.Axes=None,
                                show=False):
        """Plots a lidar point cloud onto the known map

        Args:
            heading_rad (_type_): the heading of the vehicle in the
                global frame
            pose_m (_type_): the position of the vehicle in the 
                global frame
            ax (plt.Axes, optional): A set of axes to plot on. 
                Defaults to None.
            show (bool, optional): on True, shows the plot. 
                Defaults to False.
        """

        radar_points = self.dataset.get_radar_detections(radar_points)
        radar_points = radar_points[:,:2]
        
        self.plot_detections_on_map(
            current_points=radar_points,
            heading_rad=heading_rad,
            pose_m=pose_m,
            ax=ax,
            show=show
        )
        
        return


    def plot_heading_history(self,
                         history_heading_deg:list,
                         idx=0,
                         ax:plt.Axes=None,
                         show:bool=False):
        """Plot the heading history (in degrees)

        Args:
            history_heading_deg (list): history of the heading
            idx (int, optional): max index to plot to. Defaults to 0.
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            show (bool, optional): displays plot on True. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if idx == 0:
            ax.plot(history_heading_deg)
        else:
            ax.plot(history_heading_deg[:idx])
        
        ax.set_title("Heading history (deg)",fontsize=self.font_size_title)
        ax.set_xlabel("Frame",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Heading (deg)",fontsize=self.font_size_axis_labels)
        if idx <= 300:
            ax.set_xlim(0,300)
        else:
            ax.set_xlim(0,idx)
        ax.set_ylim(-3,3)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()
    
    def plot_position_history_m(self,
                         history_position_m:np.ndarray,
                         history_position_m_gt:np.ndarray = np.empty(shape=(0,2)),
                         idx=0,
                         ax:plt.Axes=None,
                         show:bool=False):
        """Plot the heading history (in degrees)

        Args:
            history_position_m (np.ndarray): history of the heading
            label(str,optional): the plot label. Defaults to "radar"
            color(str,optional): the color of the plot. Defaults to "red"
            idx (int, optional): max index to plot to. Defaults to 0.
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            show (bool, optional): displays plot on True. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots(figsize=(3,3))
        
        if idx == 0:
            idx = -1 #plot all of the history
        ax.plot(
            history_position_m[:idx,0],
            history_position_m[:idx,1],
            color="red",
            label="estimated")
        
        if history_position_m_gt.shape[0] > 0:

            ax.plot(
                history_position_m_gt[:idx,0],
                history_position_m_gt[:idx,1],
                color="green",
                label="truth")

        ax.set_xlim(
            np.min(history_position_m_gt[:idx,0]) - 1,
            np.max(history_position_m_gt[:idx,0]) + 1)
        ax.set_ylim(
            np.min(history_position_m_gt[:idx,1]) - 1,
            np.max(history_position_m_gt[:idx,1]) + 1)
            
        ax.set_title("Position",fontsize=self.font_size_title)
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)

        #plot the map
        ax.scatter(
            self.map_handler.map_points[:,0],
            self.map_handler.map_points[:,1],
            label="map",
            marker=".",
            s=0.5,
            color="blue")
        #show the legend
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2],
                labels[:2],
                loc="lower right",
                fontsize=self.font_size_legend)
        if show:
            plt.show()
    
    def plot_pose_error(self,
                         history_position_m:np.ndarray,
                         history_position_m_gt:np.ndarray = np.empty(shape=(0,2)),
                         idx=0,
                         ax:plt.Axes=None,
                         show:bool=False):
        """Plot the heading history (in degrees)

        Args:
            history_position_m (np.ndarray): history of the heading
            label(str,optional): the plot label. Defaults to "radar"
            color(str,optional): the color of the plot. Defaults to "red"
            idx (int, optional): max index to plot to. Defaults to 0.
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            show (bool, optional): displays plot on True. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots(figsize=(3,3))
        
        if idx == 0:
            idx = -1 #plot all of the history
        ax.plot(
            history_position_m[:idx,0],
            history_position_m[:idx,1],
            color="red",
            label="estimated")
        
        if history_position_m_gt.shape[0] > 0:

            ax.plot(
                history_position_m_gt[:idx,0],
                history_position_m_gt[:idx,1],
                color="green",
                label="truth")

        ax.set_xlim(
            np.min(history_position_m_gt[:idx,0]) - 5,
            np.max(history_position_m_gt[:idx,0]) + 5)
        ax.set_ylim(
            np.min(history_position_m_gt[:idx,1]) - 5,
            np.max(history_position_m_gt[:idx,1]) + 5)
            
        ax.set_title("Position",fontsize=self.font_size_title)
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)

        #plot the map
        ax.scatter(
            self.map_handler.map_points[:,0],
            self.map_handler.map_points[:,1],
            label="map",
            marker=".",
            s=0.5,
            color="blue")
        #show the legend
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2],
                labels[:2],
                loc="lower right",
                fontsize=self.font_size_legend)
        if show:
            plt.show()