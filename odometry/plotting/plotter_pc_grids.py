import matplotlib.pyplot as plt
import numpy as np

from geometries.transforms.transformation import Transformation
from geometries.pose.orientation import Orientation

from odometry.supportFns import rotation_functions
from odometry.datasets.radnav_ds import radnavDS
from odometry.datasets.map_handler import MapHandler

class PlotterPCGrid:

    def __init__(self,dataset:radnavDS,map_handler:MapHandler) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.marker_size = 10

        #particle filter specific
        self.particle_marker_size = 5
        self.arrow_length = 0.4
        self.particles_x_buffer = 5
        self.particles_y_buffer = 5       

        #import the dataset
        self.dataset:radnavDS = dataset
        self.map_handler:MapHandler = map_handler

        return
    
    def plot_pc_grid(
            self,
            grid:np.ndarray,
            grid_bins:np.ndarray,
            valid_points:np.ndarray=np.empty(shape=(0,3)),
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a point cloud grid, and optionally the detections from the grid
        NOTE: Assumes that [0,0] is the bottom left coordinate of the grid

        Args:
            grid (np.ndarray): an NxN point cloud grid
            grid_bins (np.ndarray): N bins corresponding to the bins
                for each cell in the occupancy grid
            valid_points (np.ndarray, optional): if available, additionally
                plots the valid points on the grid as well. 
                Defaults to np.empty(shape=(0,3)).
            ax (plt.Axes, optional): A set of axes to plot on. 
                Defaults to None.
            show (bool, optional): on True, shows the plot. 
                Defaults to False.
        """
        
        if not ax:
            fig,ax = plt.subplots()

        max_rng = grid_bins.max()
        min_rng = grid_bins.min()

        #plot the occupancy grid
        ax.imshow(
            np.fliplr(grid),
            cmap='gray',
            interpolation='none',
            origin="lower",
            extent=(min_rng,max_rng,min_rng,max_rng),
            vmax=1,
            vmin=0
        )
        
        if valid_points.shape[0] > 0:
            rotation = Orientation.from_euler(
                yaw=90,
                degrees=True)
            transformation = Transformation(
                rotation=rotation._orientation
            )
            valid_points = transformation.apply_transformation(
                valid_points
            )
            ax.scatter(
                valid_points[:,0],
                valid_points[:,1],
                label="detections",
                marker="D",
                color="red",
                s=self.marker_size)
        

        ax.set_title("Occupancy Probability",fontsize=self.font_size_title)
        ax.set_xlim(min_rng,max_rng)
        ax.set_xlabel("Y(m)",fontsize=self.font_size_axis_labels)
        ax.set_ylim(min_rng,max_rng)
        ax.set_ylabel("X(m)",fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)
        # handles,labels = ax.get_legend_handles_labels()
        # ax.legend(handles[1:3], labels[1:3], loc="lower right",fontsize=self.font_size_legend)

        if show:
            plt.show()

        return