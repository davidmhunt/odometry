import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter

class _PointCloudIntegrator:

    def __init__(
            self,
            grid_resolution_m:float = 5e-2,
            grid_max_distance_m:float = 3,
            min_detection_radius:float = 0.25,
            max_detection_radius:float = 20.0,
            num_frames_detection_grid:int = 10
    )-> None:
        
        #pose tracking
        self.previous_pose:Pose = None

        #detection filtering
        self.min_detection_radius:float = min_detection_radius
        self.max_detection_radius:float = max_detection_radius
        self.pc_range_filter:pcRangeFilter = pcRangeFilter(
            min_detection_radius_m=self.min_detection_radius,
            max_detection_radius_m=self.max_detection_radius
        )
        
        #grid parameters
        self.grid_resolution_m:float = grid_resolution_m
        self.grid_max_distance_m:float = grid_max_distance_m
        self.num_frames_detection_grid:int = num_frames_detection_grid
        
        #grid to store samples
        self.grid_bins:np.ndarray = np.arange(
            start=-1 * self.grid_resolution_m,
            stop=self.grid_resolution_m,
            step=self.grid_resolution_m
        )

        #grids
        self.detection_grid:np.ndarray = \
            np.zeros(
                shape = (
                    self.num_frames_detection_grid + 1,
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.int8
            )
        
        #temporary variables
        self.accumulated_points = np.zeros(shape=(0,4))
        

    def reset(self): 
        pass

    def add_points(
            self,
            static_points:np.ndarray,
            current_pose:Pose
    ):
        
        #filter points to the desired range
        static_points = self.pc_range_filter.get_points_in_detection_range(
            points=static_points
        )

        #compute the transformation to go from the previous
        if self.previous_pose:
            
            transformation = Transformation.from_orig_to_new(
                original_pose=self.previous_pose,
                new_pose=current_pose
            )

            #move the accumulated points into the current pose's
            #sensor frame
            self.accumulated_points = \
                transformation.apply_transformation(
                    points=self.accumulated_points
                )
        
        #save the previous pose
        self.previous_pose = current_pose

        #save the add the currently sensed points
        #to the accumulated points
        self.accumulated_points = \
            np.vstack((
                self.accumulated_points,
                static_points))

    def get_latest_pc(self) -> np.ndarray:
        
        return self.accumulated_points
