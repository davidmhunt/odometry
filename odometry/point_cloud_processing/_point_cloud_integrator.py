import numpy as np

from geometries.pose.pose import Pose
from geometries.transforms.transformation import Transformation

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter
from odometry.point_cloud_processing.pc_grid.probabilistic_pc_grid import ProbabilisticPCGrid

class _PointCloudIntegrator:

    def __init__(
            self,
            probabilistic_pc_grid:ProbabilisticPCGrid,
            num_frames_persistance:int=30,
            min_detection_radius:float = 0.25,
            max_detection_radius:float = 20.0,
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
        
        #probabilistic point grid for initial detections
        self.probabilistic_pc_grid:ProbabilisticPCGrid = probabilistic_pc_grid        
        
        #accumulated points
        self.num_frames_persistance:int = num_frames_persistance
        self.detection_history = np.zeros(shape=(0,4))

        #temporary variables
        self.accumulated_points_raw = np.zeros(shape=(0,4))

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

            #add points to the probabilistic point cloud grid
            self.probabilistic_pc_grid.apply_transformation(transformation)
            self.probabilistic_pc_grid.add_points(static_points[:,:3]) #only send x,y,z (not vel)

            #move the accumulated points into the current pose's
            # sensor frame
            self.accumulated_points_raw = \
                transformation.apply_transformation(
                    points=self.accumulated_points_raw
                )
            
            self.detection_history = \
                transformation.apply_transformation(
                    points=self.detection_history
                )
        
        #save the previous pose
        self.previous_pose = current_pose

        #save the add the currently sensed points
        #to the accumulated points
        
        self.update_history(
            detections=self.probabilistic_pc_grid.get_points(),
            raw_points=static_points[:,0:3]
        )
    
    def update_history(self,detections:np.ndarray,raw_points:np.ndarray):
        """Update the detection history and prune
        detections that have since expired
        TODO: Replace this with a more robust method
        in the future

        Args:
            detections (np.ndarray): Nx3 array of 
                detections
            raw_points (np.ndarray): Nx3 array of 
                detections
        """
        
        if self.detection_history.shape[0] > 0:

            #decay the detection history by a step
            self.detection_history[:,3] = \
                self.detection_history[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.detection_history[:,3] > 0
            self.detection_history = self.detection_history[valid_idxs]
        
        if self.accumulated_points_raw.shape[0] > 0:

            #decay the detection history by a step
            self.accumulated_points_raw[:,3] = \
                self.accumulated_points_raw[:,3] - 1
            
            #prune any detections that have since decayed
            valid_idxs = self.accumulated_points_raw[:,3] > 0
            self.accumulated_points_raw = self.accumulated_points_raw[valid_idxs]

        if detections.shape[0] > 0:
            dets_to_add = np.hstack((
                detections,
                np.zeros(shape=
                         (detections.shape[0],1))
            ))
            dets_to_add[:,3] = self.num_frames_persistance

            self.detection_history = \
                np.vstack((
                    self.detection_history,
                    dets_to_add
                ))
        
        if raw_points.shape[0] > 0:
            dets_to_add = np.hstack((
                raw_points,
                np.zeros(shape=
                         (raw_points.shape[0],1))
            ))
            dets_to_add[:,3] = self.num_frames_persistance

            
            self.accumulated_points_raw = \
                np.vstack((
                    self.accumulated_points_raw,
                    dets_to_add
                ))

    def get_latest_pc(self) -> np.ndarray:
        
        return self.detection_history
