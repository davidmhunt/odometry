import numpy as np
from odometry.point_cloud_processing.vel_filtering import VelFiltering
from odometry.supportFns.rotation_functions import apply_rot_trans

class DynamicObjectTracker:

    def __init__(
            self,
            vel_filtering_v_thresh:float = 0.05,
                 ) -> None:
        
        #vel filtering initialization
        self.vel_filtering:VelFiltering = VelFiltering(
            v_thresh=vel_filtering_v_thresh
        )

        #current list of dynamic objects (x,y coordinates)
        #TODO: update the 3D
        self.current_dynamic_detections:np.ndarray = np.empty()
        self.current_tracks:np.ndarray = np.ndarray()

    def reset():
        pass

    def update(
            self,
            current_points:np.ndarray,
            ego_vel:np.ndarray,
            ego_heading_rad:np.ndarray,
            ego_pose_m:np.ndarray
    ):
        """_summary_

        Args:
            current_points (np.ndarray): Nx4 array of detections with [x,y,z,vel] info
                in the sensor coordinate frame
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle as [x,y]
            ego_heading_rad (np.ndarray): heading of the ego vehicle in the global frame
                in radians
            ego_pose_m (np.ndarray): [x,y] pose of the ego vehicle in the global frame
                in meters 
        """
        
        #get the dynamic detections
        dynamic_points = self.vel_filtering.get_dynamic_detections(
            detections=current_points,
            ego_vel=ego_vel
        )



        #move the detections into the global coordinate frame
        self.current_dynamic_detections = apply_rot_trans(
            points=dynamic_points[:,0:2],
            rot_angle_rad=ego_heading_rad,
            trans=ego_pose_m
        )


        return