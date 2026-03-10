from odometry.localization.icp2D import icp2D
from odometry.supportFns import rotation_functions
import numpy as np
from odometry.localization._localizer import _Localizer

class icp2DLocalization(icp2D):
    """a special class of the icp2D class which uses a global map
    to determine an agent's pose given its sensed point cloud
    """
    def __init__(self, 
                 icp_matching_distance_threshold=1, 
                 icp_best_points_percentile=60, 
                 icp_convergence_translation_threshold=0.001, 
                 icp_convergence_rotation_threshold=0.0001, 
                 icp_point_pairs_threshold=4, 
                 icp_max_iterations=20, 
                 self_detection_radius_m=0.25):
                
        #adding support for ICP processing
        super().__init__(
            icp_matching_distance_threshold, 
            icp_best_points_percentile, 
            icp_convergence_translation_threshold, 
            icp_convergence_rotation_threshold, 
            icp_point_pairs_threshold, 
            icp_max_iterations, 
            self_detection_radius_m
        )

        return
    
    ####################################################################
    #Using icp with the map
    ####################################################################   

    def update_odometry(
            self,
            points:np.ndarray,
            estimated_heading_rad=None,
            estimated_pose_m=np.empty(shape=(0))
    ):
        """Updates the odometry given a new set of points

        Args:
            points (np.ndarray): Nx2 numpy array of (x,y) detections from the sensor
            estimated_heading_rad (_type_,optional): estimated heading of the
                sensor in radians. Defaults to None
            estimated_pose_m (np.ndarray, optional): estimated position (in global
                reference frame) of the sensor. Defaults to None
        Returns:
            list: self.current_heading_rad (the current heading in the global frame),
                self.current_pose_m (the current position in global frame).
                returns None,None if icp odometry update was unsuccessful
        """

        current_points = points
        
        if not estimated_heading_rad:
            estimated_heading_rad = self.current_heading_rad
        
        if estimated_pose_m.shape[0] != 2:
            estimated_pose_m = self.current_pose_m
        
        if (current_points.shape[0] != 0):
            
            #remove self detections from the sensor
            current_points = self.remove_sensor_self_detections(current_points)

            #perform navigation
            new_heading_rad,new_pose_m = self.icp(
                    current_points= current_points,
                    reference_points=self.map_points,
                    estimated_heading_rad=estimated_heading_rad,
                    estimated_pose_m=estimated_pose_m
                )
            
            #update the heading and pose information as possible
            if new_heading_rad is not None:
                self.current_heading_rad = new_heading_rad
                # self.current_heading_rad = \
                #     rotation_functions.wrap_heading(self.current_heading_rad)
            if new_pose_m is not None:
                self.current_pose_m = new_pose_m

            return new_heading_rad,new_pose_m
        else:
            self.current_heading_rad = estimated_heading_rad
            self.current_pose_m = estimated_pose_m
            return estimated_heading_rad,estimated_pose_m