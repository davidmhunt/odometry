from odometry.localization.icp2D import icp2D
from odometry.supportFns import rotation_functions
import numpy as np

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
        
        super().__init__(
            icp_matching_distance_threshold, 
            icp_best_points_percentile, 
            icp_convergence_translation_threshold, 
            icp_convergence_rotation_threshold, 
            icp_point_pairs_threshold, 
            icp_max_iterations, 
            self_detection_radius_m)
        
        #add support for maps
        self.map_points = None
        self.map_bounds_x = np.array([0.0,0.0])
        self.map_bounds_y = np.array([0.0,0.0])

        #initialize the current position in the map to (0,0)
        self.current_pose_m = np.array([0.0,0.0])
        self.current_heading_rad = 0.0

        #run initialization functions
        self.reset_odometry()

        return

    ####################################################################
    #Load map points
    ####################################################################

    def load_map_point_cloud(self,map_points:np.ndarray):
        """Load a map's point cloud to be used for localization tasks

        Args:
            map_points (np.ndarray): Nx2 numy array of points for the global map
        """

        self.map_points = map_points
        self.set_map_bounds_from_map(self.map_points)
        print("loaded map with {} points".format(self.map_points.shape[0]))

        return
    
    def set_map_bounds_from_map(self, map_points:np.ndarray):
        """Set the map boundaries

        Args:
            map_points (np.ndarray): Nx2 array of map points
        """
        self.map_bounds_x = np.array([
            np.min(map_points[:,0]) - 0.5,
            np.max(map_points[:,0]) + 0.5
        ])
        self.map_bounds_y = np.array([
            np.min(map_points[:,1]) - 0.5,
            np.max(map_points[:,1]) + 0.5
        ])

    ####################################################################
    #reset odometry
    ####################################################################
    
    def reset_odometry(
            self,
            pose:np.ndarray=np.array([0.0,0.0]),
            heading_rad=0.0):
        """Reset the odometry within the map to a given pose and rotation

        Args:
            pose (np.ndarray, optional): _description_. Defaults to np.array([0.0,0.0]).
            heading_rad (float, optional): _description_. Defaults to 0.0.
        """
               
        #initialize the current position
        self.current_pose_m = pose
        self.current_heading_rad = heading_rad

        #reset the current valid points array
        self.current_valid_points = np.empty(shape=(0,3))

        return

    ####################################################################
    #Filtering point cloud
    ####################################################################

    def remove_detections_outside_of_map(self,points:np.ndarray):
        """Remove points that are outside of the bounds of the map

        Args:
            points (np.ndarray): a Nx2 array of points (in the 
                global map's reference frame)

        Returns:
            np.ndarray: All of the points in the map's reference frame
        """
        valid_idxs_x = (points[:,0] >= self.map_bounds_x[0]) & \
            (points[:,0] <= self.map_bounds_x[1])
        
        valid_idxs_y = (points[:,1] >= self.map_bounds_y[0]) & \
            (points[:,1] <= self.map_bounds_x[1])
    
        return points[(valid_idxs_x & valid_idxs_y),:]
    
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
            if new_heading_rad is not None:
                self.current_heading_rad = new_heading_rad
                # self.current_heading_rad = \
                #     rotation_functions.wrap_heading(self.current_heading_rad)
            if new_pose_m is not None:
                self.current_pose_m = new_pose_m

        return new_heading_rad,new_pose_m