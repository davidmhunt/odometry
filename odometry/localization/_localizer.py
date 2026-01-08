import numpy as np

from geometries.pose.pose import Pose
from geometries.pose.orientation import Orientation
from geometries.pose.position import Position

class _Localizer:
    """Abstract base class for localization algorithms using ICP with a global map."""

    def __init__(self):
        
        # Add support for maps
        self.map_points = None
        self.map_bounds_x = np.array([0.0, 0.0])
        self.map_bounds_y = np.array([0.0, 0.0])

        # Initialize the current position in the map to (0,0)
        self.current_pose_m = np.array([0.0, 0.0])
        self.current_heading_rad = 0.0

        self.current_valid_points = np.empty(shape=(0, 2))

        # Run initialization functions
        self.reset_odometry()

    ####################################################################
    # Load map points
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
    # Reset odometry
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
    # Filtering point cloud
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
            (points[:,1] <= self.map_bounds_y[1])
    
        return points[(valid_idxs_x & valid_idxs_y),:]
    
    ####################################################################
    # Using ICP with the map
    ####################################################################   

    def update_odometry(
            self,
            points: np.ndarray = np.empty(shape=(0,2)),
            estimated_heading_rad: float = None,
            estimated_pose_m: np.ndarray = np.empty(shape=(0))
    )-> tuple:
        """Updates the odometry given a new set of points.

        Args:
            points (np.ndarray): Nx2 numpy array of (x,y) detections from the sensor
            estimated_heading_rad (float, optional): Estimated heading of the sensor in radians. Defaults to None.
            estimated_pose_m (np.ndarray, optional): Estimated position (in global reference frame) of the sensor. Defaults to empty array.

        Returns:
            tuple: (current_heading_rad, current_pose_m) or (None, None) if update was unsuccessful
        """
        
        self.estimated_heading_rad = estimated_heading_rad
        self.estimated_pose_m = estimated_pose_m

        return self.estimated_heading_rad, self.estimated_pose_m