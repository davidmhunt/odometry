import numpy as np
from sklearn.cluster import DBSCAN

from odometry.supportFns import rotation_functions
from odometry.supportFns import coordinate_systems

class pcStacker:
    """Point Cloud stacker object which can be used to "stack" point clouds
    over time to obtained a "combined" point cloud for down-stream processing.
    Note that this object stores a quantized version of the point cloud 
    with a specified resolution and maximum distance
    """

    def __init__(
            self,
            resolution_m:float = 5e-2,
            max_distance_m:float = 20) -> None:
        """_summary_

        Args:
            resolution_m (float, optional): The resolution of the quantized
                point cloud to be stored. Defaults to 5e-2.
            max_distance_m (float, optional): The maximum distance (+/-)
                of the quantized point cloud in x,y. Defaults to 20.
        """
        
        #keeping track of the initial/current pose
        # (in the global reference frame)
        self.initial_heading_rad:float = 0.0
        self.current_heading_rad:float = 0.0
        self.rel_heading_rad:float = 0.0

        self.initial_pose_m:np.ndarray = np.array([0.0,0.0])
        self.current_pose_m:np.ndarray = np.array([0.0,0.0])
        self.rel_pose_m:np.ndarray = np.array([0.0,0.0])

        #keeping track of elapsed time
        self.initial_time_s:float = 0.0
        self.current_time_s:float = 0.0
        self.elapsed_time_s:float = 0.0

        #keeping track of the resolution of the stored point cloud
        self.resolution_m:float = resolution_m
        self.max_distance_m:float = max_distance_m

        #the quantized grid to store
        self.range_bins:np.ndarray = np.arange(
            start=-1 * self.max_distance_m,
            stop=self.max_distance_m,
            step=self.resolution_m
        )
        self.point_cloud_grid:np.ndarray = \
            np.zeros((self.range_bins.shape[0],self.range_bins.shape[0]),
                     dtype=np.int8)

        return
    
    def reset_full(
            self,
            initial_heading_rad:float=0.0,
            initial_pose_m =np.array([0.0,0.0]),
            initial_time_s:float=0.0
            ):
        """Reset the point cloud stacker to start generating a new point cloud

        Args:
            initial_heading_rad (float, optional): Initial heading for the
                 stacked point clouds. Defaults to 0.0.
            initial_pose_m (np.ndarray, optional): Initial position in (x,y)
                of the stacked point clouds. Defaults to np.array([0.0,0.0]).
            initial_time_s (float, optional): If available, the start time
                of the first frame in the stacked point cloud. Defaults to 
                0.0 seconds
        """

        #reset the heading tracking
        self.initial_heading_rad = initial_heading_rad
        self.current_heading_rad = initial_heading_rad
        self.rel_heading_rad = 0.0

        #reset the position tracking
        self.initial_pose_m = initial_pose_m
        self.current_pose_m = initial_pose_m
        self.rel_pose_m = np.array([0.0,0.0])

        #reset time tracking
        self.initial_time_s = initial_time_s
        self.current_time_s = initial_time_s
        self.elapsed_time_s = 0.0

        self.point_cloud_grid = \
            np.zeros((self.range_bins.shape[0],self.range_bins.shape[0]),
                     dtype=np.int8)

        return
    
    def reset_recenter(
            self,
            initial_heading_rad:float=0.0,
            initial_pose_m =np.array([0.0,0.0]),
            initial_time_s:float=0.0,
            initial_pc:np.ndarray = np.empty(shape=(0,2))
            ):
        """Reset the point cloud stacker to be centered around a new location

        Args:
            initial_heading_rad (float, optional): Initial heading for the
                 stacked point clouds. Defaults to 0.0.
            initial_pose_m (np.ndarray, optional): Initial position in (x,y)
                of the stacked point clouds. Defaults to np.array([0.0,0.0]).
            initial_time_s (float, optional): If available, the start time
                of the first frame in the stacked point cloud. Defaults to 
                0.0 seconds
            initial_pc (np.ndarray,optional): The initial point cloud (in the global frame) #TODO: check which frame to actually put these in
                to use if avaialble. If none provided, will translate 
                previous stacked point cloud into the new reference frame. Defaults to
                np.empty(shape=(0,2))
        """

        if (initial_pc.shape[0] == 0):
            #get the previous stacked points in new sensor frame
            self.current_heading_rad = initial_heading_rad
            self.current_pose_m = initial_pose_m

            #get the points in the new reference frame
            initial_pc = self.get_points()

        
        
        #reset the grid
        self.point_cloud_grid = \
            np.zeros((self.range_bins.shape[0],self.range_bins.shape[0]),
                    dtype=np.int8)
            
        
        #filter out points that are out of the grid now
        valid_x_idxs = np.abs(self.range_bins[:,None] - initial_pc[:,0]) <= self.resolution_m
        initial_pc = initial_pc[valid_x_idxs,:]

        valid_y_idxs = np.abs(self.range_bins[:,None] - initial_pc[:,1]) <= self.resolution_m
        initial_pc = initial_pc[valid_y_idxs,:]

        #add the points into the grid
        x_idx = np.argmin(np.abs(
            self.range_bins[:,None] - initial_pc[:,0]),
            axis=0
        )
        y_idx = np.argmin(np.abs(
            self.range_bins[:,None] - initial_pc[:,1]),
            axis=0
        )

        #add points to the grid
        self.point_cloud_grid[x_idx,y_idx] = 1

        #reset the heading tracking
        self.initial_heading_rad = initial_heading_rad
        self.current_heading_rad = initial_heading_rad
        self.rel_heading_rad = 0.0

        #reset the position tracking
        self.initial_pose_m = initial_pose_m
        self.current_pose_m = initial_pose_m
        self.rel_pose_m = np.array([0.0,0.0])

        #reset time tracking
        self.initial_time_s = initial_time_s
        self.current_time_s = initial_time_s
        self.elapsed_time_s = 0.0

        return
    
    ####################################################################
    #Compiling the point clouds
    ####################################################################

    def add_points(
            self,
            current_points:np.ndarray,
            heading_rad:float,
            pose_m:np.ndarray,
            current_time_s:float = 0.0,
    ):
        
        """Moves the current point cloud into the initial reference frame
        and then appends the points to the current combined point cloud list

        Args:
            current_points (np.ndarray): point cloud in agent frame
            heading_rad (float): the heading of the vehicle
                in the global frame
            pose_m (np.ndarray): the (x,y) position of the vehicle
                in the global frame
            current_time_s (float,optional): the time at which the point 
                cloud points were captured (i.e. current time) in seconds.
                Defaults to 0.0 seconds
        """
        #update the current position
        self.current_heading_rad = heading_rad
        self.current_pose_m = pose_m

        #compute the relative heading/pose from the initial point
        self.rel_heading_rad = heading_rad - self.initial_heading_rad
        self.rel_pose_m = pose_m - self.initial_pose_m

        #update the time tracking
        self.current_time_s = current_time_s
        self.elapsed_time_s = current_time_s - self.initial_time_s
        
        #get rot/trans from sensor frame (at current position) to global
        R_cur_to_global = rotation_functions.get_rot_matrix(heading_rad)

        #get rot/trans from global to initial global pose

        #(R from global -> initial sensor frame is inverse of sens -> global)
        R_global_to_init = rotation_functions.get_rot_matrix(self.initial_heading_rad)

        #compute transformation from current -> initial (in sensor frame)
        R = R_cur_to_global.T @ R_global_to_init
        trans = (pose_m - self.initial_pose_m) @ R_global_to_init

        #apply the rotation and translation
        aligned_points = (current_points @ R) + trans
        
        x_idx = np.argmin(np.abs(
            self.range_bins[:,None] - aligned_points[:,0]),
            axis=0
        )
        y_idx = np.argmin(np.abs(
            self.range_bins[:,None] - aligned_points[:,1]),
            axis=0
        )

        self.point_cloud_grid[x_idx,y_idx] = 1

    
    def get_points(self)->np.ndarray:
        """Obtain the currently stacked point cloud in the current sensor frame
        (translates points from the initial position to the current position)

        Returns:
            np.ndarray: Nx2 array of points in the current sensor frame
        """

        #convert the grid to a point cloud
        x_idxs,y_idxs = np.nonzero(self.point_cloud_grid)

        if x_idxs.shape[0] > 0:
        
            x_vals = self.range_bins[x_idxs]
            y_vals = self.range_bins[y_idxs]

            #return the point cloud in the reference frame of the current
            #location of the vehicle (from the current position, not the 
            #initial position)

            current_points = np.column_stack((x_vals,y_vals))
            #get rot/trans from initial sensor frame (at current position) to global
            R_init_to_global = rotation_functions.get_rot_matrix(self.initial_heading_rad)

            #(R from global -> current sensor frame is inverse of sens -> global)
            R_global_to_curr = rotation_functions.get_rot_matrix(self.current_heading_rad)

            #compute transformation from current -> initial (in sensor frame)
            R = R_init_to_global.T @ R_global_to_curr
            trans = (self.initial_pose_m - self.current_pose_m) @ R_global_to_curr

            #apply the rotation and translation
            return (current_points @ R) + trans
        
        else:
            return np.empty(shape=(0,2))
    

    def get_points_from_initial_pose(self)->np.ndarray:
        """Obtain the currently stacked point cloud in the initial sensor frame 

        Returns:
            np.ndarray: Nx2 array of points in the initial sensor frame
        """

        #convert the grid to a point cloud
        x_idxs,y_idxs = np.nonzero(self.point_cloud_grid)

        if x_idxs.shape[0] > 0:
        
            x_vals = self.range_bins[x_idxs]
            y_vals = self.range_bins[y_idxs]

            return np.column_stack((x_vals,y_vals))
        else:
            return np.empty(shape=(0,2))
    
    ####################################################################
    #Get final point cloud, check distance covered and rotation angle
    ####################################################################

    def get_rel_distance_m(self)->float:
        """Return distance traveled since last reset

        Returns:
            float: Euclidian norm of distance traveled
        """

        return np.linalg.norm(self.rel_pose_m)
    
    def get_rel_heading_deg(self)->float:
        """Obtain the total rotation since the last reset

        Returns:
            float: Absolute value of total rotation in degrees
        """

        return np.abs(np.rad2deg(self.rel_heading_rad))
    
    def get_elapsed_time(self)->float:
        """Obtain the total time elapsed since the last reset

        Returns:
            float: total time in seconds since last reset
        """

        return self.elapsed_time_s