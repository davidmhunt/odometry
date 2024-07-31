import numpy as np
from sklearn.cluster import DBSCAN

from odometry.supportFns import rotation_functions
from odometry.supportFns import coordinate_systems

from odometry.estimators.estimators import InertialIntegrator,Inertial
from odometry.point_cloud_processing.multipath import MultiPath

from odometry.point_cloud_processing.vel_filtering import VelFiltering

from odometry.point_cloud_processing.ground_detection_filtering import groundDetectionFiltering

class temporalPcStacker:
    """

    Recenter the given point cloud grids based on the change in position and heading between two poses.

    Args:
        grids (np.ndarray): The point cloud grids to be recentered.
        initial_heading_rad (float): The initial heading ("""

    def __init__(
            self,
            num_frames_static_history:int = 4,
            num_frames_dynamic_history:int = 1,
            refresh_distance_m:float = 3,
            refresh_rot_deg:float = 180,
            refresh_time_s:float = 10,
            resolution_m:float = 5e-2,
            max_distance_m:float = 20,
            multi_path_clustering_eps:float = 0.25,
            multi_path_clustering_min_samples:int = 10,
            vel_filtering_enabled:bool = True,
            vel_filtering_v_thresh:float = 1.0,
            vel_filtering_min_static_rejection_radius:float = 2.0,
            vel_filtering_dynamic_cluster_eps:float = 1.0,
            vel_filtering_dynamic_cluster_min_samples = 7,
            self_detection_radius_m = 0.25
            ) -> None:
        """_summary_

        Args:
            resolution_m (float, optional): The resolution of the quantized
                point cloud to be stored. Defaults to 5e-2.
            max_distance_m (float, optional): The maximum distance (+/-)
                of the quantized point cloud in x,y. Defaults to 20.
        """

        #initialize and add in support modules
        self.integrator:InertialIntegrator = InertialIntegrator()
        self.integrator.reset()

        self.multi_path:MultiPath = MultiPath(
            clustering_eps=multi_path_clustering_eps,
            clustering_min_samples=multi_path_clustering_min_samples
        )

        self.vel_filtering_enabled:bool = vel_filtering_enabled
        self.vel_filtering:VelFiltering = VelFiltering(
            v_thresh=vel_filtering_v_thresh,
            min_static_rejection_radius=vel_filtering_min_static_rejection_radius,
            dynamic_cluster_eps=vel_filtering_dynamic_cluster_eps,
            dynamic_cluster_min_samples=vel_filtering_dynamic_cluster_min_samples
        )

        self.ground_detection_filtering:groundDetectionFiltering = groundDetectionFiltering(
            self_detection_radius_m=self_detection_radius_m
        )

        #history parameters
        self.num_frames_static_history = num_frames_static_history
        self.num_frames_dynamic_history = num_frames_dynamic_history

        #refresh rates
        self.refresh_distance_m = refresh_distance_m
        self.refresh_rot_deg = refresh_rot_deg
        self.refresh_time_s = refresh_time_s

        #keeping track of the initial/current pose between each refresh
        # (in the inertial reference frame)
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
        self.pc_grid_static:np.ndarray = \
            np.zeros(
                shape =(num_frames_static_history + 1,
                         self.range_bins.shape[0],
                         self.range_bins.shape[0]),
                     dtype=np.int8)
        self.pc_grid_dynamic:np.ndarray = \
            np.zeros(
                shape =(num_frames_dynamic_history + 1,
                         self.range_bins.shape[0],
                         self.range_bins.shape[0]),
                     dtype=np.int8)

        return

    ####################################################################
    # resetting and recentering the point clouds
    ####################################################################

    def reset(
            self,
            initial_time_s:float=0.0
            ):
        """Reset the point cloud stacker to start generating a new point cloud

        Args:
            initial_time_s (float, optional): If available, the start time
                of the first frame in the stacked point cloud. Defaults to 
                0.0 seconds
        """

        #reset the heading tracking
        self.initial_heading_rad = 0.0
        self.current_heading_rad = 0.0
        self.rel_heading_rad = 0.0

        #reset the position tracking
        self.initial_pose_m = np.array([0.0,0.0])
        self.current_pose_m = np.array([0.0,0.0])
        self.rel_pose_m = np.array([0.0,0.0])

        #reset time tracking
        self.initial_time_s = initial_time_s
        self.current_time_s = initial_time_s
        self.elapsed_time_s = 0.0

        #reset the point cloud grids
        self.pc_grid_static:np.ndarray = \
            np.zeros(
                shape =(self.num_frames_static_history + 1,
                         self.range_bins.shape[0],
                         self.range_bins.shape[0]),
                     dtype=np.int8)
        self.pc_grid_dynamic:np.ndarray = \
            np.zeros(
                shape =(self.num_frames_dynamic_history + 1,
                         self.range_bins.shape[0],
                         self.range_bins.shape[0]),
                     dtype=np.int8)

        #reset the inertial integrator
        self.integrator.reset(
            t0=initial_time_s,
            x0 = np.zeros(shape=4,dtype=float)
        )

        return

    def check_for_refresh(self)->bool:
        """_summary_

        compares the refresh conditions of time, distance, and rotation with elapsed conditions

        Returns:
            bool: if a refresh is needed
        """

        if self.get_elapsed_time > self.refresh_time_s or \
             self.get_rel_distance_m > self.refresh_distance_m or \
            self.get_rel_heading_deg > self.refresh_rot_deg:
            return True
        else:
            return False
    
    def refresh(
        self
    ):
        """
        Recenters the grid maps based on how far and in which direction the vehicle has traveled

        Args:
        - self: The instance of the class.

        Returns: None
        """

        #recenter point clouds
        self.pc_grid_static = self.recenter_pc_grids(
            self.pc_grid_static,
            initial_heading_rad=self.initial_heading_rad,
            initial_pose_m=self.initial_pose_m,
            current_heading_rad=self.current_heading_rad,
            current_pose_m=self.current_pose_m
        )

        if self.vel_filtering_enabled:
            self.pc_grid_dynamic = self.recenter_pc_grids(
                self.pc_grid_dynamic,
                initial_heading_rad=self.initial_heading_rad,
                initial_pose_m=self.initial_pose_m,
                current_heading_rad=self.current_heading_rad,
                current_pose_m=self.current_pose_m
            )

        #TODO: move to a function
        
        #get points from current stacked static grid
        current_points = self._get_points_from_pc_grid(self.pc_grid_static[0])
        dynamic_points = self._get_points_from_pc_grids(self.pc_grid_dynamic)
        # remove dynamic clusters from static detections
        current_points = \
            self.vel_filtering.remove_dynamic_clusters_from_static_detections_knn(
                current_points,
                dynamic_points
            )

        #remove multipath
        current_points = self.multi_path.remove_multipath(current_points)

        # replace current grid with processed grid
        self.pc_grid_static[0] = self._get_pc_grid_from_points(current_points)
        

        #start a new frame
        self.pc_grid_static[1:] = self.pc_grid_static[0:-1]
        self.pc_grid_static[0] = np.zeros(
            shape=(
                self.range_bins.shape[0],
                self.range_bins.shape[0]
                ),
            dtype=np.int8
        )

        #refresh dynamic pc's
        if self.vel_filtering_enabled:
            self.pc_grid_dynamic[1:] = self.pc_grid_dynamic[0:-1]
            self.pc_grid_dynamic[0] = np.zeros(
                shape=(
                    self.range_bins.shape[0],
                    self.range_bins.shape[0]
                ),
                dtype=np.int8
            )

        #reset initial variables for next refresh
        self.initial_pose_m = self.current_pose_m
        self.initial_heading_rad = self.current_heading_rad
        self.initial_time_s = self.current_time_s

        #reset the relative variables
        self.elapsed_time_s = 0.0
        self.rel_heading_rad = 0
        self.rel_pose_m = np.array([0.0,0.0])
        
        return

    def recenter_pc_grids(
        self,
        grids: np.ndarray,
        initial_heading_rad: float,
        initial_pose_m: np.ndarray,
        current_heading_rad: float,
        current_pose_m: np.ndarray,
    ) -> np.ndarray:
        """
        Iterates through multiple pc_grids and recenters them

        Args:
            grids: An ndarray of shape (N, M, M) representing a collection of 2D grids.
            initial_heading_rad: A float representing the initial heading (in radians).
            initial_pose_m: An ndarray of shape (2,) representing the initial pose (x, y) in meters.
            current_heading_rad: A float representing the current heading (in radians).
            current_pose_m: An ndarray of shape (2,) representing the current pose (x, y) in meters.

        Returns:
            A recentered ndarray of shape (N, M, M).
        """

        
        assert grids.ndim == 3, "recenter_pc_grids expects grids to be 3D cube of 2D grids"

        #creates base array of grids
        recentered_grids = np.zeros(
                shape=(
                    grids.shape[0],
                    self.range_bins.shape[0],
                    self.range_bins.shape[0]
                ),
                dtype=np.int8
        )
        
        #loops through all grids and recenters them, setting their respective array slot
        for i in range(grids.shape[0]):
            recentered_grids[i] = self.recenter_pc_grid(
                grids[i],
                initial_heading_rad,
                initial_pose_m,
                current_heading_rad,
                current_pose_m
            )

        return recentered_grids

    def recenter_pc_grid(
        self,
        grid: np.ndarray,
        initial_heading_rad: float,
        initial_pose_m: np.ndarray,
        current_heading_rad: float,
        current_pose_m: np.ndarray,
    ):
        """
        Recenters a grid based upon vehicle movement and filers out any points that have left resolution

        Args:
            grid: An np.ndarray representing the initial grid.
            initial_heading_rad: A float representing the initial heading (in radians).
            initial_pose_m: An ndarray of shape (2,) representing the initial pose (x, y) in meters.
            current_heading_rad: A float representing the current heading (in radians).
            current_pose_m: An ndarray of shape (2,) representing the current pose (x, y) in meters.

        Returns:
            An np.ndarray representing the recentered grid.
        """

        #get points from the grid (initial reference frame)
        grid_points = self._get_points_from_pc_grid(grid)

        #change reference frame of the points to current frame
        if grid_points.shape[0] > 0:
            grid_points = self._change_pc_reference_frame(
                initial_heading_rad,
                initial_pose_m,
                current_heading_rad,
                current_pose_m,
                grid_points
            )

            #filter out points that are out of the grid now
            valid_x_idxs = np.abs(self.range_bins[:,None] - grid_points[:,0]) <= self.resolution_m
            grid_points = grid_points[valid_x_idxs,:]

            valid_y_idxs = np.abs(self.range_bins[:,None] - grid_points[:,1]) <= self.resolution_m
            grid_points = grid_points[valid_y_idxs,:]

            #convert points back to grid
            return self._get_pc_grid_from_points(
                grid_points
            )
        else:
            return np.zeros(
                shape =( self.range_bins.shape[0],
                    self.range_bins.shape[0]),
                dtype=np.int8)

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
        self.pc_grid_static = \
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
        self.pc_grid_static[x_idx,y_idx] = 1

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
    # Predicting inertial integration forward
    ####################################################################
    def predict(self,dt:float,inertial:Inertial):
        """Predict the inertial integrator forward

        Args:
            dt (float): time since last measurement
            inertial (Inertial): Inertial object with at least angular (rad/sec)
            and linear velocity (m/s) measurements
        """

        self.integrator.predict(dt,inertial)

        #update the current position
        self.current_heading_rad = self.integrator.x[2]
        self.current_pose_m = self.integrator.x[0:2]

        #compute the relative heading/pose from the initial point
        self.rel_heading_rad = self.current_heading_rad - self.initial_heading_rad
        self.rel_pose_m = self.current_pose_m - self.initial_pose_m

        #update the time tracking
        self.current_time_s = self.integrator.t
        self.elapsed_time_s = self.current_time_s - self.initial_time_s

    ####################################################################
    #Compiling the point clouds
    ####################################################################

    def add_points(
            self,
            current_points:np.ndarray,
            ego_vel:np.ndarray,
    ):

        """Moves the current point cloud into the initial reference frame
        and then appends the points to the current combined point cloud list

        Args:
            current_points (np.ndarray): nx4 array for point cloud in agent frame
                corresponding to [x,y,z,vel]
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle
        """

        #filter out the ground detections
        current_points = self.ground_detection_filtering.remove_sensor_self_detections(
            points=current_points
        )

        #update the static and dynamic grids
        if self.vel_filtering_enabled:
            static_points = self.vel_filtering.get_static_detections(
                detections=current_points,ego_vel=ego_vel
            )
            dynamic_points = self.vel_filtering.get_dynamic_detections(
                detections=current_points,ego_vel=ego_vel
            )

            self._add_points_to_grid(
                grid=self.pc_grid_dynamic,
                current_points=dynamic_points
            )
        else:
            static_points = current_points

        self._add_points_to_grid(
            grid=self.pc_grid_static,
            current_points=static_points
        )

        #TODO: add functionality to check for refreshing



    ####################################################################
    #Support functions for changing reference frames
    ####################################################################

    def _get_pc_grid_from_points(
            self,
            point_cloud:np.ndarray
    ) -> np.ndarray:
        """Get a point cloud grid from a given set of points

        Args:
            point_cloud (np.ndarray): Nx2 array of [x,y] point cloud points

        Returns:
            np.ndarray: MxM point cloud grid with indicies based on the range bins
        """

        #create a new point cloud grid
        grid = \
            np.zeros(
                shape =( self.range_bins.shape[0],
                        self.range_bins.shape[0]),
                    dtype=np.int8)

        x_idx = np.argmin(np.abs(
            self.range_bins[:,None] - point_cloud[:,0]),
            axis=0
        )
        y_idx = np.argmin(np.abs(
            self.range_bins[:,None] - point_cloud[:,1]),
            axis=0
        )

        grid[0,x_idx,y_idx] = 1

        return grid

    def _get_points_from_pc_grid(
            self,
            pc_grid:np.ndarray
    )->np.ndarray:
        """Obtain a point cloud from a current pc grid

        Args:
            pc_grid (np.ndarray): NxN point cloud grid with indicies based on
                self.range_bins where 1 indicates a point is at that location
                and 0 indicates no point is at that location

        Returns:
            np.ndarray: Nx2 array of points in the initial sensor frame
        """

        #convert the grid to a point cloud
        x_idxs,y_idxs = np.nonzero(pc_grid)

        if x_idxs.shape[0] > 0:

            x_vals = self.range_bins[x_idxs]
            y_vals = self.range_bins[y_idxs]

            return np.column_stack((x_vals,y_vals))
        else:
            return np.empty(shape=(0,2))
    
    def _get_points_from_pc_grids(
            self,
            pc_grids:np.ndarray
    )->np.ndarray:
        """Obtain a point cloud from a current pc grid

        Args:
            pc_grid (np.ndarray): MxNxN array of M point cloud grids with indicies based on
                self.range_bins where 1 indicates a point is at that location
                and 0 indicates no point is at that location

        Returns:
            np.ndarray: Nx2 array of points in the initial sensor frame
        """

        #convert the grid to a point cloud
        valid_idxs = np.sum(pc_grids,axis=0)
        x_idxs,y_idxs = np.nonzero(valid_idxs)

        if x_idxs.shape[0] > 0:

            x_vals = self.range_bins[x_idxs]
            y_vals = self.range_bins[y_idxs]

            return np.column_stack((x_vals,y_vals))
        else:
            return np.empty(shape=(0,2))

    def _change_pc_reference_frame(
        self,
        initial_heading_rad:float,
        initial_pose_m:np.ndarray,
        current_heading_rad:float,
        current_pose_m:np.ndarray,
        current_points:np.ndarray
    )->np.ndarray:
        """Change the reference frame of a given point cloud

        Args:
            initial_heading_rad (float): the initial heading in radians
                from the perspective of the "global" frame
            initial_pose_m (np.ndarray): [x,y] initial position
                from the perspective of the "global" frame
            current_heading_rad (float): the current heading in radians
                from the perspective of the "global" frame
            current_pose_m (np.ndarray): [x,y] current position
                from the perspective of the "global" frame
            current_points (np.ndarray): Nx2 [x,y] point cloud in the initial 
                reference frame.

        Returns:
            np.ndarray: Nx2 [x,y] point cloud in the new reference frame
        """

        if current_points.shape[0] > 0:

            #get rot/trans from initial sensor frame (at current position) to global
            R_init_to_global = rotation_functions.get_rot_matrix(initial_heading_rad)

            #(R from global -> current sensor frame is inverse of sens -> global)
            R_global_to_curr = rotation_functions.get_rot_matrix(current_heading_rad)

            #compute transformation from current -> initial (in sensor frame)
            R = R_init_to_global.T @ R_global_to_curr
            trans = (initial_pose_m - current_pose_m) @ R_global_to_curr

            #apply the rotation and translation
            return (current_points @ R) + trans

        else:
            return np.empty(shape=(0,2))


    def _add_points_to_grid(
            self,
            grid:np.ndarray,
            current_points:np.ndarray):
        """Add a given point cloud to a specified grid

        Args:
            grid (np.ndarray): MxM pc grid based on the range bins in the 
                initial sensor frame
            current_points (np.ndarray): Nx2 [x,y] point cloud points captured
                from the current sensor frame
        """

        if current_points.shape[0] > 0:

            #get rot/trans from sensor frame (at current position) to global
            R_cur_to_global = rotation_functions.get_rot_matrix(self.current_heading_rad)

            #get rot/trans from global to initial global pose

            #(R from global -> initial sensor frame is inverse of sens -> global)
            R_global_to_init = rotation_functions.get_rot_matrix(self.initial_heading_rad)

            #compute transformation from current -> initial (in sensor frame)
            R = R_cur_to_global.T @ R_global_to_init
            trans = (self.current_pose_m - self.initial_pose_m) @ R_global_to_init

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

            grid[0,x_idx,y_idx] = 1

        return


    ####################################################################
    # Functions used to access the point cloud from other classes
    # TODO: Fix these
    ####################################################################

    def get_points(self)->np.ndarray:
        """Obtain the currently stacked point cloud in the current sensor frame
        (translates points from the initial position to the current position)

        Returns:
            np.ndarray: Nx2 array of points in the current sensor frame
        """

        #convert the grid to a point cloud
        x_idxs,y_idxs = np.nonzero(self.pc_grid_static)

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
        x_idxs,y_idxs = np.nonzero(self.pc_grid_static)

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