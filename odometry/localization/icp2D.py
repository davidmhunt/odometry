import numpy as np
from sklearn.neighbors import NearestNeighbors
from odometry.supportFns import rotation_functions
from odometry.localization._localizer import _Localizer

from odometry.point_cloud_processing.pc_range_filter import pcRangeFilter

class icp2D(_Localizer):
    """Class that performs icp point cloud matching and computes the optimal rotation
    and translation between two point clouds
    """

    def __init__(self, 
                 icp_matching_distance_threshold = 1.0,
                 icp_best_points_percentile=60,
                 icp_convergence_translation_threshold = 1e-3,
                 icp_convergence_rotation_threshold=1e-4,
                 icp_point_pairs_threshold = 4,
                 icp_max_iterations = 20,
                 self_detection_radius_m = 0.25):
                    
        #icp and related parameters
        self.ground_detection_filtering:pcRangeFilter = \
            pcRangeFilter(self_detection_radius_m)
        
        #icp paramters
        self.icp_matching_distance_threshold = icp_matching_distance_threshold
        self.icp_best_points_percentile = icp_best_points_percentile
        self.icp_convergence_rotation_threshold = icp_convergence_rotation_threshold
        self.icp_convergence_translation_threshold = icp_convergence_translation_threshold
        self.icp_point_pairs_threshold = icp_point_pairs_threshold
        self.icp_max_iterations = icp_max_iterations

        super().__init__()
    
    ####################################################################
    #Odometry support functions
    ####################################################################

    def get_nearerest_points(self,current_points,reference_points):
        """Get nearest points in the sensor's reference frame for a given set of
            detections

        Args:
            current_points (np.ndarray): Nx2 array of points (detections)
                to be matched to the previous points (in sensor frame)
            reference_points (np.ndarray): Nx2 array of points for reference
             (i.e the previous frame's points) (in sensor frame)

        Returns:
            (np.ndarray,np.ndarray): current_matched,reference_matched
        """
        nbrs = NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(reference_points)

        distances,indicies = nbrs.kneighbors(current_points)

        #get the reference points that are nearest to the current points
        reference_matched = reference_points[indicies[:,0]]

        #filter reference and current points by distance threshold
        reference_matched = reference_matched[distances[:,0] <= self.icp_matching_distance_threshold]
        current_matched = current_points[distances[:,0] <= self.icp_matching_distance_threshold]

        return current_matched,reference_matched
    
    def get_nearerest_points_percentile(self,
                                 current_points:np.ndarray,
                                 reference_points:np.ndarray):
        """Get the nearest points in the map for a given set of detections

        Args:
            current_points (np.ndarray): Nx2 array of points (detections)
                to be matched to the map points (in global frame)
            reference_points (np.ndarray): Nx2 array of points for the global
                map (in global frame)

        Returns:
            (np.ndarray,np.ndarray): current_matched,reference_matched
        """

        nbrs = NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(reference_points)

        distances,indicies = nbrs.kneighbors(current_points)

        #get the reference points that are nearest to the current points
        reference_matched = reference_points[indicies[:,0]]

        p = np.percentile(distances[:,0],self.icp_best_points_percentile)

        reference_matched = reference_matched[
            (distances[:,0] <= p) | 
            (distances[:,0] <= self.icp_matching_distance_threshold)]
        current_matched = current_points[
            (distances[:,0] <= p) | 
            (distances[:,0] <= self.icp_matching_distance_threshold)]

        return current_matched,reference_matched
    
    def remove_sensor_self_detections(self,points:np.ndarray):
        """Removes specific detection where the sensor has detected itself
        and not an object in the environment

        Args:
            current_points (np.ndarray): numpy array containing the current list of detections

        Returns:
            np.ndarray: current detection list without the points where
            the sensor detected itself
        """
        return self.ground_detection_filtering.get_points_in_detection_range(points)

    
    def compute_optimal_rot_trans(self,points:np.ndarray,reference:np.ndarray):
        """Compute the ideal rotation and translation to minimize the distance between
        two sets of points

        Args:
            points (np.ndarray): Nx2 array of the points to compute
                the rotation and translation matrix for
            reference (np.ndarray): Nx2 array of a set of reference points

        Returns:
            rot_angle_rad,trans: the optimal rotation and translation in x/y 
        """

        x_mean = np.mean(points,axis=0)
        x_ref_mean = np.mean(reference, axis=0)

        n = points.shape[0]
        if n == 0:
            return None, None

        #compute covariance matrix
        H = (points - x_mean).T @ (reference - x_ref_mean)

        #compute SVD to determine optimal rotation matrix
        U, S, Vh = np.linalg.svd(H,full_matrices=True)

        #check the determinant and correct it if its negative
        d = np.linalg.det(Vh.T @ U.T)
        if d < 0:
            S[-1] = -S[-1]
            U[:,-1] = -U[:,-1]
        
        #get the optimal rotation matrix
        R = Vh.T @ U.T

        #get the angle from the rotation matrix
        rot_angle_rad = rotation_functions.get_angle_from_rot_matrix(R)

        #get the optimal translation matrix
        trans = x_ref_mean - (x_mean @ R.T)

        return rot_angle_rad, trans  
    
    def icp(self,
            current_points:np.ndarray,
            reference_points:np.ndarray,
            estimated_heading_rad,
            estimated_pose_m:np.ndarray):
        """Use the ICP method to compute a new position and heading
        using the current detections, a global map, and estimates of
        the current heading and position

        Args:
            current_points (np.ndarray): Nx2 array of the current 
                detections (ex: in sensor's reference frame)
            reference (np.ndarray): Nx2 array of global map as a 
                point cloud (ex: in global reference frame)
            estimated_heading_rad (_type_): estimated heading of the
                sensor in radians
            estimated_pose_m (_type_): estimated position (in global
                reference frame) of the sensor

        Returns:
            (double, np.ndarray): new_heading_rad,new_pose_m - the new
                heading and position computed using the ICP method.
                returns None,None if icp odometry update was unsuccessful
        """
        #initialize rotation and translation matrix
        new_heading_rad = estimated_heading_rad
        new_pose_m = estimated_pose_m

        #flag to confirm that icp was performed
        icp_performed = False
        
        for iter in range(self.icp_max_iterations):

            #translate current points into map reference frame:
            aligned_points = rotation_functions.apply_rot_trans(
                points=current_points,
                rot_angle_rad=new_heading_rad,
                trans=new_pose_m
            )

            #filter out points outside of the map
            aligned_points = self.remove_detections_outside_of_map(
                points=aligned_points
            )
            if aligned_points.shape[0] == 0:
                break

            #get nearest neighbors
            current_matched, map_matched = self.get_nearerest_points_percentile(
                current_points=aligned_points,
                reference_points=reference_points
            )

            #save the currently valid points from ICP 
            self.current_valid_points = current_matched

            #check to make sure that there are enough points
            if current_matched.shape[0] < self.icp_point_pairs_threshold:
                #not enough points to perform icp, just return current estimate
                break

            #get the ideal rotation and translation
            rot_angle_rad, trans_m = self.compute_optimal_rot_trans(
                points=current_matched,
                reference=map_matched
            )
            
            #update the heading
            new_heading_rad = new_heading_rad + rot_angle_rad

            #get a rotation matrix for to apply to the current position
            R = rotation_functions.get_rot_matrix(rot_angle_rad)

            #compute the new position
            new_pose_m = (new_pose_m @ R.T) + trans_m

            #set the icp_performed flag
            icp_performed = True

            #check to see if it meets the threshold
            if (abs(rot_angle_rad) < self.icp_convergence_rotation_threshold) \
                and (abs(trans_m[0]) < self.icp_convergence_translation_threshold) \
                and (abs(trans_m[1]) < self.icp_convergence_translation_threshold):

                break
        
        if icp_performed:
            return new_heading_rad,new_pose_m
        else:
            return None,None

    ####################################################################
    #Exporting extra data
    ####################################################################
    def get_current_valid_detections(self) -> np.ndarray:
        
        return self.current_valid_points