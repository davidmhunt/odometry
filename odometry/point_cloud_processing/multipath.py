import numpy as np
from sklearn.cluster import DBSCAN

from odometry.supportFns import coordinate_systems
from odometry.supportFns import rotation_functions

class MultiPath:
    """Implements method of dealing with multipath using a simple ray tracing
    and clustering algorithm
    """

    def __init__(
            self,
            clustering_eps:float = 0.25,
            clustering_min_samples:int = 10) -> None:
        """_summary_

        Args:
            clustering_eps (float, optional): The maximum distance between
                a point and a cluster for it to be included in the cluster.
                Defaults to 0.25
            clustering_min_samples (int,optional): The minimum number of 
                points in a cluster for it to be a valid cluster. 
                Defaults to 10
        """

        #initialize the DBSCAN clustering algorithm
        self.dbscan_clusterer:DBSCAN = DBSCAN(
            eps = clustering_eps,
            min_samples=clustering_min_samples
        )
        return
    
    def _cluster_points(
            self,
            point_cloud:np.ndarray
    )->np.ndarray:
        """Cluster a given point cloud

        Args:
            point_cloud (np.ndarray): Nx2 point cloud in cartesian coordinates

        Returns:
            np.ndarray: The clustered labels for each points, -1 means no 
            cluster associated with a point
        """

        return self.dbscan_clusterer.fit_predict(point_cloud)
    
    def remove_multipath(
            self,
            pc_cartesian:np.ndarray
    )->np.ndarray:
        """Removes multipath detections from a radar point cloud

        Args:
            pc_cartesian (np.ndarray): Nx2 point cloud in cartesian coordinates

        Returns:
            np.ndarray: Nx2 array of points with multipath detections removed
        """
        
        #perform initial clustering in cartesian
        labels = self._cluster_points(pc_cartesian)

        #convert to polar
        pc_polar = coordinate_systems.cartesian_to_polar(pc_cartesian)

        #enable tracking for labels and valid points
        unique_labels = np.unique(labels)[1:] #remove the -1 label
        valid_points = np.ones(pc_polar.shape[0],dtype=bool)

        #handle points already not in a cluster
        valid_points[labels==-1] = 0

        #for each cluster determine the range of its closest point
        cluster_ranges = np.array([
            np.min(pc_polar[labels==cluster_id,0]) for \
            cluster_id in unique_labels
        ])

        #sort the clusters by their ranges
        sorted_idxs = np.argsort(cluster_ranges)

        for i in range(len(sorted_idxs)):

            #get the cluster points
            cluster_idx = sorted_idxs[i]
            cluster_points = pc_polar[labels==unique_labels[cluster_idx]]

            #get the maximum range and angle bounds of the cluster
            angle_bounds = [np.min(cluster_points[:,1]),np.max([cluster_points[:,1]])]
            max_range = np.max(cluster_points[:,0])

            #handle special case where points are clustered around -pi,pi
            if angle_bounds[1] - angle_bounds[0] > np.pi: #assume no cluster occupies more than +pi
                #determine the correct bounds

                #set the positive bound
                angle_bounds[1] = np.min(cluster_points[cluster_points[:,1] > 0,1])
                #set the negative bound
                angle_bounds[0] = np.max(cluster_points[cluster_points[:,1] < 0,1])

                #identify indicies of points that are past these points
                invalid_idxs = \
                    (pc_polar[:,0] > max_range) & \
                    (pc_polar[:,1] > angle_bounds[1]) & \
                    (pc_polar[:,1] < angle_bounds[0])
            else: #standard case
                invalid_idxs = \
                    (pc_polar[:,0] > max_range) & \
                    (pc_polar[:,1] < angle_bounds[1]) & \
                    (pc_polar[:,1] > angle_bounds[0])
            
            #mark invalid points as invalid
            valid_points[invalid_idxs] = 0
        
        #perform a final round of clustering to only keep the clustered points
        valid_stacked_pc_cart = pc_cartesian[valid_points,:]
        labels = self._cluster_points(valid_stacked_pc_cart)
        
        return valid_stacked_pc_cart[labels!=-1,:]
    

    

