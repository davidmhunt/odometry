import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class VelFiltering:

    def __init__(self,
                 v_thresh:float=1.0,
                 min_static_rejection_radius:float = 2.0,
                 dynamic_cluster_eps:float = 1.0,
                 dynamic_cluster_min_samples = 7) -> None:
        
        self.v_thresh:float = v_thresh

        self.dbscan_clusterer:DBSCAN  = DBSCAN(
            eps=dynamic_cluster_eps,
            min_samples=dynamic_cluster_min_samples
        )

        self.min_static_rejection_radius = min_static_rejection_radius
        
        return

    def predict_vels(self,detections: np.ndarray, ego_vel: np.ndarray)->np.ndarray:
        """For a set of detections and ego velocity estimate, predict the
        measured velocity of each detection by the radar

        Args:
            detections (np.ndarray): Nx2 array with the [x,y] 
                coordinate for each detection
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle

        Returns:
            _type_: Nx1 array of predicted velocity measurement for each detection
        """
        #get target positions
        P = detections[:,0:2]

        #compute normalized r vectors (corresponding to angle of the target)
        H = np.divide(P,np.linalg.norm(P, axis=1).reshape(-1,1))

        #estimate the velocity that each detection would have been measured at
        return H @ ego_vel
    
    def square_error_loss(self,v_true:np.ndarray, v_pred:np.ndarray)->np.ndarray:
        """Compute the square error loss between the predicted and actual velocity
        for a set of radar detections

        Args:
            v_true (np.ndarray): Nx1 array or the actual velocity measurement for
                each detection
            v_pred (np.ndarray): Nx1 array of the predicted velocity measurement
                for each detection

        Returns:
            _type_: _description_
        """
        return (v_true - v_pred) ** 2
    
    def mean_square_error(self,v_true:np.ndarray,v_pred:np.ndarray)->np.ndarray:

        return np.sum(
            self.square_error_loss(v_true,v_pred)
        ) / v_true.shape[0]
        
    def get_dynamic_detections(self,detections:np.ndarray, ego_vel:np.ndarray)->np.ndarray:
        """For a set of detections and ego velocity estimate, get the detections corresponding
        to moving objects

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle

        Returns:
            _type_: Nx4 array of detections corresponding to moving objects
        """

        #compute the predicted velocity for each detection
        v_pred = self.predict_vels(detections[:,0:2],ego_vel)

        #compute the square error loss
        errors = self.square_error_loss(detections[:,3],v_pred)

        #identify the errors that excede the threshold
        dynamic_idxs = errors > self.v_thresh

        return detections[dynamic_idxs,:]
    
    def get_static_detections(self,detections:np.ndarray, ego_vel:np.ndarray)->np.ndarray:
        """For a set of detections and ego velocity estimate, get the detections corresponding
        to static objects

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle

        Returns:
            _type_: Nx4 array of detections corresponding to static objects
        """
        #compute the predicted velocity for each detection
        v_pred = self.predict_vels(detections[:,0:2],ego_vel)

        #compute the square error loss
        errors = self.square_error_loss(detections[:,3],v_pred)

        #identify the errors that excede the threshold
        static_idxs = errors < self.v_thresh

        return detections[static_idxs,:]
    
    def remove_dynamic_clusters_from_static_detections_centroids(
            self,
            static_detections:np.ndarray,
            dynamic_detections:np.ndarray)->np.ndarray:


        if dynamic_detections.shape[0] > 0:
            labels = self.dbscan_clusterer.fit_predict(dynamic_detections[:,0:2])
            
            #get the unique cluster ids
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels) -1

            #create a list of valid indicies
            valid_idxs = np.ones(static_detections.shape[0],dtype=bool)

            if num_clusters > 0:
                for cluster_id in unique_labels[1:]:
                    
                    #identify points in the cluster
                    cluster_points = dynamic_detections[labels==cluster_id,0:2]

                    #compute the centroid
                    centroid = np.average(cluster_points,axis=0)

                    #compute distance between centroid and all static objects
                    distances = np.linalg.norm(static_detections[:,0:2] - centroid)

                    moving_idxs = distances < self.min_static_rejection_radius

                    valid_idxs[moving_idxs] = False

                return static_detections[valid_idxs,:]
            else:
                return static_detections


        else:
            return static_detections
        
    def remove_dynamic_clusters_from_static_detections_knn(
            self,
            static_detections:np.ndarray,
            dynamic_detections:np.ndarray)->np.ndarray:


        if dynamic_detections.shape[0] > 0:
            labels = self.dbscan_clusterer.fit_predict(dynamic_detections[:,0:2])
            
            #get the unique cluster ids
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels) -1

            #get only dynamic detections that were clustered
            dynamic_detections = dynamic_detections[labels!=-1,0:2]

            if num_clusters > 0:

                nbrs = NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(dynamic_detections)
                    
                distances,indicies = nbrs.kneighbors(static_detections[:,0:2])

                valid_idxs = distances[:,0] >= self.min_static_rejection_radius

                return static_detections[valid_idxs,:]
            else:
                return static_detections


        else:
            return static_detections
    




