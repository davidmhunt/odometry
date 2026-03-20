import numpy as np
from sklearn.neighbors import NearestNeighbors

from odometry.point_cloud_processing.clustering.occlusion_aware_clustering import OcclusionAwareClustering


class TwoStageOcclusionAwareClustering(OcclusionAwareClustering):
    """Adds an additional filtering stage to the OcclusionAwareClustering class.
    
    This class performs a KNN pre-clustering stage to remove points that are too far
    from any other points before performing the occlusion-aware clustering.
    """

    def __init__(
            self,
            knn_neightbors: int = 10,
            knn_distance_threshold: float = 0.1,
            clustering_eps: float = 0.3,
            clustering_min_samples: int = 10,
            angle_res_rad: float = 0.017,
            occlusion_threshold: float = 0.7,
            subsample_percentage: float = 0.1
    ) -> None:
        """Initializes the detector with clustering and visibility parameters.

        Args:
            knn_neightbors (int): The number of neighbors to consider for the KNN pre-clustering stage.
            knn_distance_threshold (float): The maximum distance between two samples 
                for one to be considered as in the neighborhood of the other.
            clustering_eps (float): The maximum distance between two samples 
                for one to be considered as in the neighborhood of the other.
            clustering_min_samples (int): The number of samples in a neighborhood 
                for a point to be considered a core point.
            angle_res_rad (float): Resolution of the angular occlusion buffer 
                in radians (~1 degree by default).
            occlusion_threshold (float): Fraction of a cluster (0.0 to 1.0) 
                that must be covered by closer objects to be pruned.
        """
        super().__init__(
            clustering_eps=clustering_eps,
            clustering_min_samples=clustering_min_samples,
            angle_res_rad=angle_res_rad,
            occlusion_threshold=occlusion_threshold,
            subsample_percentage=subsample_percentage
        )

        self.knn_distance_threshold = knn_distance_threshold
        self.knn_clustering = NearestNeighbors(
            n_neighbors=knn_neightbors,
        )

    def _knn_pre_clustering(self, pc_cartesian: np.ndarray):
        """Performs KNN pre-clustering to remove points that are too far
        from any other points.
        
        Args:
            pc_cartesian (np.ndarray): at least Nx2 array of point detections.

        Returns:
            np.ndarray: Filtered point cloud.
        """

        self.knn_clustering.fit(pc_cartesian)
        distances, indices = self.knn_clustering.kneighbors(pc_cartesian)

        # Calculate average distance to the k neighbors for each point
        avg_distances = distances.mean(axis=1)

        # Keep only the points whose average neighbor distance is below the threshold
        filtered_pc_cartesian = pc_cartesian[avg_distances < self.knn_distance_threshold]
        
        return filtered_pc_cartesian
    
    def process(self, pc_cartesian: np.ndarray):
        """Clusters the cloud and prunes occluded objects.

        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.

        Returns:
            tuple: (filtered_points, labels, visible_labels)
                - filtered_points: Only points belonging to visible clusters.
                - labels: The cluster labels for the filtered points.
                - visible_labels: List of unique labels that passed the filter.
        """

        #1. Subsample points
        pc_cartesian = self._subsample_points(pc_cartesian)

        #2. perform knn pre-clustering
        pc_cartesian = self._knn_pre_clustering(pc_cartesian)

        #3. perform occlusion aware clustering
        filtered_points, labels, visible_labels = self._occlusion_aware_clustering(pc_cartesian)

        return filtered_points, labels, visible_labels

    