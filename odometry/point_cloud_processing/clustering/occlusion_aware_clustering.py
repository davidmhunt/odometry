import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geometries.coordinate_systems.coordinate_system_conversions import cartesian_to_spherical

class OcclusionAwareClustering:
    """Detects and filters point cloud clusters based on density and visibility.
    
    This class performs DBSCAN clustering on a 2D/3D point cloud and then 
    implements an angular depth-buffer (occlusion sweep) to remove clusters 
    that are hidden behind closer objects from the sensor's perspective.
    """

    def __init__(
            self,
            clustering_eps: float = 0.3,
            clustering_min_samples: int = 10,
            angle_res_rad: float = 0.017,
            occlusion_threshold: float = 0.7,
            subsample_percentage: float = 1.0,
            remove_occluded: bool = True,
            filter_method: str = "overlap",
    ) -> None:
        """Initializes the detector with clustering and visibility parameters.

        Args:
            clustering_eps (float): The maximum distance between two samples 
                for one to be considered as in the neighborhood of the other.
            clustering_min_samples (int): The number of samples in a neighborhood 
                for a point to be considered a core point.
            angle_res_rad (float): Resolution of the angular occlusion buffer 
                in radians (~1 degree by default).
            occlusion_threshold (float): Fraction of a cluster (0.0 to 1.0) 
                that must be covered by closer objects to be pruned.
            subsample_percentage (float): Percentage of points to subsample 
                before clustering.
            remove_occluded (bool): Whether to remove occluded points after clustering.
            filter_method (str): Method used to remove occluded points. Options 
                are "overlap" (default) or "ray_trace".
        """
        self.clusterer = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)
        self.angle_res_rad = angle_res_rad
        self.occlusion_threshold = occlusion_threshold
        self.scaler = StandardScaler()
        self.subsample_percentage = subsample_percentage
        self.remove_occluded = remove_occluded
        self.filter_method = filter_method

    def _get_spherical_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Converts input points to 3D spherical coordinates [r, theta, phi]."""
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((points.shape[0], 1))])
        return cartesian_to_spherical(points)

    def _get_cluster_angular_bounds(self, thetas: np.ndarray, rs: np.ndarray) -> dict:
        """Calculates angular span and min distance, handling pi/-pi wrap-around."""
        ref_angle = thetas[0]
        relative_thetas = (thetas - ref_angle + np.pi) % (2 * np.pi) - np.pi
        return {
            'dist': np.min(rs),
            't_min': np.min(relative_thetas) + ref_angle,
            't_max': np.max(relative_thetas) + ref_angle
        }

    def _get_occlusion_indices(self, t_min: float, t_max: float, num_bins: int) -> np.ndarray:
        """Maps angular bounds to discrete indices in the 1D depth buffer."""
        idx_start = int(((t_min + np.pi) / (2 * np.pi)) * num_bins)
        idx_end = int(((t_max + np.pi) / (2 * np.pi)) * num_bins)

        if idx_start <= idx_end:
            indices = np.arange(idx_start, idx_end + 1)
        else:
            indices = np.concatenate([
                np.arange(idx_start, num_bins), 
                np.arange(0, idx_end + 1)
            ])
        return indices % num_bins

    def _cluster_points(self, pc_cartesian: np.ndarray):
        """Performs standard DBSCAN clustering and filters out noise points.
        
        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.
            
        Returns:
            tuple: (filtered_points, labels, visible_labels)
                - filtered_points: Points that are not classified as noise (label != -1).
                - labels: The cluster labels for the filtered points.
                - visible_labels: List of unique labels that passed the filter.
        """
        if pc_cartesian.shape[0] == 0:
            return np.empty((0, pc_cartesian.shape[1])), np.array([]), []

        scaled_points = self.scaler.fit_transform(pc_cartesian)
        full_labels = self.clusterer.fit_predict(scaled_points)
        
        valid_mask = full_labels != -1
        filtered_points = pc_cartesian[valid_mask]
        labels = full_labels[valid_mask]
        
        visible_labels_arr = np.unique(labels)
        
        return filtered_points, labels, visible_labels_arr.tolist()

    def _occlusion_aware_clustering(self, pc_cartesian: np.ndarray):
        """Clusters the cloud and prunes occluded objects.

        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.

        Returns:
            tuple: (filtered_points, labels, visible_labels)
                - filtered_points: Only points belonging to visible clusters.
                - labels: The cluster labels for the filtered points.
                - visible_labels: List of unique labels that passed the filter.
        """
        if pc_cartesian.shape[0] == 0:
            return np.empty((0, pc_cartesian.shape[1])), np.array([]), []

        # 1. Clustering
        pc_filtered, labels, unique_labels = self._cluster_points(pc_cartesian)
        
        if len(unique_labels) == 0:
            return np.empty((0, pc_cartesian.shape[1])), np.array([]), []

        # 2. Extract Spherical Data
        spherical_points = self._get_spherical_coordinates(pc_filtered[:,0:2])

        cluster_list = []
        for label in unique_labels:
            mask = (labels == label)
            bounds = self._get_cluster_angular_bounds(
                thetas=spherical_points[mask, 1], 
                rs=spherical_points[mask, 0]
            )
            bounds['label'] = label
            cluster_list.append(bounds)

        # 3. Visibility Filter (Closest first)
        cluster_list.sort(key=lambda x: x['dist'])
        num_bins = int((2 * np.pi) / self.angle_res_rad)
        angular_depth_buffer = np.zeros(num_bins, dtype=bool)
        visible_labels = []

        for cluster in cluster_list:
            indices = self._get_occlusion_indices(cluster['t_min'], cluster['t_max'], num_bins)
            if len(indices) == 0: continue

            if np.mean(angular_depth_buffer[indices]) < self.occlusion_threshold:
                visible_labels.append(cluster['label'])
                angular_depth_buffer[indices] = True

        # 4. Prepare Output
        visibility_mask = np.isin(labels, visible_labels)
        return (
            pc_filtered[visibility_mask], 
            labels[visibility_mask], 
            visible_labels
        )

    def _ray_trace_filtering(self, pc_cartesian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filters point cloud by retaining only the closest points per angular bin.

        Converts the point cloud to spherical coordinates, discretizes the azimuthal
        angle (theta) into bins based on `angle_res_rad`, and retains only the point
        with the minimum distance (r) in each non-empty bin.

        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.

        Returns:
            tuple:
                - np.ndarray: Filtered point cloud containing only the closest points.
                - np.ndarray: Indices of the closest points in the original point cloud.
        """
        if pc_cartesian.shape[0] == 0:
            return pc_cartesian, np.array([], dtype=int)

        spherical_points = self._get_spherical_coordinates(pc_cartesian[:, 0:2])
        rs = spherical_points[:, 0]
        thetas = spherical_points[:, 1]

        num_bins = int((2 * np.pi) / self.angle_res_rad)
        bin_indices = (((thetas + np.pi) / (2 * np.pi)) * num_bins).astype(int)
        bin_indices = bin_indices % num_bins

        sort_idx = np.argsort(rs)
        sorted_bins = bin_indices[sort_idx]

        _, unique_idx = np.unique(sorted_bins, return_index=True)
        closest_point_indices = sort_idx[unique_idx]

        return pc_cartesian[closest_point_indices], closest_point_indices

    def _ray_trace_occlusion_clustering(self, pc_cartesian: np.ndarray):
        """Applies ray-tracing filtering before clustering the point cloud.

        This acts as an alternative to `_occlusion_aware_clustering`. Points are
        first filtered via `_ray_trace_filtering` to remove occluded points,
        followed by standard DBSCAN clustering.

        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.

        Returns:
            tuple: (filtered_points, labels, visible_labels)
                - filtered_points: Only points belonging to visible clusters.
                - labels: The cluster labels for the filtered points.
                - visible_labels: List of unique labels that passed the filter.
        """
        if pc_cartesian.shape[0] == 0:
            return np.empty((0, pc_cartesian.shape[1])), np.array([]), []

        points, labels, visible_labels = self._cluster_points(pc_cartesian)

        # pc_filtered, closest_point_indices = self._ray_trace_filtering(points)

        pc_filtered, closest_point_indices = self._ray_trace_filtering_with_threshold(
            points,
            threshold=0.1)

        
        return pc_filtered, labels[closest_point_indices], visible_labels

    def _ray_trace_filtering_with_threshold(
        self,
        pc_cartesian: np.ndarray,
        threshold: float = 0.1):
        """
        Args:
            pc_cartesian (np.ndarray): Nx2 or Nx3 array of point detections.
            threshold (float): Threshold depth for filtering points.
        Returns:
            tuple: (filtered_points, closest_point_indices)
                - filtered_points: Only points belonging to visible clusters.
                - closest_point_indices: Indices of the closest points in the original point cloud.
        """
        if pc_cartesian.shape[0] == 0:
            return pc_cartesian, np.array([], dtype=int)

        # 1. Coordinate conversion & Binning
        spherical_points = self._get_spherical_coordinates(pc_cartesian[:, 0:2])
        rs, thetas = spherical_points[:, 0], spherical_points[:, 1]
        num_bins = int((2 * np.pi) / self.angle_res_rad)
        bin_indices = (((thetas + np.pi) / (2 * np.pi)) * num_bins).astype(int) % num_bins

        # 2. Sort by bin (primary) and distance (secondary)
        # This groups bins together and orders them by proximity
        sort_idx = np.lexsort((rs, bin_indices))
        sorted_bins = bin_indices[sort_idx]
        sorted_rs = rs[sort_idx]

        # 3. Find the minimum distance for each bin
        # unique_idx points to the first (closest) point in each bin group
        _, unique_idx = np.unique(sorted_bins, return_index=True)
        
        # Broadcast the minimum distances back to the shape of sorted_rs
        # We create an array where every point knows the r_min of its bin
        min_rs_per_point = np.repeat(sorted_rs[unique_idx], np.diff(np.append(unique_idx, len(sorted_rs))))

        # 4. Filter by the threshold
        mask = sorted_rs <= (min_rs_per_point + threshold)
        final_indices = sort_idx[mask]

        return pc_cartesian[final_indices], final_indices

    # def _subsample_points(self, pc_cartesian: np.ndarray):
    #     """Subsamples the point cloud."""
    #     num_points = int(pc_cartesian.shape[0] * self.subsample_percentage)
    #     indices = np.random.choice(pc_cartesian.shape[0], num_points, replace=False)
    #     return pc_cartesian[indices]

    def _subsample_points(self, pc_cartesian: np.ndarray, grid_size = 1.0) -> np.ndarray:
        """
        Subsamples the point cloud using Spatially Stratified Random Sampling 
        to preserve density while guaranteeing even spatial coverage.
        """
        num_points = pc_cartesian.shape[0]
        target_count = int(num_points * self.subsample_percentage)
        
        if target_count == 0 or target_count >= num_points:
            return pc_cartesian

        # 1. Quantize coordinates to create coarse spatial bins (e.g., 2-meter grids)
        # This groups nearby points into the same logical bucket
        x_bins = np.floor(pc_cartesian[:, 0] / grid_size)
        y_bins = np.floor(pc_cartesian[:, 1] / grid_size)
        
        # 2. Sort the array spatially. 
        # lexsort sorts by y_bins, then x_bins, then the actual y coordinate.
        # Now, points next to each other in the array are physically next to each other in 3D space.
        sort_indices = np.lexsort((pc_cartesian[:, 1], x_bins, y_bins))
        sorted_pc = pc_cartesian[sort_indices]
        
        # 3. Stratified Sampling Step
        # Divide the array into `target_count` chunks and pick a random point within each chunk.
        step = num_points / target_count
        
        # Generate random offsets [0.0 to 1.0) and scale them to the chunk step size
        random_offsets = np.random.rand(target_count) * step
        
        # Calculate the exact array indices to pull
        base_indices = np.arange(target_count) * step
        sample_indices = np.floor(base_indices + random_offsets).astype(np.int32)
        
        # Failsafe clip to ensure no index out of bounds due to floating point math
        sample_indices = np.clip(sample_indices, 0, num_points - 1)
        
        # Return the systematically sampled points
        return sorted_pc[sample_indices]

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

        #2. perform clustering (with or without occlusion filter)
        if self.remove_occluded:
            if self.filter_method == "ray_trace":
                filtered_points, labels, visible_labels = self._ray_trace_occlusion_clustering(pc_cartesian)
            else:
                filtered_points, labels, visible_labels = self._occlusion_aware_clustering(pc_cartesian)
        else:
            filtered_points, labels, visible_labels = self._cluster_points(pc_cartesian)

        return filtered_points, labels, visible_labels