import numpy as np
from odometry.point_cloud_processing.vel_filtering import VelFiltering
from odometry.supportFns.rotation_functions import apply_rot_trans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin2D
from avstack.modules.perception.detections import CentroidDetection
from avstack.modules.tracking import BasicXyTracker
from collections import defaultdict



class DynamicObjectTracker:

    def __init__(
            self,
            vel_filtering_v_thresh:float = 0.02,
            dbscan_eps=0.2,
            dbscan_min_samples=20
            ) -> None:
        
        #vel filtering initialization
        self.vel_filtering:VelFiltering = VelFiltering(
            v_thresh=vel_filtering_v_thresh
        )

        #dynamic points clustering
        self.dbscan_dynamic_clustering:DBSCAN = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
        
        #current list of dynamic objects (x,y coordinates)
        #TODO: update the 3D
        # self.current_dynamic_detections:np.ndarray = np.empty()
        self.current_dynamic_detections:np.ndarray = None
        
        # self.history_dynamic_objects:np.ndarray = np.empty()
        self.history_dynamic_objects:list = None
        
        self.history_clustered_dynamic_clusters_nums:list = None
        self.history_clustered_dynamic_centroids:list = None
        
        self.track_history_full = defaultdict(list)  # track_id -> list of [frame_id, x, y]
        self.track_history = defaultdict(list)       # track_id -> list of [frame_id, x, y] (active only)
        self.current_tracks = []                     # list of (track_id, [x, y])
        
        self.xy_tracker = None  # initialize later


    def reset(self,
            n:int):
        """_summary_

        Args:
            n (int): number of total history frames
        """
        #reset the pose histories
        self.current_dynamic_detections:np.ndarray = np.zeros(shape=(n,2),dtype=np.double)
        self.history_dynamic_objects = list()
        self.history_clustered_dynamic_clusters_nums = list()
        self.history_clustered_dynamic_centroids = list()
        
        self.current_tracks = np.zeros(shape=(n,2),dtype=np.double)
        #TODO: update the tracks
        pass

    def update(
            self,
            current_points:np.ndarray,
            ego_vel:np.ndarray,
            ego_heading_rad:np.ndarray,
            ego_pose_m:np.ndarray
    ):
        """_summary_

        Args:
            current_points (np.ndarray): Nx4 array of detections with [x,y,z,vel] info
                in the sensor coordinate frame
            ego_vel (np.ndarray): Nx2 array corresponding to the velocity 
                of the ego vehicle as [x,y]
            ego_heading_rad (np.ndarray): heading of the ego vehicle in the global frame
                in radians
            ego_pose_m (np.ndarray): [x,y] pose of the ego vehicle in the global frame
                in meters 
        """
        
        #get the dynamic detections
        dynamic_points = self.vel_filtering.get_dynamic_detections(
            detections=current_points,
            ego_vel=ego_vel
        )



        #move the detections into the global coordinate frame
        self.current_dynamic_detections = apply_rot_trans(
            points=dynamic_points[:,0:2],
            rot_angle_rad=ego_heading_rad,
            trans=ego_pose_m
        )


        return
    
    def history_update_dynamic_objects(
            self,
            current_points:np.ndarray,
            cluster_nums:int,
            centroids:np.ndarray
    ):
        """_summary_

        Args:
            current_points (np.ndarray): np.ndarray: An Nx2 array of the transformed points
        """
        self.history_dynamic_objects.append(current_points)
        self.history_clustered_dynamic_clusters_nums.append(cluster_nums)
        self.history_clustered_dynamic_centroids.append(centroids)
        #TODO: update the tracks
        
        
    '''Dynamic Point Cloud Clustering'''
    def dynamic_point_cloud_clustering(self,
                                    current_points:np.ndarray):
        """
        Applies DBSCAN clustering to the given N*2 dynamic point cloud.

        Parameters:
        - points: np.ndarray, shape (N, 2)
            The input point cloud (N points, each with 2 features: x, y).

        Returns:
        - labels: np.ndarray, shape (N,)
            Cluster labels for each point in the dataset. Noise points are labeled as -1.
        - num_clusters: int
            The number of clusters found (excluding noise points).
        - clusters: dict
            A dictionary where keys are cluster labels and values are arrays of points in those clusters.
        """
        
        if current_points.shape[0] == 0:
            return 0, {}, {}
        # Apply DBSCAN clustering
        labels = self.dbscan_dynamic_clustering.fit_predict(current_points)
        
        # Count the number of clusters (excluding noise)
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Group points by clusters
        clusters = {label: current_points[labels == label] for label in unique_labels if label != -1}
        
        # # Compute cluster centroids
        # centroids = {label: np.mean(points, axis=0) for label, points in clusters.items()}
        

        # Compute cluster centroids based on KDE density peak
        centroids = {}
        for label, points in clusters.items():
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(points)
            scores = kde.score_samples(points)  # log density
            max_density_idx = np.argmax(scores)
            centroids[label] = points[max_density_idx]


        return num_clusters, clusters, centroids
    
    def init_xy_tracker(
            self,
            threshold_confirmed: float=10,   
            threshold_coast: float=5.0,     
            v_max: float=10.0,             
            assign_metric="center_dist",
            assign_radius=1.0,
            P0:np.ndarray=np.diag([1, 1, 5, 5]) ** 2,
        ) :
        """_summary_

        Args:
            threshold_confirmed (float): Minimum hits to confirm a track
            threshold_coast (float): Maximum frames without updates before deletion
            v_max (float): Maximum allowed velocity
            P0 (_type_): _description_
            assign_radius (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: avstack.modules.tracking.tracker2d.BasicXyTracker
        """
        
        self.xy_tracker = BasicXyTracker(
            threshold_confirmed,  
            threshold_coast,     
            v_max,            
            assign_metric,
            assign_radius,
            P0=P0,
        )

    def update_tracks(
        self, 
        timestamp, 
        i_frame,
        centroids_dict,
        initial_timestamp, 
        noise=np.array([0.5, 0.5])
        ):
        if self.xy_tracker is None:
            raise RuntimeError("xy_tracker not initialized. Please call `init_xy_tracker()` first.")
        
        self.current_tracks.clear()
        
        # current_imu_data = dataset.get_imu_full_data(idx=i_frame)
        # current_timestamp = current_imu_data[-1][0]
        # timestamp = current_timestamp - initial_timestamp

        msmts = DataContainer(
            frame=i_frame,
            timestamp=timestamp,
            source_identifier="sensor",
            data=[],
        )

        for centroid in centroids_dict:
            msmts.append(CentroidDetection(
                data=centroid,
                noise=noise,
                source_identifier="sensor",
                reference=GlobalOrigin2D,
            ))
            
        tracks = self.xy_tracker(
            detections=msmts,
            platform=GlobalOrigin2D,
            check_reference=False,  # if you already enforce consistent reference, set to False
        )

        frame_tracks = []
        for track in tracks:
            track_id = track.ID
            pos = track.position  # [x, y]
            
            # ---- 1. full history
            self.track_history_full[track_id].append([i_frame, pos[0], pos[1]])
            
            if track.active:
                self.track_history[track_id].append([i_frame, pos[0], pos[1]])
            else:
                if track_id in self.track_history:
                    del self.track_history[track_id]
        
            if track.active:
                frame_tracks.append((track_id, pos))

        self.current_tracks = frame_tracks

    def get_current_tracks(self):
        return self.current_tracks

    def get_track_history(self):
        return self.track_history
