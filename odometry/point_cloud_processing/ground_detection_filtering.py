import numpy as np

class groundDetectionFiltering:

    def __init__(self,
                 self_detection_radius_m=0.5):
        
        self.self_detection_radius_m = self_detection_radius_m

    def remove_sensor_self_detections(self,points:np.ndarray):
        """Removes specific detection where the sensor has detected itself
        and not an object in the environment

        Args:
            current_points (np.ndarray): Nx2 numpy array containing
             at least the [x,y] coordinates of radar detections

        Returns:
            np.ndarray: current detection list without the points where
            the sensor detected itself
        """

        self_det_rad = self.self_detection_radius_m
        distances = np.linalg.norm(points,axis=1)
        self_detection_idxs = distances < self_det_rad

        return points[~self_detection_idxs,:]

