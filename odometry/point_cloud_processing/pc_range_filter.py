import numpy as np

class pcRangeFilter:

    def __init__(self,
                 min_detection_radius_m=0.5,
                 max_detection_radius_m=20):
        
        self.min_detection_raduis_m = min_detection_radius_m
        self.max_detection_radius_m = max_detection_radius_m

    def get_points_in_detection_range(self,points:np.ndarray):
        """Removes specific detection where the sensor has detected itself
        and not an object in the environment

        Args:
            current_points (np.ndarray): Nx2 numpy array containing
             at least the [x,y] coordinates of radar detections

        Returns:
            np.ndarray: current detection list without the points where
            the sensor detected itself
        """

        distances = np.linalg.norm(points,axis=1)

        invalid_idxs_too_short = distances < self.min_detection_raduis_m
        invalid_idxs_too_long = distances > self.max_detection_radius_m
        invalid_idxs = np.logical_or(invalid_idxs_too_long,invalid_idxs_too_short)

        return points[~invalid_idxs,:]

