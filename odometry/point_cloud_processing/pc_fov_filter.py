import numpy as np
from geometries.coordinate_systems.coordinate_system_conversions import cartesian_to_spherical

class pcFovFilter:

    def __init__(
        self,
        valid_fovs_deg: list[tuple[float, float]] = [(-180,180)]
    ):
        """
        Initializes the Field of View filter.

        Args:
            valid_fovs (list of tuples): A list of valid FOVs in degrees, e.g. [(-60, 60)].
                0 degrees is the +x axis, +90 degrees is the +y axis.
        """
        if valid_fovs_deg is None:
            valid_fovs_deg = [(-180.0, 180.0)]
            
        self.valid_fovs_deg = valid_fovs_deg

    def get_points_in_fov(self, points: np.ndarray) -> np.ndarray:
        """
        Filters points that fall within the valid FOVs.

        Args:
            points (np.ndarray): Nx2 or Nx3 numpy array containing at least 
             the [x, y] coordinates of points.

        Returns:
            np.ndarray: Filtered points that fall within ANY of the specified FOVs.
        """
        if len(points) == 0:
            return points

        points_3d = points.copy()
        if points_3d.shape[1] == 2:
            points_3d = np.hstack((points_3d, np.zeros((points_3d.shape[0], 1))))
        elif points_3d.shape[1] > 3:
            points_3d = points_3d[:, :3]

        spherical_pts = cartesian_to_spherical(points_3d)
        theta_rad = spherical_pts[:, 1]
        theta_deg = np.rad2deg(theta_rad)

        valid_mask = np.zeros(len(points), dtype=bool)

        for min_deg, max_deg in self.valid_fovs_deg:
            if min_deg <= max_deg:
                current_fov_mask = (theta_deg >= min_deg) & (theta_deg <= max_deg)
            else:
                # Wrap around case (e.g. min_deg = 150, max_deg = -150)
                current_fov_mask = (theta_deg >= min_deg) | (theta_deg <= max_deg)
            
            valid_mask = np.logical_or(valid_mask, current_fov_mask)

        return points[valid_mask, :]
