import numpy as np

def get_rot_matrix(rot_angle_rad):
    """Get a rotation matrix for a given rotation angle

    Args:
        rot_angle_rad (double): the desired rotation angle in radians

    Returns:
        np.ndarray: a 2x2 rotation matrix for the desired rotation
    """
    return np.array([
        [np.cos(rot_angle_rad), -1.0 * np.sin(rot_angle_rad)],
        [np.sin(rot_angle_rad), np.cos(rot_angle_rad)]
    ])

def get_angle_from_rot_matrix(rot_matrix:np.ndarray):
    """Returns the rotation angle from a given rotation matrix

    Args:
        rot_matrix (np.ndarray): the rotation matrix

    Returns:
        np.float64: the rotation angle in radians
    """
    return np.arctan2(rot_matrix[1,0],rot_matrix[0,0])

def apply_rot_trans(points:np.ndarray,rot_angle_rad,trans):
        """Apply a rotation and translation to a set of points using
        the formulat out = (points * R.T) + trans

        Args:
            points (np.ndarray): Nx2 array of points to be transformed
            rot_angle_rad (_type_): the rotation angle in radians to rotate
                the points by
            trans (_type_): the [x,y] translation vector to apply

        Returns:
            np.ndarray: An Nx2 array of the transformed points
        """

        # transform 'points' (using the calculated rotation and translation)
        R = get_rot_matrix(rot_angle_rad)
        
        #apply the rotation matrix and translation
        return (points @ R.T) + trans    

def wrap_heading(heading_rad):
    """wraps the heading to be between [-pi,pi]

    Args:
        heading_rad (_type_): the heading in radians

    Returns:
        _type_: the wrapped heading in radians
    """
    #implement wrapping around when abs(heading) > pi
    if np.abs(heading_rad) > np.pi:
        return (-1 * np.sign(heading_rad)
            * (2 * np.pi - np.abs(heading_rad))
        ) 

    else:
        return heading_rad