import numpy as np

def cartesian_to_polar(points_cartesian:np.ndarray)->np.ndarray:

    ranges = np.sqrt(points_cartesian[:,0]**2 + points_cartesian[:,1]**2)
    thetas = np.arctan2(points_cartesian[:,1],points_cartesian[:,0])

    return np.column_stack((ranges,thetas))

def polar_to_cartesian(points_polar:np.ndarray)->np.ndarray:

    x = points_polar[:,0] * np.cos(points_polar[:,1])
    y = points_polar[:,0] * np.sin(points_polar[:,1])

    return np.column_stack((x,y))