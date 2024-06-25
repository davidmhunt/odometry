import numpy as np


class VelFiltering:

    def __init__(self,v_thresh:float=1.0) -> None:
        
        self.v_thresh:float = v_thresh
        
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
    




