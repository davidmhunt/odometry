import numpy as np
import pandas as pd
from IPython.display import display
import os

class Analyzer:

    def __init__(self) -> None:
        
        return
    
    def get_percentile(self, distances: np.ndarray, percentile: float):
        """Get the percentile (ex: 90th) for a given set of error values

        Args:
            distances (np.ndarray): numpy array of errors
            percentile (float): desired percentile to compute

        Returns:
            float: the percentile value
        """

        sorted_data = np.sort(np.abs(distances))
        p = 1.0 * np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        # compute the index of the percentile
        idx = (np.abs(p - percentile)).argmin()

        return sorted_data[idx]
    
    def compute_absolute_errors_position(self,
                                 history_position_m:np.ndarray,
                                 history_position_m_gt:np.ndarray)->np.ndarray:
        """Compute the euclidian distance errors between estimated position
        at each frame and the gt position at each frame

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array

        Returns:
            np.ndarray: N element euclidian absolute position error
        """
        return np.linalg.norm(
            history_position_m - history_position_m_gt,
            axis=-1
        )
    
    def compute_relative_errors_position(self,
                                     history_position_m:np.ndarray,
                                     history_position_m_gt:np.ndarray)->np.ndarray:
        """Compute the relative trajectory errors from one frame to the next

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array

        Returns:
            np.ndarray: N-1 element euclidian relative trajectory error
        """
        
        est_relative_trajectories = history_position_m[1:,:] - history_position_m[0:-1,:]
        gt_relative_trajectories = history_position_m_gt[1:,:] - history_position_m_gt[0:-1,:]

        return np.linalg.norm(
            est_relative_trajectories - gt_relative_trajectories,
            axis=-1
        )
    
    def compute_x_y_state_estimate_errors(
            self,
            history_position_m,
            history_position_m_gt
    )->tuple:
        """Compute the x,y position errors between estimated position
        and the gt position at each frame

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array

        Returns:
            tuple: x_errors, y_errors
        """
        
        # compute errors in x
        x_estimated = history_position_m[:, 0]
        x_gt = history_position_m_gt[:, 0]
        x_errors = x_gt - x_estimated

        # compute errors in y
        y_estimated = history_position_m[:, 1]
        y_gt = history_position_m_gt[:, 1]
        y_errors = y_gt - y_estimated

        return x_errors,y_errors
    
    def compute_absolute_errors_heading(
            self,
            history_heading_deg:list,
            history_heading_deg_gt:list
    )->np.ndarray:
        """Compute the heading absolute value of the error (in degrees) 
        between the estimated heading at each frame and the gt heading.

        Args:
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees

        Returns:
            np.ndarray: N element array of heading errors in degrees
        """
        
        estimated_heading_deg = np.array(history_heading_deg)
        gt_heading_deg = np.array(history_heading_deg_gt)

        return np.abs(estimated_heading_deg - gt_heading_deg)
    
    def compute_relative_errors_heading(
            self,
            history_heading_deg:list,
            history_heading_deg_gt:list
    )->np.ndarray:
        """Compute the absolute value of the heading error (in degrees) 
        between the estimated and ground truth relative heading update 
        between subsequent frames.

        Args:
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees

        Returns:
            np.ndarray: heading errors in degrees (absolute value)
        """
        
        estimated_heading_deg = np.array(history_heading_deg)
        gt_heading_deg = np.array(history_heading_deg_gt)

        est_relative_trajectories = estimated_heading_deg[1:] - estimated_heading_deg[:-1]
        gt_relative_trajectories = gt_heading_deg[1:] - gt_heading_deg[:-1]

        return np.absolute(est_relative_trajectories - gt_relative_trajectories)
    
    def compute_total_distance_traveled(self,history_pose_gt:np.ndarray):
        """Compute the total distance for a given trial

        Args:
            history_pose_gt (np.ndarray): Nx2 array of gt positions

        Returns:
            float: total distance traveled
        """
        dist = 0
        for i in range(len(history_pose_gt)-1):
            dist += np.linalg.norm(
                history_pose_gt[i+1] -\
                history_pose_gt[i]
            )

        return dist
    
    def show_summary_statistics(
            self,
            history_position_m:np.ndarray,
            history_position_m_gt:np.ndarray,
            history_heading_deg:list,
            history_heading_deg_gt:list,
            percentile=0.90
    ):
        """Generate a tabular summary of errors between estimates and gt

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees
            percentile (float, optional): Tail error percentile. Defaults to 0.90.
        """
        
        #Absolute position
        absolute_errors_pos = self.compute_absolute_errors_position(
            history_position_m,
            history_position_m_gt
        )
        absolute_errors_pose_mean = np.mean(absolute_errors_pos)
        absolute_errors_pose_var = np.var(absolute_errors_pos)
        absolute_errors_pose_median = np.median(absolute_errors_pos)
        absolute_errors_pose_tail = self.get_percentile(absolute_errors_pos, percentile)

        #relative position
        relative_errors_pos = self.compute_relative_errors_position(
            history_position_m,
            history_position_m_gt
        )
        relative_errors_pose_mean = np.mean(relative_errors_pos)
        relative_errors_pose_var = np.var(relative_errors_pos)
        relative_errors_pose_median = np.median(relative_errors_pos)
        relative_errors_pose_tail = self.get_percentile(relative_errors_pos, percentile)

        #absolute heading
        absolute_errors_heading = self.compute_absolute_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        absolute_errors_heading_mean = np.mean(absolute_errors_heading)
        absolute_errors_heading_variance = np.var(absolute_errors_heading)
        absolute_errors_heading_median = np.median(absolute_errors_heading)
        absolute_errors_heading_tail = self.get_percentile(absolute_errors_heading, percentile)

        # relative heading
        relative_errors_heading = self.compute_relative_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        relative_errors_heading_mean = np.mean(relative_errors_heading)
        relative_errors_heading_variance = np.var(relative_errors_heading)
        relative_errors_heading_median = np.median(relative_errors_heading)
        relative_errors_heading_tail = self.get_percentile(relative_errors_heading, percentile)

        # create the table
        dict = {
            "Metric": [
                "Mean",
                "Variance",
                "Median",
                "{}th percentile".format(percentile),
            ],
            "Absolute position (m)": [
                absolute_errors_pose_mean,
                absolute_errors_pose_var,
                absolute_errors_pose_median,
                absolute_errors_pose_tail,
            ],
            "Relative position (m)": [
                relative_errors_pose_mean,
                relative_errors_pose_var,
                relative_errors_pose_median,
                relative_errors_pose_tail,
            ],
            "Absolute heading (deg)": [
                absolute_errors_heading_mean,
                absolute_errors_heading_variance,
                absolute_errors_heading_median,
                absolute_errors_heading_tail
            ],
            "Relative heading (deg)": [
                relative_errors_heading_mean,
                relative_errors_heading_variance,
                relative_errors_heading_median,
                relative_errors_heading_tail],
        }

        df = pd.DataFrame(dict)
        display(df)

        return
    
    
    def record_error_statistics(
            self,
            history_position_m:np.ndarray,
            history_position_m_gt:np.ndarray,
            history_heading_deg:list,
            history_heading_deg_gt:list,
            save_folder:str,
            file_name:str
    ):
        """Generate csv file of errors between estimates and gt

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees
            save_path (str): path to save the csv file
        """
       
        #compute absolute position error
        absolute_errors_position = self.compute_absolute_errors_position(
            history_position_m,
            history_position_m_gt
        )
                
        #compute absolute heading errors
        absolute_errors_heading = self.compute_absolute_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        dict = {
            "Position": absolute_errors_position,
            "Heading": absolute_errors_heading,
        }
        df = pd.DataFrame(dict)
        path = os.path.join(save_folder,file_name + "_absolute.csv")
        df.to_csv(path,index=False)

        #compute relative position error
        relative_errors_position = self.compute_relative_errors_position(
            history_position_m,
            history_position_m_gt
        )        
        #compute relative heading errors
        relative_errors_heading = self.compute_relative_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        dict = {
            "Position": relative_errors_position,
            "Heading": relative_errors_heading,
        }
        df = pd.DataFrame(dict)
        path = os.path.join(save_folder,file_name + "_relative.csv")
        df.to_csv(path,index=False)
        

        total_distance = self.compute_total_distance_traveled(
            history_pose_gt=history_position_m_gt
        )
        dict = {
            "total_distance":[total_distance],
            "final_position_error":[absolute_errors_position[-1]],
            "final_heading_error_deg":[absolute_errors_heading[-1]],
            "num_frames":[absolute_errors_position.shape[0]]
        }
        df = pd.DataFrame(dict)
        path = os.path.join(save_folder,file_name + "_summary.csv")
        df.to_csv(path,index=False)

        print("finished saving csv file")

        return