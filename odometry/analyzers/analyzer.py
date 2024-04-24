import numpy as np
import pandas as pd
from IPython.display import display

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
    
    def compute_euclidian_errors(self,
                                 history_position_m:np.ndarray,
                                 history_position_m_gt:np.ndarray)->np.ndarray:
        """Compute the euclidian distance errors between estimated position
        at each frame and the gt position at each frame

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array

        Returns:
            np.ndarray: _description_
        """
        return np.linalg.norm(
            history_position_m - history_position_m_gt,
            axis=-1
        )
    
    def compute_position_state_estimate_errors(
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
    
    def compute_heading_state_estimate_errors(
            self,
            history_heading_deg:list,
            history_heading_deg_gt:list
    )->np.ndarray:
        """Compute the heading error (in radians) between the estimated heading
        at each frame and the gt heading.

        Args:
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees

        Returns:
            np.ndarray: heading errors in radians
        """
        
        estimated_heading_rad = np.deg2rad(np.array(history_heading_deg))
        gt_heading_rad = np.deg2rad(np.array(history_heading_deg_gt))

        return estimated_heading_rad - gt_heading_rad
    
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
        
        #compute euclid
        # compute euclidian error
        euclidian_errors = self.compute_euclidian_errors(
            history_position_m,
            history_position_m_gt
        )
        euclidian_mean = np.mean(euclidian_errors)
        euclidian_variance = np.var(euclidian_errors)
        euclidian_median = np.median(euclidian_errors)
        euclidian_tail = self.get_percentile(euclidian_errors, percentile)

        # compute position state errors
        x_errors,y_errors = self.compute_position_state_estimate_errors(
            history_position_m,
            history_position_m_gt
        )

        # x errors
        x_mean = np.mean(x_errors)
        x_variance = np.var(x_errors)
        x_median = np.median(x_errors)
        x_tail = self.get_percentile(x_errors, percentile)

        # y errors
        y_mean = np.mean(y_errors)
        y_variance = np.var(y_errors)
        y_median = np.median(y_errors)
        y_tail = self.get_percentile(y_errors, percentile)

        #compute heading state errors
        heading_errors_rad = self.compute_heading_state_estimate_errors(
            history_heading_deg,
            history_heading_deg_gt
        )

        # heading errors
        heading_mean = np.mean(heading_errors_rad)
        heading_variance = np.var(heading_errors_rad)
        heading_median = np.median(heading_errors_rad)
        heading_tail = self.get_percentile(heading_errors_rad, percentile)

        # create the table
        dict = {
            "Metric": [
                "Mean",
                "Variance",
                "Median",
                "{}th percentile".format(percentile),
            ],
            "Euclidian": [
                euclidian_mean,
                euclidian_variance,
                euclidian_median,
                euclidian_tail,
            ],
            "x": [x_mean, x_variance, x_median, x_tail],
            "y": [y_mean, y_variance, y_median, y_tail],
            "phi": [heading_mean, heading_variance, heading_median, heading_tail],
        }

        df = pd.DataFrame(dict)
        display(df)

        return