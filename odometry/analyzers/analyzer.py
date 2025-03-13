import numpy as np
import pandas as pd
from IPython.display import display
import os
import fnmatch
import ast

from odometry.plotting.plotter_analyzer import PlotterAnalyzer

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
            pc_quality_distances:list,
            num_quality_points:list,
            percentile=0.90
    ):
        """Generate a tabular summary of errors between estimates and gt

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees
            pc_quality_distances (list): list of distances for each frame's 
                point cloud point to nearest point in the map
            num_quality_points (list): list of number of quality points for each
                point cloud frame
            percentile (float, optional): Tail error percentile. Defaults to 0.90.
        """
        
        #Absolute position
        absolute_errors_pos = self.compute_absolute_errors_position(
            history_position_m,
            history_position_m_gt
        )
        absolute_errors_pose_mean = np.mean(absolute_errors_pos)
        absolute_errors_pose_stdev = np.std(absolute_errors_pos)
        absolute_errors_pose_median = np.median(absolute_errors_pos)
        absolute_errors_pose_tail = self.get_percentile(absolute_errors_pos, percentile)

        #relative position
        relative_errors_pos = self.compute_relative_errors_position(
            history_position_m,
            history_position_m_gt
        )
        relative_errors_pose_mean = np.mean(relative_errors_pos)
        relative_errors_pose_stdev = np.std(relative_errors_pos)
        relative_errors_pose_median = np.median(relative_errors_pos)
        relative_errors_pose_tail = self.get_percentile(relative_errors_pos, percentile)

        #absolute heading
        absolute_errors_heading = self.compute_absolute_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        absolute_errors_heading_mean = np.mean(absolute_errors_heading)
        absolute_errors_heading_stdev = np.std(absolute_errors_heading)
        absolute_errors_heading_median = np.median(absolute_errors_heading)
        absolute_errors_heading_tail = self.get_percentile(absolute_errors_heading, percentile)

        # relative heading
        relative_errors_heading = self.compute_relative_errors_heading(
            history_heading_deg,
            history_heading_deg_gt
        )
        relative_errors_heading_mean = np.mean(relative_errors_heading)
        relative_errors_heading_stdev = np.std(relative_errors_heading)
        relative_errors_heading_median = np.median(relative_errors_heading)
        relative_errors_heading_tail = self.get_percentile(relative_errors_heading, percentile)

        #compute point cloud qualities
        avg_dist = np.array([np.mean(arr) for arr in pc_quality_distances])
        max_dist = np.array([np.max(arr) for arr in pc_quality_distances])
        num_pts = np.array([len(arr) for arr in pc_quality_distances])

        #avg pc distance
        pc_quality_avg_dist = np.array(avg_dist)
        pc_quality_avg_dist_mean = np.mean(pc_quality_avg_dist)
        pc_quality_avg_dist_stdev = np.std(pc_quality_avg_dist)
        pc_quality_avg_dist_median = np.median(pc_quality_avg_dist)
        pc_quality_avg_dist_tail = self.get_percentile(pc_quality_avg_dist, percentile)

        #max pc distance
        pc_quality_max_dist = np.array(max_dist)
        pc_quality_max_dist_mean = np.mean(pc_quality_max_dist)
        pc_quality_max_dist_stdev = np.std(pc_quality_max_dist)
        pc_quality_max_dist_median = np.median(pc_quality_max_dist)
        pc_quality_max_dist_tail = self.get_percentile(pc_quality_max_dist, percentile)

        #num points
        pc_quality_num_pts = np.array(num_pts)
        pc_quality_num_pts_mean = np.mean(pc_quality_num_pts)
        pc_quality_num_pts_stdev = np.std(pc_quality_num_pts)
        pc_quality_num_pts_median = np.median(pc_quality_num_pts)
        pc_quality_num_pts_tail = self.get_percentile(pc_quality_num_pts, percentile)

        # num quality points
        num_quality_points = np.array(num_quality_points)
        num_quality_points_mean = np.mean(num_quality_points)
        num_quality_points_stdev = np.std(num_quality_points)
        num_quality_points_median = np.median(num_quality_points)
        num_quality_points_tail = self.get_percentile(num_quality_points, percentile)

        # create the table
        dict = {
            "Metric": [
                "Mean",
                "stdev",
                "Median",
                "{}th percentile".format(percentile),
            ],
            "Absolute position (m)": [
                absolute_errors_pose_mean,
                absolute_errors_pose_stdev,
                absolute_errors_pose_median,
                absolute_errors_pose_tail,
            ],
            "Relative position (m)": [
                relative_errors_pose_mean,
                relative_errors_pose_stdev,
                relative_errors_pose_median,
                relative_errors_pose_tail,
            ],
            "Absolute heading (deg)": [
                absolute_errors_heading_mean,
                absolute_errors_heading_stdev,
                absolute_errors_heading_median,
                absolute_errors_heading_tail
            ],
            "Relative heading (deg)": [
                relative_errors_heading_mean,
                relative_errors_heading_stdev,
                relative_errors_heading_median,
                relative_errors_heading_tail],
            "Average Point Cloud Dist": [
                pc_quality_avg_dist_mean,
                pc_quality_avg_dist_stdev,
                pc_quality_avg_dist_median,
                pc_quality_avg_dist_tail],
            "Max Point Cloud Dist": [
                pc_quality_max_dist_mean,
                pc_quality_max_dist_stdev,
                pc_quality_max_dist_median,
                pc_quality_max_dist_tail],
            "Number of Points": [
                pc_quality_num_pts_mean,
                pc_quality_num_pts_stdev,
                pc_quality_num_pts_median,
                pc_quality_num_pts_tail],
            "Quality Points": [
                num_quality_points_mean,
                num_quality_points_stdev,
                num_quality_points_median,
                num_quality_points_tail],
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
            pc_quality_distances:list,
            num_quality_points:list,
            save_folder:str,
            file_name:str
    ):
        """Generate csv file of errors between estimates and gt

        Args:
            history_position_m (np.ndarray): Nx2 estimated position array
            history_position_m_gt (np.ndarray): Nx2 gt position array
            history_heading_deg (list): est heading for each frame in degrees
            history_heading_deg_gt (list): gt heading for each frame in degrees
            pc_quality_distances (list): list of distances for each frame's 
                point cloud point to nearest point in the map
            num_quality_points (list): list of number of quality points for each
                point cloud frame
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
        
        #point cloud quality
        df = pd.DataFrame.from_dict({"pc_quality_distances": [list(arr) for arr in pc_quality_distances]})
        path = os.path.join(save_folder,file_name + "_pc_quality_distances.csv")
        df.to_csv(path,index=False)

        num_quality_points = np.array(num_quality_points)
        dict = {
            "num_quality_points":num_quality_points
        }
        df = pd.DataFrame(dict)
        path = os.path.join(save_folder,file_name + "_num_quality_points.csv")
        df.to_csv(path,index=False)
        
        #summary statistics
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
    
    def find_files_in_directory(self,directory:str,keywords:str)->list:
        """Find files in a given directory that contain a set of keywords

        Args:
            directory (str): path to the directory
            keywords (str): keywords to search for using '*keyword1*keyword2*'

        Returns:
            list: list of files containing the specified keywords
        """
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, keywords):
                matches.append(os.path.join(root, filename))
        return matches
    
    def get_absolute_errors_from_csvs(self,save_folder:str):
        """Get the absolute  errors from a folder containing .csv files
        with absolute error computations from multiple datasets

        Args:
            save_folder (str): path to the results directory

        Returns:
            (np.ndarray,np.ndarray): absolute_errors_position,absolute_errors_heading_deg
        """
        #get absolute errors first
        absolute_errors_position = []
        absolute_errors_heading_deg = []
        
        absolute_error_files = self.find_files_in_directory(save_folder,'*_absolute*')
        for file_path in absolute_error_files:

            df = pd.read_csv(file_path)
            absolute_errors_position.extend(
                df["Position"].astype(float).tolist()
            )
            absolute_errors_heading_deg.extend(
                df["Heading"].astype(float).tolist()
            )
        absolute_errors_position = np.array(absolute_errors_position)
        absolute_errors_heading_deg = np.array(absolute_errors_heading_deg)

        return absolute_errors_position,absolute_errors_heading_deg
    
    def get_relative_errors_from_csvs(self,save_folder:str):
        """Get the relative errors from a folder containing .csv files
        with relative error computations from multiple datasets

        Args:
            save_folder (str): path to the results directory

        Returns:
            (np.ndarray,np.ndarray): relative_errors_position,relative_errors_heading_deg
        """
        #get absolute errors first
        relative_errors_position = []
        relative_errors_heading_deg = []
        
        relative_error_files = self.find_files_in_directory(save_folder,'*_relative*')
        for file_path in relative_error_files:

            df = pd.read_csv(file_path)
            relative_errors_position.extend(
                df["Position"].astype(float).tolist()
            )
            relative_errors_heading_deg.extend(
                df["Heading"].astype(float).tolist()
            )
        relative_errors_position = np.array(relative_errors_position)
        relative_errors_heading_deg = np.array(relative_errors_heading_deg)

        return relative_errors_position,relative_errors_heading_deg
    
    def get_pc_quality_stats_from_csvs(self,save_folder:str):
        """Get the point cloud quality statistics from a folder 
        containing .csv files with point cloud quality data
        from multiple datasets

        Args:
            save_folder (str): path to the results directory

        Returns:
            (np.ndarray,np.ndarray): avg_dist, max_dist, num_pts, num_quality_points
        """

        #get absolute errors first
        avg_dist = []
        max_dist = []
        num_pts = []
        num_quality_points = []
        
        #pc quality distances
        pc_quality_distances_files = self.find_files_in_directory(save_folder,'*_pc_quality_distances*')
        for file_path in pc_quality_distances_files:
            #get the point cloud distances list
            df = pd.read_csv(file_path)
            pc_quality_distances = [np.array(ast.literal_eval(row)) for row in df["pc_quality_distances"]]

            average_values = np.array([np.mean(arr) for arr in pc_quality_distances])
            max_values = np.array([np.max(arr) for arr in pc_quality_distances])
            num_elements = np.array([len(arr) for arr in pc_quality_distances])

            avg_dist.extend(average_values)
            max_dist.extend(max_values)
            num_pts.extend(num_elements)

        #num quality points
        num_quality_points_files = self.find_files_in_directory(save_folder,'*_num_quality_points*')
        for file_path in num_quality_points_files:

            df = pd.read_csv(file_path)
            num_quality_points.extend(
                df["num_quality_points"].astype(float).tolist()
            )
        num_quality_points = np.array(num_quality_points)

        return avg_dist,max_dist,num_pts,num_quality_points
    
    def get_summary_statistics_from_csvs(self,save_folder:str)->dict:
        total_distance = 0
        trial_distances = []
        final_position_errors = []
        final_heading_errors_deg = []
        num_frames = 0
        trial_frames = []

        summary_files = self.find_files_in_directory(save_folder,'*_summary*')
        for file_path in summary_files:

            df = pd.read_csv(file_path)
            total_distance += float(df.at[0,"total_distance"])
            trial_distances.append(float(df.at[0,"total_distance"]))
            final_position_errors.append(float(df.at[0,"final_position_error"]))
            final_heading_errors_deg.append(float(df.at[0,"final_heading_error_deg"]))
            num_frames += float(df.at[0,"num_frames"])
            trial_frames.append(float(df.at[0,"num_frames"]))
        
        out_dict = {
            "total_distance":total_distance,
            "trial_distances":np.array(trial_distances),
            "final_position_errors":np.array(final_position_errors),
            "final_heading_errors_deg":np.array(final_heading_errors_deg),
            "num_frames":num_frames,
            "trial_frames":np.array(trial_frames),
        }

        return out_dict
    
    def show_cumulative_summary_from_csvs(self,save_folder:str):

        percentile = 0.9

        #get absolute errors first
        absolute_errors_pos,absolute_errors_heading = \
            self.get_absolute_errors_from_csvs(save_folder)

        absolute_errors_pose_mean = np.mean(absolute_errors_pos)
        absolute_errors_pose_var = np.std(absolute_errors_pos)
        absolute_errors_pose_median = np.median(absolute_errors_pos)
        absolute_errors_pose_tail = self.get_percentile(absolute_errors_pos, percentile)

        absolute_errors_heading_mean = np.mean(absolute_errors_heading)
        absolute_errors_heading_stdev = np.std(absolute_errors_heading)
        absolute_errors_heading_median = np.median(absolute_errors_heading)
        absolute_errors_heading_tail = self.get_percentile(absolute_errors_heading, percentile)
        
        #get relative errors next
        relative_errors_pos,relative_errors_heading = \
            self.get_relative_errors_from_csvs(save_folder)
        relative_errors_pose_mean = np.mean(relative_errors_pos)
        relative_errors_pose_var = np.std(relative_errors_pos)
        relative_errors_pose_median = np.median(relative_errors_pos)
        relative_errors_pose_tail = self.get_percentile(relative_errors_pos, percentile)

        relative_errors_heading_mean = np.mean(relative_errors_heading)
        relative_errors_heading_stdev = np.std(relative_errors_heading)
        relative_errors_heading_median = np.median(relative_errors_heading)
        relative_errors_heading_tail = self.get_percentile(relative_errors_heading, percentile)

        #get point cloud quality statistics
        avg_dist, max_dist, num_pts, num_quality_points = \
            self.get_pc_quality_stats_from_csvs(save_folder)
        
        #avg pc distance
        pc_quality_avg_dist = np.array(avg_dist)
        pc_quality_avg_dist_mean = np.mean(pc_quality_avg_dist)
        pc_quality_avg_dist_stdev = np.std(pc_quality_avg_dist)
        pc_quality_avg_dist_median = np.median(pc_quality_avg_dist)
        pc_quality_avg_dist_tail = self.get_percentile(pc_quality_avg_dist, percentile)

        #max pc distance
        pc_quality_max_dist = np.array(max_dist)
        pc_quality_max_dist_mean = np.mean(pc_quality_max_dist)
        pc_quality_max_dist_stdev = np.std(pc_quality_max_dist)
        pc_quality_max_dist_median = np.median(pc_quality_max_dist)
        pc_quality_max_dist_tail = self.get_percentile(pc_quality_max_dist, percentile)

        #num points
        pc_quality_num_pts = np.array(num_pts)
        pc_quality_num_pts_mean = np.mean(pc_quality_num_pts)
        pc_quality_num_pts_stdev = np.std(pc_quality_num_pts)
        pc_quality_num_pts_median = np.median(pc_quality_num_pts)
        pc_quality_num_pts_tail = self.get_percentile(pc_quality_num_pts, percentile)


        # num quality points
        num_quality_points = np.array(num_quality_points)
        num_quality_points_mean = np.mean(num_quality_points)
        num_quality_points_stdev = np.std(num_quality_points)
        num_quality_points_median = np.median(num_quality_points)
        num_quality_points_tail = self.get_percentile(num_quality_points, percentile)

        #get summary statistics
        summary_dict = self.get_summary_statistics_from_csvs(save_folder)
        
        #final errors
        final_position_errors = summary_dict["final_position_errors"]
        final_position_errors_mean = np.mean(final_position_errors)
        final_position_errors_stdev = np.std(final_position_errors)
        final_position_errors_median = np.median(final_position_errors)
        final_position_errors_tail = self.get_percentile(final_position_errors, percentile)

        final_heading_errors = summary_dict["final_heading_errors_deg"]
        final_heading_errors_mean = np.mean(final_heading_errors)
        final_heading_errors_stdev = np.std(final_heading_errors)
        final_heading_errors_median = np.median(final_heading_errors)
        final_heading_errors_tail = self.get_percentile(final_heading_errors, percentile)

        # create the table
        dict = {
            "Metric": [
                "Mean",
                "stdev",
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
            "Final position (m)": [
                final_position_errors_mean,
                final_position_errors_stdev,
                final_position_errors_median,
                final_position_errors_tail,
            ],
            "Absolute heading (deg)": [
                absolute_errors_heading_mean,
                absolute_errors_heading_stdev,
                absolute_errors_heading_median,
                absolute_errors_heading_tail
            ],
            "Relative heading (deg)": [
                relative_errors_heading_mean,
                relative_errors_heading_stdev,
                relative_errors_heading_median,
                relative_errors_heading_tail],
            "Final heading (deg)": [
                final_heading_errors_mean,
                final_heading_errors_stdev,
                final_heading_errors_median,
                final_heading_errors_tail],
            "Average Point Cloud Dist": [
                pc_quality_avg_dist_mean,
                pc_quality_avg_dist_stdev,
                pc_quality_avg_dist_median,
                pc_quality_avg_dist_tail],
            "Max Point Cloud Dist": [
                pc_quality_max_dist_mean,
                pc_quality_max_dist_stdev,
                pc_quality_max_dist_median,
                pc_quality_max_dist_tail],
            "Number of Points": [
                pc_quality_num_pts_mean,
                pc_quality_num_pts_stdev,
                pc_quality_num_pts_median,
                pc_quality_num_pts_tail],
            "Quality Points": [
                num_quality_points_mean,
                num_quality_points_stdev,
                num_quality_points_median,
                num_quality_points_tail],
        }

        df = pd.DataFrame(dict)
        display(df)

        print("total distance: {}".format(summary_dict["total_distance"]))
        print("average trial distance: {}".format(
            np.average(summary_dict["trial_distances"])
        ))
        print("total frames: {}".format(summary_dict["num_frames"]))
        print("average frames per trial: {}".format(
            np.average(summary_dict["trial_frames"])
        ))
        print("max position error: {}".format(np.max(absolute_errors_pos)))

        return        


