import numpy as np

from geometries.transforms.transformation import Transformation
from odometry.point_cloud_processing.pc_grid.pc_grid import PCGrid

class ProbabilisticPCGrid(PCGrid):

    def __init__(
            self,
            grid_resolution_m = 0.05,
            grid_max_distance_m = 3,
            num_frames_history:int=10,
            occupancy_threshold:float=0.5):

        self.num_frames_history:int = num_frames_history
        self.threshold:float = occupancy_threshold

        super().__init__(grid_resolution_m, grid_max_distance_m)

        return
    
    def _reset_points(
            self,
            new_points:np.ndarray=np.empty(shape=(0,3)),
            new_gt_points:np.ndarray=np.empty(shape=(0,3))):
        """
        Reset the saved point cloud and optionally initialize it with new points.
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved point cloud with these points. 
                Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                saved ground truth point cloud with these points. 
                Defaults to an empty set of 3D points.
        """
        self.points:list = [np.empty(shape=(0,3)) for _ in range(self.num_frames_history)]
        self.gt_points:list = [np.empty(shape=(0,3)) for _ in range(self.num_frames_history)]

        if new_points.shape[0] > 0:
            self.add_points(
                new_points=new_points,
                new_gt_points=new_gt_points
            )

        return
    
    def _reset_grid(
            self,
            new_points:np.ndarray = np.empty(shape=(0,3)),
            new_gt_points:np.ndarray = np.empty(shape=(0,3))):
        """
        Reset the saved grid and optionally initialize it with new points.
        Args:
            new_points (np.ndarray, optional): If provided, initializes the 
                saved grid with these points. 
                Defaults to an empty set of 3D points.
            new_gt_points (np.ndarray, optional): If provided, initializes the 
                saved ground truth point cloud with these points. 
                Defaults to an empty set of 3D points.
        """

        #setup point cloud grid
        if new_points.shape[0] > 0:
            self.grid = self._get_grid_from_points(new_points).astype(np.float32) \
                / float(self.num_frames_history)
        else:
            # Reset grid to empty state
            self.grid: np.ndarray = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.float32
            )
        
        #initialize ground truth grid
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            new_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m
            )
            self.gt_grid = self._get_grid_from_points(new_gt_points).astype(np.float32)
            self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.float32)
        else:
            # Reset grid to empty state
            self.gt_grid: np.ndarray = np.zeros(
                shape=(
                    self.grid_bins.shape[0],
                    self.grid_bins.shape[0]
                ), dtype=np.float32
            )
        return
    
    def add_points(
            self,
            new_points: np.ndarray,
            new_gt_points: np.ndarray=np.empty(shape=(0,3))):
        """
        Add new points to the point cloud grid and update the grid representation.

        Args:
            new_points (np.ndarray): Nx3 array of [x, y, z] points to add.
            gt_points (np.ndarray, optional): Nx3 array of [x,y,z] ground 
                truth detections (if available). 
                Defaults to np.empty(shape=0,3).
        Raises:
            ValueError: If the input points do not have a shape of Nx3.
        """
        if new_points.shape[1] != 3 and new_gt_points.shape[1] != 3:
            raise ValueError("Input points must be a 3D point (3,) or an Nx3 array of points.")
        
        #remove the last element of the array
        self.points.pop()
        self.points.insert(0,new_points)

        #remove the last element of the gt array
        self.gt_points.pop()
        if new_gt_points.shape[0] > 0 and new_points.shape[0] > 0:
            new_gt_points = self._get_dets_close_to_gt_points(
                dets=new_points,
                gt_points=new_gt_points,
                threshold=self.grid_resolution_m
            )
            self.gt_points.insert(0,new_gt_points)
        else:
            self.gt_points.insert(0,np.empty(shape=(0,3)))

        #update the probability grid
        self.update_probability_grid()

    def apply_transformation(self, transformation):
        """
        Apply a coordinate transformation to the current set of points.

        Args:
            transformation (Transformation): Transformation to apply to the points.
        """
        #transform all points in the grid, then update the probability grid
        for i in range(self.num_frames_history):

            #detections
            new_points = transformation.apply_transformation(self.points[i])
            new_points = self.filter_points_outside_grid(new_points)
            self.points[i] = new_points

            #ground truth detections
            new_points = transformation.apply_transformation(self.gt_points[i])
            new_points = self.filter_points_outside_grid(new_points)
            self.gt_points[i] = new_points

        #then reset the probability grid
        self.update_probability_grid()
    
    def get_points(self):
        return self._get_points_from_pc_grid(
            pc_grid=(self.grid >= self.threshold)
        )
    
    def get_gt_points(self):
        return self._get_points_from_pc_grid(
            pc_grid=(self.grid > 0)
        )
    

    def update_probability_grid(self):
        
        # reset the grid
        self._reset_grid()

        # create an array of grids from points
        grids = np.array([
            self._get_grid_from_points(points=self.points[i]).astype(np.float32)\
                for i in range(self.num_frames_history)
            ])
        gt_grids = np.array([
            self._get_grid_from_points(points=self.gt_points[i]).astype(np.float32)\
                for i in range(self.num_frames_history)
            ])

        # sum all grids element-wise
        self.grid = np.sum(grids, axis=0)
        self.gt_grid = np.sum(gt_grids,axis=0)

        # average the det grid
        self.grid /= float(self.num_frames_history)

        #set the gt grid to all ones
        self.gt_grid[self.gt_grid > 0] = 1
        self.gt_grid = ((self.grid > 0) & (self.gt_grid > 0)).astype(np.float32)

    


        

    
