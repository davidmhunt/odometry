import numpy as np
from sklearn.neighbors import NearestNeighbors
from odometry.supportFns import rotation_functions
import scipy.stats
from scipy.stats import norm
from odometry.estimators.estimators import Inertial
from odometry.estimators.motion_models import MotionModel

class particleFilter:

    def __init__(
            self,
            motion_model:MotionModel,
            max_particles=1000,
            measurement_sigma = 0.5) -> None:
        
        #initializing particles (indexed as [x,y,heading (radians)])
        self.max_particles = max_particles
        self.particles:np.ndarray = np.empty((max_particles,3),dtype=float)
        self.weights:np.ndarray = np.ones((max_particles,),dtype=float) / max_particles

        #map points
        self.map_points:np.ndarray = None
        self.map_free_points:np.ndarray = None
        self.map_bounds_x:np.ndarray = np.array([0.0,0.0])
        self.map_bounds_y:np.ndarray = np.array([0.0,0.0])

        #random number generator
        self.rng:np.random.Generator = np.random.default_rng()

        #gaussian distribution for liklihood fields (mean z)
        self.gaus_dist:scipy.stats.norm_gen = norm(0,measurement_sigma)

        #kneighbors for identifying nearest points
        self.nbrs:NearestNeighbors = None

        #filter parameters for computing motion between states
        self.motion_model:MotionModel = motion_model
    
    ####################################################################
    #Load map information
    ####################################################################
        
    def load_map_point_cloud(self,map_points:np.ndarray):
        """Load a map's point cloud to be used for localization tasks

        Args:
            map_points (np.ndarray): Nx2 numy array of points for the global map
        """

        self.map_points = map_points
        print("loaded map with {} points\n".format(self.map_points.shape[0]))

        #fit the nearest neighbors model to it
        self.nbrs = NearestNeighbors(
            n_neighbors=1,
            algorithm='kd_tree').fit(self.map_points)

        return
    
    def load_map_free_points(self,map_free_points:np.ndarray):
        """Load a point cloud corresponding to the free points in a occupance map

        Args:
            map_free_points (np.ndarray): Nx2 numy array of free  points
        """

        self.map_free_points = map_free_points
        print("loaded map with {} free points\n".format(self.map_points.shape[0]))


        return
    
    def set_map_bounds_from_map(self, map_points:np.ndarray):
        """Set the map boundaries

        Args:
            map_points (np.ndarray): Nx2 array of map points
        """
        self.map_bounds_x = np.array([
            np.min(map_points[:,0]) - 0.5,
            np.max(map_points[:,0]) + 0.5
        ])
        self.map_bounds_y = np.array([
            np.min(map_points[:,1]) - 0.5,
            np.max(map_points[:,1]) + 0.5
        ])
    
    ####################################################################
    #sampling from distributions
    ####################################################################
    
    def get_uniform_particles(
            self,
            x_range:tuple,
            y_range:tuple,
            hdg_range:tuple = (0,2 * np.pi),
            N=1000)->np.ndarray:
        """Generate a set of random particles using a uniform disribution

        Args:
            x_range (tuple): the (x_min,x_max) range of possible x 
                values (in meters)
            y_range (tuple): the (y_min,y_max) range of possible y
                values (in meters)
            hdg_range (tuple, optional): the range of heading values.
                Defaults to (0,2 * np.pi).
            N (int, optional): The number of particles to generate.
                Defaults to 1000.

        Returns:
            np.ndarray: Nx3 array of particles
        """
        
        particles = np.empty((N,3))
        particles[:, 0] = self.rng.uniform(x_range[0],x_range[1],size=N)
        particles[:, 1] = self.rng.uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = self.rng.uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi #make sure the heading is between zero and 2 pi
        return particles
    
    def get_uniform_particles_from_free_space(
            self,
            hdg_range:tuple = (0,2 * np.pi),
            N=1000) ->np.ndarray:
        """Generate a set of particles from the free space in a map
        (samples using uniform distribution)

        Args:
            hdg_range (tuple, optional): the range of heading values.
                Defaults to (0,2 * np.pi).
            N (int, optional): The number of particles to generate.
                Defaults to 1000.

        Returns:
            np.ndarray: Nx3 array of particles
        """

        assert self.map_free_points is not None, "Attempted to sample from map \
            free points but self.map_free_points hasn't been initialized"

        particles = np.empty((N,3))
        
        particles[:,0:2] = self.rng.choice(
            a=self.map_free_points,
            size=N,
            replace=True)
        
        #compute the headings
        particles[:, 2] = self.rng.uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi #make sure the heading is between zero and 2 pi

        return particles
        
    
    def get_gaussian_particles(
            self,
            mean:np.ndarray,
            cov:np.ndarray,
            N = 1000) -> np.ndarray:
        """Obtains N particles using a multi-variate gaussian

        Args:
            mean (np.ndarray): mean expressed as [x,y,heading (rad)]
            cov (np.ndarray): 3x3 covariance matrix
            N (int, optional): number of particles to compute.
              Defaults to 1000.

        Returns:
            np.ndarray: Nx3 array of particles 
        """

        return self.rng.multivariate_normal(mean,cov,size=N)


    ####################################################################
    #initializing particles and weights
    ####################################################################

    def initialize_from_uniform_dist(
            self,
            hdg_range:tuple = (0,2 * np.pi),
            N=1000):
        """Initialize particles using a uniform distribution
        (initialize randomly from all of th efree space)

        Args:
            hdg_range (tuple, optional): the range of heading values.
                Defaults to (0,2 * np.pi).
            N (int, optional): The number of particles to generate.
                Defaults to 1000.
        """

        self.particles = self.get_uniform_particles_from_free_space(
            hdg_range=hdg_range,
            N=N
        )

        self.max_particles = N

        self.weights = np.ones(shape=N,dtype=float) / N

    ####################################################################
    #measurement models
    ####################################################################
    def liklihood_field_measurement_model(
            self,
            particles:np.ndarray,
            points:np.ndarray) ->np.ndarray:
        """Compute the liklihood field for a given set of particles

        Args:
            particles (np.ndarray): Nx3 array of particles
            points (np.ndarray): Mx2 array of detections in the 
                sensor reference frame

        Returns:
            np.ndarray: Nx1 array of updated weights
        """
        
        #initialize an empty weights vector
        weights = np.zeros(shape=(particles.shape[0],),dtype=float)

        #re-initialize the neighbors
        self.nbrs = NearestNeighbors(
            n_neighbors=1,
            n_jobs=1,
            algorithm='kd_tree'
        ).fit(self.map_points)

        #apply the rotation and translation for each particle for each of the points
        aligned_points = rotation_functions.apply_multiple_rot_trans(
            points=points,
            rot_angles_rad=particles[:,2],
            translations=particles[:,0:2]
        )
        
        #for each point, compute the distance to the nearest point in the map
        distances = np.array([self.nbrs.kneighbors(aligned_points[i, :, :])[0] \
             for i in range(aligned_points.shape[0])])
        
        #compute the pdf value for each distance
        pdf_vals = self.gaus_dist.pdf(distances)

        #compute the weights for each particle
        weights = np.prod(pdf_vals,axis=1)
        weights = weights / np.sum(weights)

        return weights

    ####################################################################
    #motion model
    ####################################################################

    def motion_model_reset(self,*args,**kwargs):

        self.motion_model.reset(*args,**kwargs)
    
    def motion_model_predict(self,dt,*args,**kwargs):

        self.motion_model.predict(dt,*args,**kwargs)
    
    def motion_model_sample(self,n_samples):

        self.motion_model.sample(n_samples)