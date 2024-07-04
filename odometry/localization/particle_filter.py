import numpy as np
from sklearn.neighbors import NearestNeighbors
from odometry.supportFns import rotation_functions
import scipy.stats
from scipy.stats import norm
from odometry.estimators.estimators import Inertial

class particleFilter:

    def __init__(
            self,
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
        self.x:np.ndarray = None #[x,y,phi, speed, gyro_bias, encoder_bias]
        self.P:np.ndarray = None #state covariance matrix
        self.Q:np.ndarray = None #state process noise

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
    #sampling
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
    #motion model (multivariate gaussian)
    #               -from KalmanYPhiSpeedGyroEncoder
    ####################################################################

    def reset_states(self):
        pass
    
    def f_func(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float) -> np.ndarray:
        
        assert dt >= 0
        
        #get the angular and linear velocity from the inertial
        omega = inertial.gyro
        vel = inertial.sencode

        # propagate omega and velocity with IMU/encoder
        x_old = x.copy()
        x[2] = x[2] + (omega - x[4]) * dt
        x[3] = (vel - x[5]) * dt

        # propagate position
        v_avg = (x_old[3] + x[3]) / 2
        phi_avg = (x_old[2] + x[2]) / 2
        x[0] = x[0] + dt * v_avg * np.cos(phi_avg)
        x[1] = x[1] + dt * v_avg * np.sin(phi_avg)
        return x
    
    def get_F_matrix(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float)->np.ndarray:
        
        assert dt >= 0
        # x position
        f02 = -dt * x[3] * np.sin(x[2])
        f03 = dt * np.cos(x[2])
        # y position
        f12 = dt * x[3] * np.cos(x[2])
        f13 = dt * np.sin(x[2])
        # phi
        f24 = -dt
        # speed
        f35 = -dt
        # full matrix
        # fmt: off
        F = np.array(
            [
                [1, 0, f02, f03,   0,   0],
                [0, 1, f12, f13,   0,   0],
                [0, 0,   1,   0, f24,   0],
                [0, 0,   0,   1,   0, f35],
                [0, 0,   0,   0,   1,   0],
                [0, 0,   0,   0,   0,   1],
            ]
        )
        # fmt: on
        return F
    
    def get_Q_matrix(
        x: np.ndarray,
        dt: float,
        sigma_x=0.001, #originally .001
        sigma_h=0.005, #originally 0.005
        sigma_s=0.01, #originally 0.01
        sigma_g=1e-5,
        sigma_e=1e-5,
        **kwargs,
    ):
        """Process noise matrix
        sigma_x - position process noise
        sigma_h - heading process noise
        sigma_s - speed process noise
        sigma_g - gyro bias random walk, units of rad/sec * 1/sqrt(Hz)
        sigma_e - encoder bias random walk, units of m/sec * 1/sqrt(Hz)
        """
        assert dt >= 0
        # TODO: tune the process noise parameters
        # row 0
        q00 = dt * sigma_x
        q02 = 0
        q03 = 0
        q04 = 0
        q05 = 0
        # row 1
        q11 = dt * sigma_x
        q12 = 0
        q13 = 0
        q14 = 0
        q15 = 0
        # row 2
        q20 = 0
        q21 = 0
        q22 = dt * sigma_h
        q23 = 0
        q24 = 0
        q25 = 0
        # row 3
        q30 = 0
        q31 = 0
        q32 = 0
        q33 = dt * sigma_s
        q34 = 0
        q35 = 0
        # row 4
        q40 = 0
        q41 = 0
        q42 = 0
        q43 = 0
        q44 = dt * sigma_g
        q45 = 0
        # row 5
        q50 = 0
        q51 = 0
        q52 = 0
        q53 = 0
        q54 = 0
        q55 = dt * sigma_e
        # fmt: off
        Q = np.array(
            [
                [q00,   0, q02, q03, q04, q05],
                [  0, q11, q12, q13, q14, q15],
                [q20, q21, q22, q23, q24, q25],
                [q30, q31, q32, q33, q34, q35],
                [q40, q41, q42, q43, q44, q45],
                [q50, q51, q52, q53, q54, q55],
            ]
        )
        # fmt: on
        return Q

    def predict_states_forward(
            self,
            dt:float,
            inertial: Inertial):
        
        #update the states
        self.x = self.f_func(self.x,inertial,dt)
        
        #update the covariance matrix
        F = self.get_F_matrix(
            x=self.x,
            inertial=inertial,
            dt=dt
        )
        Q = self.get_Q

        