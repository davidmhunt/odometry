import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
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
            measurement_model_sigma = 0.5,
            measurement_model_z_hit = 0.8,
            measurement_model_z_rand = 0.2,
            valid_pose_var_thresh = 1.0,
            valid_heading_var_thresh = 0.5) -> None:
        
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

        #measurement model parameters (only use hit and random
        # as the radar has no "max hit")
        self.mm_z_hit = measurement_model_z_hit
        self.mm_z_rand = measurement_model_z_rand
        self.gaus_dist:scipy.stats.norm_gen = norm(0,measurement_model_sigma)

        #kneighbors for identifying nearest points
        self.nbrs:NearestNeighbors = None

        #filter parameters for computing motion between states
        self.motion_model:MotionModel = motion_model

        #particle clustering
        self.dbscan_particle_clustering:DBSCAN = DBSCAN(
            eps=0.2,
            min_samples=20
        )

        #keep track of current odometry
        self.current_pose_m:np.ndarray = np.array([0.0,0.0]) #pose including motion model
        self.current_heading_rad = 0.0 #heading including motion model

        self.last_measured_pose_m:np.ndarray = np.array([0.0,0.0]) #based on last msmt with particles
        self.last_measured_heading_rad = 0.0 #based on last msmt with particles

        #tracking if the valid pose is valid
        self.current_odom_valid = False
        self.valid_pose_var_thresh = valid_pose_var_thresh
        self.valid_heading_var_thresh = valid_heading_var_thresh

    
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
    
    #TODO: Update the following functions and documentation for them
    def multinomial_sampling(self,weights: np.ndarray, n: int) -> np.ndarray:
        """Draw samples from multinomial distribution

        For a more efficient implementation, see
        https://en.wikipedia.org/wiki/Alias_method
        """
        weights /= sum(weights)  # normalize
        w_cum = np.cumsum(weights)  # CDF
        w_cum[-1] = 1.0  # avoid round-off error
        rands = np.random.rand(n)  # get points on cdf curve to sample
        indices = np.searchsorted(w_cum, rands)
        return indices

    
    def residual_sampling(self,weights: np.ndarray, n: int) -> np.ndarray:
        """Use the residual method to get samples

        Uses multinomial sampling on the residuals
        """
        if n != len(weights):
            raise NotImplementedError(
                "Not sure how to implement for n less than weights len"
            )
        indices = np.zeros(n, dtype=int)

        # take floor(N*w copies of each weight)
        num_copies = (n * np.asarray(weights)).astype(int)
        k = 0
        for i in range(n):
            for _ in range(num_copies[i]):  # make n copies
                indices[k] = i
                k += 1

        # use multinomial sampling on residuals for the rest
        residual = weights - num_copies  # fractional part
        indices[k:n] = self.multinomial_sampling(residual, n - k)

        return indices


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
        self.weights = np.ones(shape=N,dtype=float) / N
    
    def initialize_from_gaussian_dist(
            self,
            mean:np.ndarray,
            cov:np.ndarray,
            N=1000):
        """Initialize particles using a gaussian distribution
        (initialize randomly from all of th efree space)

        Args:
            mean (np.ndarray): mean expressed as [x,y,heading (rad)]
            cov (np.ndarray): 3x3 covariance matrix
            N (int, optional): number of particles to compute.
              Defaults to 1000
        """

        self.particles = self.get_gaussian_particles(
            mean=mean,
            cov=cov,
            N=N
        )
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
        weights = np.zeros(shape=(particles.shape[0],),dtype=np.float128)

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
        pdf_vals = self.mm_z_hit * \
            np.float128(self.gaus_dist.pdf(distances)) \
            + self.mm_z_rand

        #compute the weights for each particle
        weights = np.prod(pdf_vals,axis=1)
        weights = weights / np.sum(weights)

        return weights
    
    def liklihood_field_measurement_model_thresholded(
            self,
            particles:np.ndarray,
            points:np.ndarray,
            threshold_m:float = 0.1,
            percentile:float = 90) ->np.ndarray:
        """Compute the liklihood field for a given set of particles

        Args:
            particles (np.ndarray): Nx3 array of particles
            points (np.ndarray): Mx2 array of detections in the 
                sensor reference frame

        Returns:
            np.ndarray: Nx1 array of updated weights
        """
        
        #initialize an empty weights vector
        weights = np.zeros(shape=(particles.shape[0],),dtype=np.float128)

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
        
        #compute the probability based on the percentage of points that meet the threshold
        weights = np.sum(distances <= threshold_m, axis=1) / distances.shape[1]

        #only keep the top percentile of the weights
        p = np.percentile(weights,percentile)
        weights[weights<p] = 0.0

        #compute the weights for each particle
        weights = weights / np.sum(weights)

        return weights
    
    def liklihood_field_measurement_model_sprt(
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
        weights = np.zeros(shape=(particles.shape[0],),dtype=np.float128)

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
        
        #compute the pdf value
        pdf_vals = np.float64(self.gaus_dist.pdf(distances))

        #divide by some uniform value
        max_gaus = self.gaus_dist.pdf(0)
        weights = np.sum(np.log(pdf_vals/(max_gaus/3)),axis=1)

        #compute the probability of a true detection
        weights = np.exp(weights) / (1.0 + np.exp(weights))

        if(np.sum(weights>0.5) > 5):

            #compute the weights for each particle
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(shape=(particles.shape[0])) / particles.shape[0]

        return weights

    ####################################################################
    #motion model
    ####################################################################

    def motion_model_reset(self,t0,*args,**kwargs):
        """reset the motion model

        Args:
            t0 (float): start time in seceonds.
        """
        self.motion_model.reset(t0,*args,**kwargs)

        #update the current position
        self.update_current_position()
    
    def motion_model_predict(self,dt,inertial:Inertial, *args,**kwargs):
        """Predict the motion model forward with inertial sensor
        measurements

        Args:
            dt (float): time since last measurement
            inertial (Inertial): Inertial object with at least angular (rad/sec)
            and linear velocity (m/s) measurements
        """

        self.motion_model.predict(dt,inertial, *args,**kwargs) 

        #update the current position
        self.update_current_position()       
    
    def motion_model_sample(self,n_samples)->np.ndarray:
        """Compute N randomly distributed samples 
            based on the motion model to apply to N particles

        Args:
            n_samples (int): the number of samples to generate
                from the motion model sampler

        Returns:
            np.ndarray: Nx3 samples with [x,y,phi]
        """
        return self.motion_model.sample(n_samples)
    
    def motion_model_update_particles(self):
        """Update the current list of particle using the motion model
        """
        self.particles = \
            self.motion_model.get_updated_particles(self.particles)
    

    ####################################################################
    # Particle filter update algorithms
    ####################################################################
    
    def run_MCL_alg(self,measured_point_cloud:np.ndarray):

        #propagate particles forward
        self.motion_model_update_particles()

        #run the measurement model
        # self.weights = self.liklihood_field_measurement_model_thresholded(
        #     particles=self.particles,
        #     points=measured_point_cloud,
        #     threshold_m=0.5,
        #     percentile=99
        # )

        self.weights = self.liklihood_field_measurement_model(
            particles=self.particles,
            points=measured_point_cloud
        )

        resampled_idxs = self.multinomial_sampling(
            self.weights,self.weights.shape[0])
        
        self.particles = self.particles[resampled_idxs,:]
    
    ####################################################################
    # odometry updating
    ####################################################################

    def update_current_position(self):

        self.current_pose_m = rotation_functions.apply_rot_trans(
            self.motion_model.x[0:2],
            rot_angle_rad=self.last_measured_heading_rad,
            trans=self.last_measured_pose_m
        )

        self.current_heading_rad = self.last_measured_heading_rad \
            + self.motion_model.x[2]
    
    def odometry_reset(self):
        """Reset the current particle filter odometry
        """
        self.current_heading_rad = 0.0
        self.current_pose_m = np.array([0.0,0.0])

        self.last_measured_heading_rad = 0.0
        self.last_measured_pose_m = np.array([0.0,0.0])

        self.current_odom_valid = False
    
    def odometry_update_from_measurement(
            self,
            measured_point_cloud:np.ndarray):
        """Update the odometry and particle filter from a measuremnt
        NOTE: also resets the motion model

        Args:
            measured_point_cloud (np.ndarray): _description_
        """
        #run the particle filter algorithm
        self.run_MCL_alg(measured_point_cloud)

        #cluster the resampled particles
        labels = self.dbscan_particle_clustering.fit_predict(self.particles[:,0:2])
        
        #determine the most popular label
        if (np.max(labels) >= 0):
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]

            label_counts = np.array(
                [np.sum(labels==unique_labels[i]) \
                for i in range(unique_labels.shape[0])]
            )

            most_likely_particles_cluster = self.particles[
                labels == np.argmax(label_counts),:
            ]
        else:
            most_likely_particles_cluster = self.particles

        #update the pose mean and variance from the resampled particles
        self.last_measured_pose_m = np.average(
            a=most_likely_particles_cluster[:,0:2],
            axis=0
        )

        pose_m_var = np.var(
            a=most_likely_particles_cluster[:,0:2],
            axis=0
        )

        #update the heading mean and variance from the resampled particles
        self.last_measured_heading_rad = np.average(
            a=most_likely_particles_cluster[:,2]
        )

        heading_rad_var = np.var(
            a=most_likely_particles_cluster[:,2]
        )
        
        if (heading_rad_var < self.valid_heading_var_thresh) and \
            (pose_m_var[0] < self.valid_pose_var_thresh) and \
            (pose_m_var[1] < self.valid_pose_var_thresh):

            self.current_odom_valid = True
        else:
            #TODO: set current odom valid to false here if needed
            pass

        #reset the motion model (also updates the current position)
        self.motion_model_reset(
            t0=self.motion_model.t
        )


        return
        
