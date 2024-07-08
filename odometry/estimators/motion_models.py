import numpy as np
from odometry.estimators.estimators import Inertial
from odometry.supportFns.rotation_functions import apply_unique_rot_trans_to_multiple_points

class MotionModel:
    """Base motion model class with methods to be implemented
    by the child class
    """
    def __init__(self) -> None:

        #time tracking
        self.t0:float = 0.0
        self.t:float = 0.0

        #state tracking
        self.x:np.ndarray = None #minimum of [x,y,phi (rad), vel]
        self.states_initialized:bool = False
        #random number generation
        self.rng =  np.random.default_rng()
        
        return

    def reset(self,t0,x0:np.ndarray,*args,**kwargs):
        """reset the motion model

        Args:
            t0 (float,optional): start time in seceonds.
            x0 (np.ndarray, optional): Initial state space
                minimum of [x,y,phi,vel].
        """
        #reset time tracking
        self.t0 = t0
        self.t = t0

        #reset state matrix
        self.x = x0

        self.states_initialized = True
        
        return

    def predict(self,dt:float,inertial:Inertial, *args,**kwargs):
        """Predict the motion model forward with inertial sensor
        measurements

        Args:
            dt (float): time since last measurement
            inertial (Inertial): Inertial object with at least angular (rad/sec)
            and linear velocity (m/s) measurements
        """
        
        #implemented by child class
        
        pass

    def sample(self,n_samples)->np.ndarray:
        """Compute N randomly distributed samples 
            based on the motion model to apply to N particles

        Args:
            n_samples (int): the number of samples to generate
                from the motion model sampler

        Returns:
            np.ndarray: Nx3 samples with [x,y,phi]
        """
        
        pass

    def get_updated_particles(self,particles:np.ndarray)->np.ndarray:
        """Obtain an updated set of particles that have 
        been propagated forward using information from the 
        motion model

        Args:
            particles (np.ndarray): Nx3 array of N particles
                containing [x,y,phi (rad)] for each particle

        Returns:
            np.ndarray: Nx3 array of N particles that 
                have been propagated forward using the 
                motion model
        """

        #sample the motion model (in the sensor coordinate frame)
        odom_update_samples = self.sample(particles.shape[0])

        #apply a rotation and translation to the x,y coordinate of
        #each sample so that the sampled odometry update is in 
        #each unique particle's sensor frame        
        odom_updates_in_particle_frames = \
            apply_unique_rot_trans_to_multiple_points(
                points=odom_update_samples[:,0:2],
                rot_angles_rad=particles[:,2],
                translations=particles[:,0:2]
            )
        
        #update the odom_update_samples wiht the updates in each
        #particle's reference frame
        odom_update_samples[:,0:2] = odom_updates_in_particle_frames[:,0,:]
        
        return odom_update_samples

####################################################################
# Multi-variate gaussian measurement models
    #inspiried by EKF computation (don't work too well right now)
####################################################################

class InertialIntegratorMM(MotionModel):
    def __init__(self):
        super().__init__()
        
        # add variable for state covariance
        self.P:np.ndarray = None

    def reset(self, t0, x0: np.ndarray, P0:np.ndarray, *args, **kwargs):
        """reset motion model states

        Args:
            t0 (float,optional): start time in seceonds.
            x0 (np.ndarray, optional): Initial state space
            P0 (np.ndarray, optional): Initial state covariance
                matrix.
        """
        super().reset(t0, x0, *args, **kwargs)
        
        self.P = P0

        return

    def f_func(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float) -> np.ndarray:
        """State f function to propagate states forward

        Args:
            x (np.ndarray): current states defined as 
                [x,y,z,phi,vel,gyro_bias,encoder bias]
            inertial (Inertial): Inertial object with angular
                velocity (omega) in rad/sec and linear velocity
                in m/s
            dt (float): time since the last sample

        Returns:
            np.ndarray: updated state space
        """
        #implemented by child
        pass
    
    def get_F_matrix(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float)->np.ndarray:
        """Jacobian matrix based on the f function

        Args:
            x (np.ndarray): current state space
            inertial (Inertial): inertial object (unused)
            dt (float): time since last sample (in seconds)

        Returns:
            np.ndarray: the Jacobian matrix based on the f function
        """

        #implemented by child
        pass

    def get_Q_matrix(
        self,
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
        #implemented by child
        pass

    def predict(
            self,
            dt:float,
            inertial: Inertial):
        """Predict the motion model forward with inertial sensor
        measurements

        Args:
            dt (float): time since last measurement
            inertial (Inertial): Inertial object with at least angular (rad/sec)
            and linear velocity (m/s) measurements
        """      
        #update the states
        self.x = self.f_func(self.x,inertial,dt)
        
        #update the covariance matrix
        F = self.get_F_matrix(
            x=self.x,
            inertial=inertial,
            dt=dt
        )
        Q = self.get_Q_matrix(
            x=self.x,
            inertial=inertial,
            dt=dt
        )

        self.P = F @ self.P @ F.T + Q

        self.t += dt
    
    def sample(self,n_samples):
        #to be implemented by the child class
        pass

class GyroEncoderIntegratorMM(InertialIntegratorMM):

    def __init__(
            self,
            t0:float=0.0,
            x0:np.ndarray=np.zeros(shape=6),
            P0:np.ndarray=np.diag([5,5,0.1,1,1e-2,1e-2])
    ):
        """Initialize a new GyroEncoderIntegrator object
        states are [x,y,phi,vel,gyro_bias, encoder_bias]

        Args:
            t0 (float): start time in seceonds
            x0 (np.ndarray, optional): Initial state space
                [x,y,z,phi,vel,gyro_bias,encoder bias].
                Defaults to np.zeros(shape=6).
            P0 (np.ndarray, optional): Initial state covariance
                matrix. Defaults to np.diag([5,5,0.1,1,1e-2,1e-2]).
        """

        #initialize state variables
        super().__init__()

        self.reset(t0,x0,P0)

    def reset(
            self,
            t0:float=0.0,
            x0:np.ndarray=np.zeros(shape=6),
            P0:np.ndarray=np.diag([5,5,0.1,1,1e-2,1e-2])
    ):
        """reset the encoder object
        states are [x,y,phi,vel,gyro_bias, encoder_bias]

        Args:
            t0 (float,optional): start time in seceonds. Defaults to 0.0
            x0 (np.ndarray, optional): Initial state space
                [x,y,z,phi,vel,gyro_bias,encoder bias].
                Defaults to np.zeros(shape=6).
            P0 (np.ndarray, optional): Initial state covariance
                matrix. Defaults to np.diag([5,5,0.1,1,1e-2,1e-2]).
        """
        
        #reset all state matricies
        super().reset(t0,x0,P0)

        return
    
    def f_func(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float) -> np.ndarray:
        """State f function to propagate states forward

        Args:
            x (np.ndarray): current states defined as 
                [x,y,z,phi,vel,gyro_bias,encoder bias]
            inertial (Inertial): Inertial object with angular
                velocity (omega) in rad/sec and linear velocity
                in m/s
            dt (float): time since the last sample

        Returns:
            np.ndarray: updated state space
        """

        assert dt >= 0
        
        #get the angular and linear velocity from the inertial
        omega = inertial.gyro
        vel = inertial.sencode

        # propagate omega and velocity with IMU/encoder
        x_old = x.copy()
        x[2] = x[2] + (omega - x[4]) * dt
        x[3] = vel - x[5] * dt

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
        """Jacobian matrix based on the f function

        Args:
            x (np.ndarray): current state space
            inertial (Inertial): inertial object (unused)
            dt (float): time since last sample (in seconds)

        Returns:
            np.ndarray: the Jacobian matrix based on the f function
        """
        
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
        self,
        x: np.ndarray,
        dt: float,
        sigma_x=0.01, #originally .001
        sigma_h=0.005, #originally 0.005
        sigma_s=0.001, #originally 0.01
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

    def sample(self, n_samples:int)->np.ndarray:
        """Compute N randomly distributed samples 
            based on the motion model to apply to N particles

        Args:
            n_samples (int): the number of samples to generate
                from the motion model sampler

        Returns:
            np.ndarray: Nx3 samples with [x,y,phi]
        """
        return self.rng.multivariate_normal(
            mean=self.x[0:3],
            cov=self.P[0:3,0:3],
            size=n_samples
        )

####################################################################
# Odometry motion model
    #"sample_motion_model_odometry" algorithm from Probabilistic
    #robotics text book
####################################################################

class OdometryMM(MotionModel):
    def __init__(
            self,
            a1=0.01,
            a2=0.01,
            a3=0.01,
            a4=0.01) -> None:
        """initialize odometry motion model
        NOTE: states defined as [x,y,z,phi,vel]


        Args:
            a1 (float, optional): Error scalar applied to rotations when
                computing rotation variances. Defaults to 0.01.
            a2 (float, optional): Error scalar applied to translation when
                computing rotation variances. Defaults to 0.01.
            a3 (float, optional): Error scalar applied to translation when
                computing translation variances. Defaults to 0.01.
            a4 (float, optional): Error scalar applied to rotations when
                computing translation variances. Defaults to 0.01.

        Returns:
            _type_: _description_
        """

        #initialize parent class
        super().__init__()

        #define scalars for error constants
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

        self.reset()
        
        return

    def reset(
            self,
            t0=0.0,
            x0:np.ndarray=np.zeros(shape=4,dtype=float),
            *args, **kwargs):
        """reset the motion model, if previously initialized
        only resets the x,y, and phi terms of the motion
        model states

        Args:
            t0 (float,optional): start time in seceonds.
                defaults to 0.0
            x0 (np.ndarray, optional): Initial state space
                as [x,y,phi,vel]. Defaults to np.zeros(shape=4)
        """
        
        if self.states_initialized:
            x = self.x.copy()
            x[0:3] = x0[0:3]
        
            super().reset(t0, x, *args, **kwargs)
        else:
            super().reset(t0, x0, *args, **kwargs)

        return
    
    def predict(self, dt: float, inertial: Inertial):
        """Predict the motion model forward with inertial sensor
        measurements

        Args:
            dt (float): time since last measurement
            inertial (Inertial): Inertial object with at least angular (rad/sec)
            and linear velocity (m/s) measurements
        """   
        assert dt >= 0
        
        #get the angular and linear velocity from the inertial
        omega = inertial.gyro
        vel = inertial.sencode

        # propagate omega and velocity with IMU/encoder
        x_old = self.x.copy()
        self.x[2] = self.x[2] + (omega * dt)
        self.x[3] = vel

        # propagate position
        v_avg = (x_old[3] + self.x[3]) / 2
        phi_avg = (x_old[2] + self.x[2]) / 2
        self.x[0] = self.x[0] + dt * v_avg * np.cos(phi_avg)
        self.x[1] = self.x[1] + dt * v_avg * np.sin(phi_avg)
        
        return
    
    def sample(self, n_samples) -> np.ndarray:
        
        #1. compute the rotation, translation, and rotation used 
        #to decompose the ego motion from the accumulated odometry

        #from eq5.34 in probabilistic robotics
        d_rot_1 = np.arctan2(self.x[1], self.x[0]) - self.x[2]

        #from eq5.35 in probabilistic robotics
        d_trans = np.linalg.norm(self.x[0:2])

        #from eq5.36 from probabilistic robotics
        d_rot_2 = self.x[2] - d_rot_1

        #2. using a multivariate gaussian, generate n 
        mean = np.array([d_rot_1,d_trans,d_rot_2])

        c_rot_1 = (self.a1 * (d_rot_1 ** 2)) + \
            (self.a2 * (d_trans ** 2))
        c_trans = (self.a4 * (d_rot_1 ** 2)) + \
            (self.a4 * (d_rot_2 ** 2)) + \
            (self.a3 * (d_trans ** 2))
        c_rot_2 = (self.a1 * (d_rot_2 ** 2)) + \
            (self.a2 * (d_trans ** 2))
        
        cov = np.diag([c_rot_1,c_trans,c_rot_2])

        #compute the preterbed vals with N samples and cols of
        #[d_rot_1, d_trans, and d_rot_2]
        preturbed_vals = self.rng.multivariate_normal(
            mean=mean,
            cov=cov,
            size=n_samples
        )

        #compute the sampels
        samples = np.zeros(shape=(n_samples,3),dtype=float)

        #compute x vals
        samples[:,0] = np.multiply(
            preturbed_vals[:,1],
            np.cos(self.x[2] + preturbed_vals[:,0]))
        
        #compute y vals
        samples[:,1] = np.multiply(
            preturbed_vals[:,1],
            np.sin(self.x[2] + preturbed_vals[:,0]))
        
        #compute phi vals
        samples[:,2] = preturbed_vals[:,0] + preturbed_vals[:,2]

        return samples

