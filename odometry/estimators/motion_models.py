import numpy as np
from odometry.estimators.estimators import Inertial


class MotionModel:
    """Base motion model class with methods to be implemented
    by the child class
    """
    def __init__(self) -> None:
        pass

    def reset(self,*args,**kwargs):
        pass

    def predict(self,dt,*args,**kwargs):
        pass

    def sample(self,n_samples)->np.ndarray:
        pass

####################################################################
# Inertial Integration measurement models
####################################################################

class InertialIntegrator(MotionModel):
    def __init__(self):
        super().__init__()
        
        #state variables
        self.t0:float = 0.0
        self.t:float = 0.0
        self.x:np.ndarray = None
        self.P:np.ndarray = None
        self.Q:np.ndarray = self.get_Q_matrix()

        #initialize a random number genreator class
        self.rng =  np.random.default_rng()

    def reset(self):
        #implemented by child class
        pass

    def f_func(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float) -> np.ndarray:
        
        #implemented by child
        pass
    
    def get_F_matrix(
            self,
            x:np.ndarray,
            inertial:Inertial,
            dt: float)->np.ndarray:
        
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
        #implemented by child
        pass

    def predict(
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

class GyroEncoderIntegrator(InertialIntegrator):

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
        self.t0=t0
        self.t = t0
        self.x = x0
        self.P = P0

        self.Q = self.get_Q_matrix()
    
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

    def sample(self, n_samples):
        """Compute N samples of the 

        Args:
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.rng.multivariate_normal(
            mean=self.x[0:3],
            cov=self.P[0:3,0:3],
            size=n_samples
        )