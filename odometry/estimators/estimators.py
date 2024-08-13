from typing import List, Union

import numpy as np
from scipy.stats.distributions import chi2


EPS = 1e-6


####################################################################
# Common Inertail class to specify inertial data
####################################################################

class Inertial:
    def __init__(
        self,
        gyro: Union[float, np.ndarray, None] = None,
        accel: Union[float, np.ndarray, None] = None,
        sencode: Union[float, None] = None,
    ):
        """Inertial data class

        gyro - angular rate in rad/sec
        accel - acceleration in meters/sec^2
        sencode - velocity in meters/sec
        """
        self.gyro = gyro
        self.accel = accel
        self.sencode = sencode

    def __str__(self) -> str:
        return f"Inerital data with gyro: {self.gyro}, accel: {self.accel}, wheel encoder: {self.sencode}"

    def __repr__(self) -> str:
        return self.__str__()

####################################################################
# KF and EKF Filters
####################################################################

class _ExtendedKalmanFilter:
    def __init__(self, t0, x0, P0, chi2_pct=0.95, do_chi2=True):
        self.t0 = t0
        self.t = t0
        self.x:np.ndarray = x0
        self.P:np.ndarray = P0
        self._x_p = x0  # for debugging
        self._P_p = P0  # for debugging
        self.do_chi2 = do_chi2
        self.g_thresh = {i: chi2.ppf(chi2_pct, i) for i in range(8)}
        self.g = None
        self.y = None

        # h_func in subclass
        # f_func in subclass

    def update(
        self, t: float, z: np.ndarray, R: np.ndarray, msmt_components: List[str]
    ):
        """Linear update"""
        if t != self.t:
            raise RuntimeError(
                f"Need to call predict to get filter to msmt time, {t} vs {self.t}"
            )
        H = self.get_H_matrix(msmt_components)

        # update if integrity passes
        self.y = z - self.h_func(self.x, msmt_components)
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        self.g = self.y.T @ Sinv @ self.y
        if self.g < 0:
            raise RuntimeError(f"g cannot be negative! P was {self.P}")
        if self.do_chi2 and (self.g > self.g_thresh[len(z)]):
            pass  # test failed
        else:
            # update
            K = self.P @ H.T @ Sinv
            self.x = self.x + K @ self.y
            self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P

    def predict(self, dt: float, inertial: Inertial, check_P: bool = False):
        if dt > EPS:
            # propagate state and covariance
            self.x = self.f_func(x=self.x, inertial=inertial, dt=dt)
            F = self.get_F_matrix(x=self.x, inertial=inertial, dt=dt)
            Q = self.get_Q_matrix(x=self.x, inertial=inertial, dt=dt)
            self.P = F @ self.P @ F.T + Q
            # for debugging only (SLOW)
            if check_P:
                try:
                    _ = np.linalg.cholesky(self.P)
                except np.linalg.LinAlgError:
                    raise RuntimeError(
                        f"P is not positive definite! P: \n{self.P}, \nF: \n{F}, \nQ: \n{Q}"
                    )

            # TODO: check if we need to add .copy() method to this
            self._x_p = self.x.copy()
            self._P_p = self.P.copy()
            self.t += dt


class _KalmanXYPhiSpeed(_ExtendedKalmanFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # measurement function
        def h_func(x: np.ndarray, msmt_components: List[str]):
            z = []
            for component in msmt_components:
                if component == "x":
                    z.append(x[0])
                elif component == "y":
                    z.append(x[1])
                elif component == "phi":
                    z.append(x[2])
                elif component == "speed":
                    z.append(x[3])
                elif component == "vx":
                    raise NotImplementedError
                elif component == "vy":
                    raise NotImplementedError
                else:
                    raise NotImplementedError(component)
            if len(z) == 0:
                raise RuntimeError(f"Did not populate z using {msmt_components}")
            z = np.asarray(z)
            return z

        self.h_func = h_func

    @classmethod
    def get_H_matrix(cls, msmt_components: List[str]):
        """Assumes state is [x, y, phi, speed]"""
        if not isinstance(msmt_components, list):
            raise ValueError(
                f"msmt_components must be a list, got {type(msmt_components)}"
            )
        H = []
        for component in msmt_components:
            hm = np.zeros((cls.n_states,))
            if component == "x":
                hm[0] = 1
            elif component == "y":
                hm[1] = 1
            elif component == "phi":
                hm[2] = 1
            elif component == "speed":
                hm[3] = 1
            else:
                raise NotImplementedError(component)
            H.append(hm)
        return np.asarray(H)


class KalmanXYPhiSpeedKinematic(_KalmanXYPhiSpeed):
    n_states = 4

    def __init__(self, *args, **kwargs):
        """Kalman filter using kinematic state prediction

        state vector: [x, y, phi, speed]
        prediction: kinematic
        update: (see super class)
        """
        super().__init__(*args, **kwargs)

        # state propagation function
        def f_func(x: np.ndarray, dt: float, **kwargs):
            assert dt >= 0
            x[0] = x[0] + dt * x[3] * np.cos(x[2])
            x[1] = x[1] + dt * x[3] * np.sin(x[2])
            return x

        self.f_func = f_func

    @staticmethod
    def get_F_matrix(x: np.ndarray, dt: float, **kwargs):
        assert dt >= 0
        f02 = -dt * x[3] * np.sin(x[2])
        f03 = dt * np.cos(x[2])
        f12 = dt * x[3] * np.cos(x[2])
        f13 = dt * np.sin(x[2])
        F = np.array(
            [
                [1, 0, f02, f03],
                [0, 1, f12, f13],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return F

    @staticmethod
    def get_Q_matrix(
        x: np.ndarray, dt: float, sigma_x=1, sigma_h=0.5, sigma_s=0.01, **kwargs
    ):
        # TODO: tune the process noise parameters
        assert dt >= 0
        # row 0
        q00 = dt * sigma_x
        q02 = 0
        q03 = 0
        # row 1
        q11 = dt * sigma_x
        q12 = 0
        q13 = 0
        # row 2
        q20 = 0
        q21 = 0
        q22 = dt * sigma_h
        q23 = 0
        # row 3
        q30 = 0
        q31 = 0
        q32 = 0
        q33 = dt * sigma_s
        # fmt: off
        Q = np.array(
            [
                [q00,   0, q02, q03],
                [  0, q11, q12, q13],
                [q20, q21, q22, q23],
                [q30, q31, q32, q33],
            ]
        )
        # fmt: on
        return Q


class KalmanXYPhiSpeedGyroEncoder(_KalmanXYPhiSpeed):
    n_states = 6

    def __init__(self, *args, **kwargs):
        """Kalman filter using gyro and wheel encoder state prediction

        state vector: [x, y, phi, speed, gyro_bias, encoder_bias]
        prediction: gyro, wheel encoder
        update: (see super class)

        Note:
        - gyro bias is on omega in units of rad/sec
        - encoder bias is on acceleration in units of meters/sec^2
        """
        super().__init__(*args, **kwargs)
        self.gyro_buffer = []

        # state propagation function
        def f_func(x: np.ndarray, inertial: Inertial, dt: float):
            """gyro + wheel encoder propagation function

            omega --> angular velocity from gyro (1-axis)
            sencode --> "instantaneous" speed from wheel encoder
            """
            assert dt >= 0
            omega = inertial.gyro
            sencode = inertial.sencode

            # propagate omega and velocity with IMU/encoder
            x_old = x.copy()
            x[2] = x[2] + (omega - x[4]) * dt
            x[3] = sencode - x[5] * dt

            # propagate position
            v_avg = (x_old[3] + x[3]) / 2
            phi_avg = (x_old[2] + x[2]) / 2
            x[0] = x[0] + dt * v_avg * np.cos(phi_avg)
            x[1] = x[1] + dt * v_avg * np.sin(phi_avg)
            return x

        self.f_func = f_func

    @staticmethod
    def get_F_matrix(x: np.ndarray, inertial: Inertial, dt: float):
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

    @staticmethod
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

####################################################################
# Inertial Integration (prediction with no "update")
####################################################################

class InertialIntegrator:

    def __init__(self) -> None:
        
        #time tracking
        self.t:float = 0.0

        #state tracking
        self.x:np.ndarray = None
        self.states_initialized:bool = False

    def reset(self,
              t0:float=0.0,
              x0:np.ndarray=np.zeros(shape=4,dtype=float),
              *args,**kwargs):
        """Reset the inertial integrator

        Args:
            t0 (float): start time in seconds
            x0 (np.ndarray): Initial state space
                minimum of [x,y,phi,vel, gyro_bias, encoder_bias].
        """

        self.t = t0

        #reset state matrix
        self.x = x0

        self.states_initialized = True

    def predict(self, dt:float, inertial: Inertial):
        """Predict the inertial integrator forward

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

        #update the time
        self.t += dt
        
        return
    

class InertialIntegratorGyroEncoder(InertialIntegrator):

    def __init__(self) -> None:

        super().__init__()
    
    def reset(self, 
              t0: np.float = 0,
              x0: np.ndarray = np.zeros(shape=6, dtype=float),
              *args, **kwargs):
        """Reset the inertial integrator

        Args:
            t0 (float): start time in seconds
            x0 (np.ndarray): Initial state space
                minimum of [x,y,phi,vel, gyro_bias, encoder_bias].
        """

        return super().reset(t0, x0, *args, **kwargs)
    
    def predict(self, dt:float, inertial: Inertial):
        """Predict the inertial integrator forward

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
        self.x[2] = self.x[2] + (omega -self.x[4]) * dt
        self.x[3] = vel - self.x[5] * dt

        # propagate position
        v_avg = (x_old[3] + self.x[3]) / 2
        phi_avg = (x_old[2] + self.x[2]) / 2
        self.x[0] = self.x[0] + dt * v_avg * np.cos(phi_avg)
        self.x[1] = self.x[1] + dt * v_avg * np.sin(phi_avg)

        #update the time
        self.t += dt
        
        return
    
    def zero_vel_update(self):

        raise NotImplementedError
