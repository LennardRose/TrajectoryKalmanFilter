import numpy as np


class KalmanFilter:

    def __init__(self, A, B, C, Q, R, a, m, s):
        # state transition model
        self.A = A
        # control input model
        self.B = B
        # observation model
        self.C = C
        # covarianbce of process noise
        self.Q = Q
        # covariance of observation noise
        self.R = R

        # control vector
        self.a = a
        # state
        self.m = m
        # estimate covariance
        self.s = s


    def estimate(self, observation, only_positions=True):
        """
        function for the kalman filter to estimate the current state
        :param observation: the current observation
        :param only_positions: wether to return only the x and y coordinates (positions) or the whole state
        :return: x and y positions or the state depending on flag set
        """

        # predict a priori estimates
        m0 = self.A.dot(self.m) + self.B.dot(self.a)
        s0 = (self.A.dot(self.s)).dot(self.A.T) + self.Q

        # Kalman gain, divided into subcalculations for debugging
        k1 = s0.dot(self.C.T)
        k2 = np.linalg.pinv(self.C.dot(s0).dot(self.C.T) + self.R)
        self.K = k1.dot(k2)

        # update state and estimate covariance
        self.m = m0 + self.K.dot(observation - self.C.dot(m0))
        self.s = (np.identity(4) - self.K.dot(self.C)).dot(s0)

        # return only x and y positions or whole state
        if only_positions:
            return self.m[0], self.m[1]
        else:
            return self.m
