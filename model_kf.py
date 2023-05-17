import numpy as np
import pandas as pd


def ode_kf(x, u, P, Q):
    """
    ODE equations

    :param x: [x, y, theta, v].T  , all system states
    :param u: [a_z, w_y].T  , 'inputs' of the system
    :return: dot_x
    """

    A = np.array([[0, 0, 0, np.cos(x[2][0])],
                  [0, 0, 0, np.sin(x[2][0])],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, -1],
                  [1, 0]])
    dx = dx = A @ x + B @ u

    dP = A @ P + P @ A.T + Q

    return dx, dP


def f_kf(x, u, Ts, P, Q):  # RK-4 integral

    x = np.reshape(x, (-1, 1))
    u = np.reshape(u, (-1, 1))
    k1, kp1 = ode_kf(x, u, P, Q)
    k2, kp2 = ode_kf(x + Ts / 2 * k1, u, P + Ts / 2 * kp1, Q)
    k3, kp3 = ode_kf(x + Ts / 2 * k2, u, P + Ts / 2 * kp2, Q)
    k4, kp4 = ode_kf(x + Ts * k3, u, P + Ts / 2 * kp3, Q)

    x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
    P_next = P + 1 / 6 * Ts * (kp1 + 2 * kp2 + 2 * kp3 + kp4)

    return x_next, P_next


def sim_kf(x0, data, Q, R, P0):
    # Initial state
    x = np.reshape(x0, (-1, 1))
    x_all = x
    P = P0

    # C output matrix
    C = np.array([[0, 0, 0, 1]])

    # get U: a_z w_y
    u_k = np.append(data[:, 13].reshape(1, -1), data[:, 2].reshape(1, -1), axis=0)  # a_z, w_y, with nan, shape: (2, n)

    # switch, to determine prediction or correction in KF, if False,then correction
    switch = np.isnan(data[:, -1])

    Ts = np.diff(data[:, 0])

    for i in range(u_k.shape[1] - 1):  # the last U is useless

        if switch[i] is True:  # # prediction step
            u = u_k[:, i]
            x, P = f_kf(x, u, Ts[i], P, Q)

        else:
            # # prediction step using last u, attention: if the first u is NaN, the code will go wrong
            x, P = f_kf(x, u, Ts[i], P, Q)

            # # correction step
            # get velocity measurement
            y = - data[i, 30].reshape(1, 1) / 1000  # unit of data is mm/s and data is inverse

            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
            x = x + K @ (y - C @ x)
            P = (np.eye(4) - K @ C) @ P

        x_all = np.append(x_all, x, axis=1)

    return x_all


