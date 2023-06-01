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


def f_k(x, u, Ts, P, Q):  # RK-4 integral

    x = np.reshape(x, (-1, 1))
    u = np.reshape(u, (-1, 1))
    k1, kp1 = ode_kf(x, u, P, Q)
    k2, kp2 = ode_kf(x + Ts / 2 * k1, u, P + Ts / 2 * kp1, Q)
    k3, kp3 = ode_kf(x + Ts / 2 * k2, u, P + Ts / 2 * kp2, Q)
    k4, kp4 = ode_kf(x + Ts * k3, u, P + Ts / 2 * kp3, Q)

    x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
    P_next = P + 1 / 6 * Ts * (kp1 + 2 * kp2 + 2 * kp3 + kp4)

    # 0 <= theta < 2*pi
    if x_next[2][0] < 0:
        x_next[2][0] += 2 * np.pi
    elif x_next[2][0] >= 2 * np.pi:
        x_next[2][0] -= 2 * np.pi

    return x_next, P_next


def sim_kf(x0, u_k, y_k, switch, Ts, Q, R, P0):
    # Initial state
    x = np.reshape(x0, (-1, 1))
    x_all = x
    P = P0

    # C output matrix
    C = np.array([[0, 0, 0, 1]])

    u = np.array([[0.], [0.]])  # initialize the input
    for i in range(u_k.shape[1] - 1):  # the last U is useless

        if switch[i] == True:  # # prediction step
            u = u_k[:, i]
            x, P = f_k(x, u, Ts[i], P, Q)

        else:
            # # prediction step using last u
            x, P = f_k(x, u, Ts[i], P, Q)

            # # correction step
            # get velocity measurement
            y = y_k[i].reshape(1, 1)

            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
            x = x + K @ (y - C @ x)
            P = (np.eye(4) - K @ C) @ P

        x_all = np.append(x_all, x, axis=1)

    return x_all


