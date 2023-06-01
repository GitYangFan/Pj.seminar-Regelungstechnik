from euler_from_quaternion import *
import numpy as np


def ode_all(x, u, P, Q):
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


def f_k_all(x, u, Ts, P, Q):  # RK-4 integral

    x = np.reshape(x, (-1, 1))
    u = np.reshape(u, (-1, 1))
    k1, kp1 = ode_all(x, u, P, Q)
    k2, kp2 = ode_all(x + Ts / 2 * k1, u, P + Ts / 2 * kp1, Q)
    k3, kp3 = ode_all(x + Ts / 2 * k2, u, P + Ts / 2 * kp2, Q)
    k4, kp4 = ode_all(x + Ts * k3, u, P + Ts / 2 * kp3, Q)

    x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
    P_next = P + 1 / 6 * Ts * (kp1 + 2 * kp2 + 2 * kp3 + kp4)

    # 0 <= theta < 2*pi
    if x_next[2][0] < 0:
        x_next[2][0] += 2*np.pi
    elif x_next[2][0] >= 2*np.pi:
        x_next[2][0] -= 2*np.pi


    return x_next, P_next


def sim_all(x0, u_k, Ts, y_k_vel, y_k_tf, s_imu, s_vel, s_tf, Q, R_v, R_t, P0):
    """

    :param R_v: R for velocity
    :param R_t: R for tf

    """
    # Initial state
    x = np.reshape(x0, (-1, 1))
    x_all = x
    P = P0

    # C_v output matrix for velocity
    C_v = np.array([[0, 0, 0, 1]])

    # C_t output matrix for tf
    C_t = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])


    u = np.array([[0.], [0.]])  # initialize the input
    for i in range(u_k.shape[1] - 1):  # the last U is useless

        # # pure prediction step
        if not s_imu[i]:
            u = u_k[:, i]  # update input
            x, P = f_k_all(x, u, Ts[i], P, Q)

        # # correction with velocity data
        elif not s_vel[i]:
            # prediction step using last u
            x, P = f_k_all(x, u, Ts[i], P, Q)

            # correction step
            y = y_k_vel[i].reshape(1, 1)

            K = P @ C_v.T @ np.linalg.inv(C_v @ P @ C_v.T + R_v)
            x = x + K @ (y - C_v @ x)
            P = (np.eye(4) - K @ C_v) @ P

        # # correction with tf data
        elif not s_tf[i]:
            # prediction step using last u
            x, P = f_k_all(x, u, Ts[i], P, Q)

            # correction step
            # rot = data[i, 32:36]  # rot = [w, x, y, z]
            # _, _, yaw = euler_from_quaternion(rot[1], rot[2], rot[3], rot[0])
            # y = np.array([[data[i, 36]], [data[i, 37]], [yaw]])  # get tf measurement y = [x,y,theta].T

            y = y_k_tf[:, i].reshape(-1, 1)

            K = P @ C_t.T @ np.linalg.inv(C_t @ P @ C_t.T + R_t)

            # -pi < the difference of theta < pi
            y_delta = y - C_t @ x
            if y_delta[2][0] > np.pi:
                y_delta[2][0] -= 2 * np.pi
            elif y_delta[2][0] < -np.pi:
                y_delta[2][0] += 2 * np.pi

            x = x + K @ (y_delta)
            P = (np.eye(4) - K @ C_t) @ P

        x_all = np.append(x_all, x, axis=1)

    return x_all