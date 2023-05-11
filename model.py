import numpy as np


def ode(x, u):
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
    dx = np.dot(A, x) + np.dot(B, u)

    return dx


def f_xu(x, u, Ts):
    """
    RK-4 integral

    :param x:
    :param u:
    :param Ts:
    :return:
    """
    x = np.reshape(x, (-1, 1))
    u = np.reshape(u, (-1, 1))
    k1 = ode(x, u)
    k2 = ode(x + Ts / 2 * k1, u)
    k3 = ode(x + Ts / 2 * k2, u)
    k4 = ode(x + Ts * k3, u)

    x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def sim(x0, u_k, timestamp):
    """

    :param x0: n*1
    :param u_k: p*k
    :param timestamp: 1*k
    :return: x_all, n*k
    """
    x = np.reshape(x0, (-1, 1))
    x_all = x

    ts = np.diff(timestamp)

    for i in range(u_k.shape[1] - 1):  # the last U is useless
        u = u_k[:, i]
        x = f_xu(x, u, ts[0][i])
        x_all = np.append(x_all, x, axis=1)

    return x_all

