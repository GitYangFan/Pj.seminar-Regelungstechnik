import numpy as np


def reshape_x2m(x):
    """
    state x to matrix and vector
    :param x: 15*1 numpy.ndarray
    :return: o_r_c, o_v_c, o_R_c
    """

    o_r_c = x[0:3]
    o_v_c = x[3:6]
    o_R_c = x[6:15].reshape(3, 3)
    return o_r_c, o_v_c, o_R_c


def reshape_m2x(o_r_c, o_v_c, o_R_c):
    """
    matrix and vector to state x
    :param o_r_c: 3*1 numpy.ndarray
    :param o_v_c: 3*1 numpy.ndarray
    :param o_R_c: 3*3 numpy.ndarray

    :return: x , 15*1 numpy.ndarray
    """
    return np.concatenate((o_r_c, o_v_c, o_R_c.reshape(-1, 1)), axis=0)


def ode_3d(x, u):
    """
    ODE equations

    :param x: [o_r_c, o_v_c, o_R_c].T , (3+3+9)*1 all system states,
        o_r_c: linear position 3*1=3 ,
        o_v_c: linear velocity 3*1=3 ,
        o_R_c: rotation matrix 3*3=9 ,

    :param u: [w_x, w_y, w_z, a_x, a_y, a_z].T  , 'inputs' of the system
    :return: dot_x
    """

    o_r_c, o_v_c, o_R_c = reshape_x2m(x)    # transform x vector to 3 vector, matrix with meaning
    c_w = u[0: 3]   # angular velocity related to car coordinate
    c_a = u[3: 6]   # linear acceleration related to car coordinate

    B_w = np.array([[        0., -c_w[2][0],  c_w[1][0]],
                    [ c_w[2][0],         0., -c_w[0][0]],
                    [-c_w[1][0],  c_w[0][0],        0.]])

    d_o_R_c = np.dot(o_R_c, B_w)    # derivative of rotation matrix
    # translate the linear acceleration from the car coordinate to the basic one
    # and minus the gravity
    o_a_c = np.dot(o_R_c, c_a) + np.array([[0], [0], [9.721866068606383]])

    dx = reshape_m2x(o_v_c, o_a_c, d_o_R_c)
    return dx


def f_3d(x, u, Ts):
    """
    RK-4 integral

    :param x:
    :param u:
    :param Ts:
    :return:
    """
    # x = np.reshape(x, (-1, 1))
    # u = np.reshape(u, (-1, 1))

    k1 = ode_3d(x, u)
    k2 = ode_3d(x + Ts / 2 * k1, u)
    k3 = ode_3d(x + Ts / 2 * k2, u)
    k4 = ode_3d(x + Ts * k3, u)

    x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)

    # correct the rotation matrix in system states
    o_r_c, o_v_c, o_R_c = reshape_x2m(x_next)
    o_R_c, _ = np.linalg.qr(o_R_c)  # orthogonalize the rotation matrix using numpy's QR decomposition
    x_next = reshape_m2x(o_r_c, o_v_c, o_R_c)

    return x_next


def sim_3d(x0, u_k, ts):
    """

    :param x0: n*1
    :param u_k: p*k
    :param ts: 1*k , list of time step = np.diff(timestamp)
    :return: x_all, n*k
    """
    x = np.reshape(x0, (-1, 1))
    x_all = x

    for i in range(u_k.shape[1] - 1):  # the last U is useless
        u = u_k[:, i].reshape(-1,1)
        x = f_3d(x, u, ts[0][i])
        x_all = np.append(x_all, x, axis=1)

    return x_all

