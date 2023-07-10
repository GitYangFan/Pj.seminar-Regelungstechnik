import numpy as np
import casadi as ca
import pandas as pd
import time


def nlinear_diff_yaw_angle(x):
    # select one of the following syntax
    # x[2] = ca.tan(0.5 * x[2])
    # x[2] = 2 - ca.cos(x[2])
    x[2] = ca.fmod(x[2] + np.pi, 2*np.pi) - np.pi
    # x[2] = 2*ca.sin(x[2]/2)
    return x


def constrain_yaw_angle(x):
    # # 0 <= theta < 2*pi
    # x[2][0] = x[2][0] % (2 * np.pi)

    # or
    # -pi <= theta < pi
    x[2][0] = (x[2][0] + np.pi) % (2 * np.pi) - np.pi
    return x


class MheMultipleShooting:
    def __init__(self, horizon, q_matrix: np.ndarray, r_vel: np.ndarray, r_tf: np.ndarray, p_matrix: np.ndarray):
        """

        :param horizon:
        :param q_matrix:    covariance matrix for state noise
        :param r_vel:       covariance matrix for velocity measurement noise
        :param r_tf:        covariance matrix for TF measurement noise
        :param p_matrix:    covariance matrix for arrival cost
        """
        if horizon < 1: raise ValueError("'horizon' should be >= 1 !")
        if q_matrix.shape != (4, 4): raise ValueError("'q_matrix' should be a 4*4 numpy array!")
        if r_vel.shape != (1, 1): raise ValueError("'r_vel' should be a 1*1 numpy array!")
        if r_tf.shape != (3, 3): raise ValueError("'r_tf' should be a 3*3 numpy array!")
        if p_matrix.shape != (4, 4): raise ValueError("'p_matrix' should be a 4*4 numpy array!")

        self.horizon = horizon                  # N
        self.r_horizon = 0                      # real horizon, smaller than horizon at the beginning
        self.f_horizon = 0                      # fake horizon, >= real horizon

        self.q_matrix = np.linalg.inv(q_matrix)     # weight matrix for state noise
        self.r_vel = np.linalg.inv(r_vel)           # weight matrix for velocity measurement noise
        self.r_tf = np.linalg.inv(r_tf)             # weight matrix for TF measurement noise
        self.p_matrix = np.linalg.inv(p_matrix)     # weight matrix for arrival cost

        self.c_vel = np.array([[0, 0, 0, 1]])   # output matrix for velocity, shape: (1, 4)
        self.c_tf = np.array([[1, 0, 0, 0],     # output matrix for TF, shape: (3, 4)
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]])

        # # Info holder
        self.initialized = False
        dict_init = {'time':            [],  # 0
                     'sensor.type':     [],  # 1
                     'U.a_z':           [],  # 2
                     'U.w_y':           [],  # 3
                     'velocity':        [],  # 4
                     'tf.x':            [],  # 5
                     'tf.y':            [],  # 6
                     'tf.yaw_z':        [],  # 7
                     'x':               [],  # 8
                     'y':               [],  # 9
                     'theta':           [],  # 10
                     'v':               [],  # 11
                     'w_x':             [],  # 12
                     'w_y':             [],  # 13
                     'w_theta':         [],  # 14
                     'w_v':             [],  # 15
                     't_obj':           [],  # 16
                     't_nlp':           []}  # 17
        self.df = pd.DataFrame(dict_init)
        self.last_u = np.array([[0.], [0.]])
        self.id_list_mea = None
        self.xk0 = None

        # # Objective Function
        self.sym_xx = None          # x_0 ~ x_N, shape:(4, N+1)
        self.sym_xx_next = None     # x_1 ~ x_N, shape:(4, N)
        self.sym_ww = None          # w_0 ~ w_N-1, shape:(4, N)
        # self.sym_uu = None       # u_0 ~ u_N-1, shape:(2, N)
        # self.sym_yy = None       # y_0 ~ y_N, shape:(xx, N+1)

        self.sym_sum_wqw = 0  # SUM ||w||_Q
        self.sym_sum_vrv = 0  # SUM ||y-Cx||_R
        self.sym_arrival_cost = 0  # ||x_0 - x^_0||_P
        self.object_function = 0

        # # Symbolic Functions
        self.f = None               # only used for symbolic calculation, if input numeric, output ca.DM
        self.f_plus_w = None        # only used for symbolic calculation, if input numeric, output ca.DM
        self.y_cx_tf = None
        self.setup_sym_function()   # set up self.f , self.f_plus_w and self.y_cx_tf

        # # Solver options
        self.opts = None
        self.str_solver = None
        self.w_lb = None
        self.w_ub = None
        # self.x_lb = None
        # self.x_ub = None

    def __call__(self, t_stamp: float, data_typ: str, data):
        self.update_data(t_stamp, data_typ, data)

    def initialization(self, solver: str = "ipopt", w_lb: list = None, w_ub: list = None):
        """
        initialize options of NLP
        :param solver:
        :param w_lb: lower bound of w
        :param w_ub: upperbound of w
        :return:
        """

        self.str_solver = solver

        if w_lb is None:
            self.w_lb = [-ca.inf] * 4
        elif isinstance(w_lb, list) and len(w_lb) == 4:
            self.w_lb = w_lb

        if w_ub is None:
            self.w_ub = [ca.inf] * 4
        elif isinstance(w_ub, list) and len(w_ub) == 4:
            self.w_ub = w_ub

        self.opts = {'print_time': False,
                     'ipopt.max_iter': 50,
                     'ipopt.print_level': 0,  # 0: print nothing 3:
                     'ipopt.acceptable_tol': 1e-8,
                     'ipopt.acceptable_obj_change_tol': 1e-6}

    def update_data(self, t_stamp: float, data_typ: str, data):
        if self.initialized:
            if data_typ == 'imu':
                ts = t_stamp - self.df['time'].iloc[-1]
                x_last = self.df[['x', 'y', 'theta', 'v']].iloc[-1].to_numpy().reshape(-1, 1)
                x_curr = self.f_next_x(x_last, self.last_u, ts)
                x_curr = constrain_yaw_angle(x_curr)               # make yaw angle in an interval
                self.last_u = data                                      # only when data from imu come, u will be updated
                dict_data = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [data[0][0]],
                             'U.w_y':           [data[1][0]],
                             # states
                             'x':               [x_curr[0][0]],
                             'y':               [x_curr[1][0]],
                             'theta':           [x_curr[2][0]],
                             'v':               [x_curr[3][0]]}
                self.update_df_by_dict(dict_data)                       # update imu and integrated states

            elif data_typ == 'vel':
                dict_data = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [self.last_u[0][0]],    # repeat last imu data, for convenient MHE
                             'U.w_y':           [self.last_u[1][0]],
                             'velocity':        [data[0][0]],
                             'w_x':             [0],
                             'w_y':             [0],
                             'w_theta':         [0],
                             'w_v':             [0]}
                self.update_df_by_dict(dict_data)   # update data
                t0 = time.perf_counter()
                self.update_obj()
                t1 = time.perf_counter()
                x_curr = self.solve_nlp()
                t2 = time.perf_counter()
                x_curr = constrain_yaw_angle(x_curr)
                self.x2df_all(x_curr)
                self.df.iloc[-1, 16] = t1 - t0
                self.df.iloc[-1, 17] = t2 - t1

            elif data_typ == 'tf':
                dict_data = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [self.last_u[0][0]],    # repeat last imu data, for convenient MHE
                             'U.w_y':           [self.last_u[1][0]],
                             'tf.x':            [data[0][0]],
                             'tf.y':            [data[1][0]],
                             'tf.yaw_z':        [data[2][0]],
                             'w_x':             [0],
                             'w_y':             [0],
                             'w_theta':         [0],
                             'w_v':             [0]}
                self.update_df_by_dict(dict_data)   # update data
                t0 = time.perf_counter()
                self.update_obj()
                t1 = time.perf_counter()
                x_curr = self.solve_nlp()
                t2 = time.perf_counter()
                x_curr = constrain_yaw_angle(x_curr)
                self.x2df_all(x_curr)
                self.df.iloc[-1, 16] = t1 - t0
                self.df.iloc[-1, 17] = t2 - t1

            else:
                raise ValueError("'data_typ' not exist!")
        else:
            if data_typ == 'tf':
                dict_data = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [self.last_u[0][0]],
                             'U.w_y':           [self.last_u[1][0]],
                             'tf.x':            [data[0][0]],
                             'tf.y':            [data[1][0]],
                             'tf.yaw_z':        [data[2][0]],
                             'x':               [data[0][0]],
                             'y':               [data[1][0]],
                             'theta':           [data[2][0]],
                             'v':               [0],
                             'w_x':             [0],
                             'w_y':             [0],
                             'w_theta':         [0],
                             'w_v':             [0],
                             't_obj':           [0],
                             't_nlp':           [0]}
                self.update_df_by_dict(dict_data)  # update data
                self.initialized = True

    def x2df_all(self, x):
        self.df.iloc[-1, 8] = x[0][0]
        self.df.iloc[-1, 9] = x[1][0]
        self.df.iloc[-1, 10] = (x[2][0] + np.pi) % (2 * np.pi) - np.pi        # -pi < theta < pi
        self.df.iloc[-1, 11] = x[3][0]

    def update_df_by_dict(self, dict_data):
        df_curr = pd.DataFrame(dict_data)
        self.df = pd.concat((self.df, df_curr), axis=0, ignore_index=True)
        # self.df.sort_values(by=['time'], inplace=True)

    def update_obj(self):
        # TODO: shorten the computational cost: using Function of casadi and so on
        df_mea = self.df[self.df['sensor.type'] != 'imu']           # drop all imu data

        # df within horizon drops all rows of imu data, (all tf or vel)
        df_mea_horizon = df_mea.iloc[-self.horizon-1:, :]                   # df y in a horizon (len = N + 1)
        self.id_list_mea = df_mea_horizon.index.to_list()                        # id list of y for MHE (len = N)

        # df within horizon includes all data, (first and last row can't be imu, see class method 'update_data()' )
        df_all_horizon = self.df.loc[self.id_list_mea[0]:]              # all data with the horizon (len M >= N)
        id_list_all = df_all_horizon.index.to_list()                        # id list of all data for MHE (len M >= N)

        self.f_horizon = len(id_list_all) - 1                               # fake horizon, M, for integral of symbolic
        self.r_horizon = len(self.id_list_mea) - 1                          # real horizon, N， at begin < horizon

        # param_y = df_mea_horizon[['tf.x','tf.y','tf.yaw_z']].to_numpy().T   # param for MHE shape: ()
        # param_u for MHE, shape: (2,M-1), last u is no used, anyway just a copy, refer to class method update_data()
        param_u = df_all_horizon[['U.a_z', 'U.w_y']].iloc[:-1].to_numpy().T
        param_t = df_all_horizon['time'].to_numpy()                         # param_t for MHE, shape: (M,)
        ts_horizon = np.diff(param_t)                                       # Ts for integral, len = M - 1

        self.sym_ww = ca.SX.sym('w', 4, self.r_horizon)                     # w_0 ~ w_N-1, shape: (4, N)
        self.sym_xx = ca.SX.sym('x', 4, self.r_horizon + 1)                 # x_0 ~ x_N, shape: (4, N+1)
        self.sym_xx_next = ca.SX.sym('xn', 4, self.r_horizon)               # x_1 ~ x_N, shape: (4, N)
        # self.sym_u = ca.SX.sym('u', 2, f_horizon)                    # u_0 ~ u_M-2, shape: (2, M-1)

        # #  States noise part: SUM ||w||_Q
        sum_w_matrix = np.multiply(self.sym_ww, self.q_matrix @ self.sym_ww)  # np.multiply() element wise
        self.sym_sum_wqw = ca.SX.ones(1,4) @ sum_w_matrix @ ca.SX.ones(self.r_horizon,1)  # same as sum loop and faster

        # # Measurement noise part: SUM ||y-Cx||_R
        # Calculate N+1 symbolic x
        for (i, id_i) in enumerate(id_list_all[:-1]):       # loop M-1 times, the M point is y_N and x_N, not included
            # If the current point is measurement, record sym_xx and integrate to the next point using w in this point
            if id_i in self.id_list_mea:
                # find the same item in 'id_list_mea' and get its list id, n < N, n!=N, last point not included
                n = self.id_list_mea.index(id_i)            # n: 0 ~ N-1
                self.sym_xx_next[:,n] = self.f_plus_w(self.sym_xx[:,n], param_u[:,i], ts_horizon[i], self.sym_ww[:,n])
            # If the current point is imu, integrate to the next point using latest w
            else:
                # n kept in the measurement point
                self.sym_xx_next[:,n] = self.f_plus_w(self.sym_xx_next[:,n], param_u[:,i], ts_horizon[i], self.sym_ww[:,n])

        # Calculate N symbolic ||y-Cx||_R iteratively, start from x0 which is also involved in arrival cost
        self.sym_sum_vrv = 0                                            # reset
        for (i, id_i) in enumerate(self.id_list_mea):                        # loop N+1 times
            sensor_typ = df_mea_horizon['sensor.type'].loc[id_i]        # only 'vel' or 'tf'
            if sensor_typ == 'vel':
                y_k = df_mea_horizon['velocity'].loc[id_i]
                c_matrix = self.c_vel
                r_matrix = self.r_vel
                v_k = y_k - c_matrix @ self.sym_xx[:, i]  # start from x1
            elif sensor_typ == 'tf':
                y_k = df_mea_horizon[['tf.x', 'tf.y', 'tf.yaw_z']].loc[id_i].to_numpy().reshape(3, 1)
                # c_matrix = self.c_tf
                r_matrix = self.r_tf
                # the squared difference of yaw angle should be a function with period 2pi
                v_k = self.y_cx_tf(self.sym_xx[:, i], y_k)  # start from x1
                # v_k = y_k - c_matrix @ self.sym_xx[:, i]    # start from x1
            else:
                raise RuntimeError("Something goes wrong by manipulating data!")

            self.sym_sum_vrv += v_k.T @ r_matrix @ v_k

        # # Arrival cost part: ||x_0 - x^_0||_P
        id_x0 = self.id_list_mea[0]
        self.xk0 = self.df[['x', 'y', 'theta', 'v']].iloc[id_x0].to_numpy().reshape(4, 1)
        delta_x0 = self.sym_xx[:, 0] - self.xk0
        delta_x0 = nlinear_diff_yaw_angle(delta_x0)
        self.sym_arrival_cost = delta_x0.T @ self.p_matrix @ delta_x0

        # # Objective function
        self.object_function = self.sym_sum_wqw + self.sym_sum_vrv + self.sym_arrival_cost

    def solve_nlp(self):
        xx_ww = ca.reshape(ca.horzcat(self.sym_xx, self.sym_ww), -1, 1)

        ww_lb = self.w_lb * self.r_horizon
        ww_ub = self.w_ub * self.r_horizon
        lb_xx_ww = [-ca.inf]*4*(self.r_horizon+1) + ww_lb
        ub_xx_ww = [ca.inf]*4*(self.r_horizon+1) + ww_ub
        g = self.sym_xx[:, 1:] - self.sym_xx_next
        g = ca.reshape(g, -1, 1)

        nlp = {'x': xx_ww, 'f': self.object_function, 'g': g}

        init_xx = self.df[['x', 'y', 'theta', 'v']].loc[self.id_list_mea[:-1]].to_numpy()
        init_xx = np.concatenate((init_xx, init_xx[-1, :].reshape(1, 4)), axis=0).reshape(-1)
        # init_ww = np.array([0]*(4*self.r_horizon))
        init_ww = self.df[['w_x', 'w_y', 'w_theta', 'w_v']].loc[self.id_list_mea[:-1]].to_numpy().reshape(-1)

        # solve NLP
        nlp_solver = ca.nlpsol("solver", self.str_solver, nlp, self.opts)
        sol = nlp_solver(x0=ca.DM(np.append(init_xx, init_ww)), lbx=lb_xx_ww, ubx=ub_xx_ww,
                         lbg=[0]*4*self.r_horizon, ubg=[0]*4*self.r_horizon)
        xx_est = np.array(sol['x'][0:4 * (self.r_horizon + 1)]).reshape(self.r_horizon + 1, 4).T
        ww_est = np.array(sol['x'][4 * (self.r_horizon + 1):]).reshape(self.r_horizon, 4)
        self.df.iloc[self.id_list_mea[:-1], 12:16] = ww_est     # update ww

        return xx_est[:, -1].reshape(4, 1)

    @staticmethod
    def ode(x, u):
        """
        ODE equations

         dx = [x[3] * ca.cos(x[2]),
               x[3] * ca.sin(x[2]),
               -u[1],
               u[0]])

        :param x: [x, y, theta, v].T  , all system states
        :param u: [a_z, w_y].T  , 'inputs' of the system
        :return: dot_x
        """
        if type(x) == np.ndarray:
            A = np.array([[0, 0, 0, np.cos(x[2][0])],
                          [0, 0, 0, np.sin(x[2][0])],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        else:  # casadi.DM
            A = np.array([[0, 0, 0, np.cos(x[2])],
                          [0, 0, 0, np.sin(x[2])],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

        B = np.array([[0, 0],
                      [0, 0],
                      [0, -1],
                      [1, 0]])
        dx = A @ x + B @ u

        return dx

    def f_next_x(self, x, u, Ts):
        """
        integral

        :param x:
        :param u:
        :param Ts:
        :return:
        """
        # TODO: add different integration method, switch among them using class property, and initialize it before mhe
        # RK-4
        k1 = self.ode(x, u)
        k2 = self.ode(x + Ts / 2 * k1, u)
        k3 = self.ode(x + Ts / 2 * k2, u)
        k4 = self.ode(x + Ts * k3, u)

        x_next = x + 1 / 6 * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def f_next_x_plus_w(self, x, u, Ts, w):
        """
        integral plus state noise

        regard state noise as constant and integrated along timestep

        :param x:
        :param u:
        :param Ts:
        :return:
        """
        x_next = self.f_next_x(x, u, Ts) + Ts * w
        return x_next

    def setup_sym_function(self):
        x = ca.SX.sym("x", 4, 1)
        u = ca.SX.sym("u", 2, 1)
        ts = ca.SX.sym("ts", 1, 1)
        w = ca.SX.sym("w", 4, 1)

        y_tf = ca.SX.sym("y", 3, 1)
        y_dash = self.c_tf @ x

        y_delta = y_tf - y_dash
        # -pi < theta < pi, -2pi < delta_theta <2pi
        # actual angle difference minimized when delta_theta closes to 0 or 2*k*pi, k is integer
        y_delta = nlinear_diff_yaw_angle(y_delta)

        self.f = ca.Function('f', [x, u, ts], [self.f_next_x(x, u, ts)])
        self.f_plus_w = ca.Function('f_plus_w', [x, u, ts, w], [self.f_next_x_plus_w(x, u, ts, w)])
        self.y_cx_tf = ca.Function('y_cx_tf', [x, y_tf], [y_delta])

