import numpy as np
import casadi as ca
import pandas as pd
import time


class MheHamster:
    def __init__(self, horizon, q_matrix: np.ndarray, r_vel: np.ndarray, r_tf: np.ndarray, p_matrix: np.ndarray):

        if horizon < 1: raise ValueError("'horizon' should be >= 1 !")
        if q_matrix.shape != (4, 4): raise ValueError("'q_matrix' should be a 4*4 numpy array!")
        if r_vel.shape != (1, 1): raise ValueError("'r_vel' should be a 1*1 numpy array!")
        if r_tf.shape != (3, 3): raise ValueError("'r_tf' should be a 3*3 numpy array!")

        self.horizon = horizon                  # N
        self.r_horizon = 0                      # real horizon, smaller than horizon at the beginning

        self.q_matrix = q_matrix                # weight matrix for state noise
        self.r_vel = r_vel                      # weight matrix for TF measurement noise
        self.r_tf = r_tf                        # weight matrix for velocity measurement noise
        self.p_matrix = p_matrix                # weight matrix for arrival cost

        self.c_vel = np.array([[0, 0, 0, 1]])   # output matrix for velocity, shape: (1, 4)
        self.c_tf = np.array([[1, 0, 0, 0],     # output matrix for TF, shape: (3, 4)
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]])

        # # Info holder
        dict_init = {'time':            [0.], # TODO: add w, init: all 0, used for initial solution of the NLP
                     'sensor.type':     ['imu'],
                     'velocity':        [np.nan],
                     'tf.x':            [np.nan],
                     'tf.y':            [np.nan],
                     'tf.yaw_z':        [np.nan],
                     'U.a_z':           [0.],
                     'U.w_y':           [0.],
                     'x':               [0.],
                     'y':               [0.],
                     'theta':           [0.],
                     'v':               [0.]}
        self.df_all = pd.DataFrame(dict_init)
        self.last_u = np.array([[0.], [0.]])

        self.id_list_horizon = [0] * horizon

        # # Objective Function
        self.sym_x0 = ca.SX.sym('x0', 4, 1)
        self.sym_w = None       # w_0 ~ w_N-1, shape:(4, N)
        # self.sym_u = None       # u_0 ~ u_N-1, shape:(2, N)
        self.sym_x = None       # x_0 ~ x_N, shape:(4, N+1)
        # self.sym_y = None       # y_0 ~ y_N, shape:(xx, N+1)

        self.sym_sum_wqw = 0                # SUM ||w||_Q
        self.sym_sum_vrv = 0                # SUM ||y-Cx||_R
        self.sym_arrival_cost = 0           # ||x_0 - x^_0||_P
        self.object_function = 0

        self.f = None                       # only used for symbolic calculation, if input numeric, output ca.DM
        self.f_plus_w = None                # only used for symbolic calculation, if input numeric, output ca.DM
        self.setup_integral_function()      # set up self.f and self.f_plus_w

    def init_problem(self, solver: str = "ipopt",):
        nlp =
        opts = {}
        nlp_solver = ca.nlpsol("solver", solver, nlp, opts)

    # def forward(self, t_stamp, sensor_type: str, data: np.ndarray):
        pass

    def update_data(self, t_stamp: float, data_typ: str, data):

        if data_typ == 'imu':
            ts = t_stamp - self.df_all['time'].iloc[-1]
            x_last = self.df_all[['x', 'y', 'theta', 'v']].iloc[-1].to_numpy().reshape(-1, 1)
            x_curr = self.f_next_x(x_last, self.last_u, ts)
            self.last_u = data                      # only when data from imu come, u will be updated
            dict_data = {'time':            [t_stamp],
                         'sensor.type':     [data_typ],
                         'U.a_z':           [data[0][0]],
                         'U.w_y':           [data[1][0]],
                         # states
                         'x':               [x_curr[0][0]],
                         'y':               [x_curr[1][0]],
                         'theta':           [x_curr[2][0]],
                         'v':               [x_curr[3][0]]}
            self.update_df_by_dict(dict_data)   # update data

        elif data_typ == 'vel':
            dict_data = {'time':            [t_stamp],
                         'sensor.type':     [data_typ],
                         'U.a_z':           [self.last_u[0][0]],        # repeat last imu data, for convenient MHE
                         'U.w_y':           [self.last_u[1][0]],
                         'velocity':        [data[0][0]]}
            self.update_df_by_dict(dict_data)   # update data
            self.update_mhe_problem()

        elif data_typ == 'tf':
            # x_curr = self.step_forward(t_stamp)

            dict_data = {'time':            [t_stamp],
                         'sensor.type':     [data_typ],
                         'U.a_z':           [self.last_u[0][0]],        # repeat last imu data, for convenient MHE
                         'U.w_y':           [self.last_u[1][0]],
                         'tf.x':            [data[0][0]],
                         'tf.y':            [data[1][0]],
                         'tf.yaw_z':        [data[2][0]]}
            self.update_df_by_dict(dict_data)   # update data
            self.update_mhe_problem()

        else:
            raise ValueError("'data_typ' not exist!")

    def update_df_by_dict(self, dict_data):
        df_curr = pd.DataFrame(dict_data)
        self.df_all = pd.concat((self.df_all, df_curr), axis=0, ignore_index=True)

    def update_mhe_problem(self):
        # TODO: shorten the computational cost: using Function of casadi and so on
        df_mea = self.df_all[self.df_all['sensor.type'] != 'imu']           # drop all imu data

        # df within horizon drops all rows of imu data, (all tf or vel)
        df_mea_horizon = df_mea.iloc[-(self.horizon+1):, :]                 # df y in a horizon (len = N + 1)
        id_list_mea = df_mea_horizon.index.to_list()                        # id list of y for MHE (len = N + 1)

        # df within horizon includes all data, (first and last row can't be imu, see class method 'update_data()' )
        df_all_horizon = self.df_all.loc[id_list_mea[0]:]                   # all data with the horizon (len >= N + 1)
        id_list_all = df_all_horizon.index.to_list()                        # id list of all data for MHE (len >= N + 1)

        f_horizon = len(id_list_all)                                        # fake horizon, for integral of symbolic
        self.r_horizon = len(id_list_mea) - 1                               # real horizon, N

        # param_y = df_mea_horizon[['tf.x','tf.y','tf.yaw_z']].to_numpy().T   # param for MHE shape: ()
        param_u = df_all_horizon[['U.a_z', 'U.w_y']].to_numpy().T           # param for MHE shape: (2,xx)
        param_t = df_all_horizon['time'].to_numpy()                         # param for MHE shape: (xx,)
        ts_horizon = np.diff(param_t)                                       # Ts for integral, len = f_horizon - 1

        self.sym_w = ca.SX.sym('w', 4, self.r_horizon)      # w_0 ~ w_N-1, shape: (4, N)
        # self.sym_u = ca.SX.sym('u', 2, f_horizon)           # u_0 ~ u_xx, shape: (2, xx)
        self.sym_x = ca.SX.sym('x', 2, self.r_horizon + 1)  # x_0 ~ x_N, shape: (4, N+1)
        self.sym_x[:, 0] = self.sym_x0
        sym_x_next = self.sym_x0

        # #  States noise part: SUM ||w||_Q
        sum_w_matrix = np.multiply(self.sym_w, self.q_matrix @ self.sym_w)  # np.multiply() element wise
        self.sym_sum_wqw = ca.SX.ones(1,4) @ sum_w_matrix @ ca.SX.ones(self.r_horizon,1)  # same as sum loop and faster

        # # Measurement noise part: SUM ||y-Cx||_R
        # Calculate N+1 symbolic x
        for (i, id_i) in enumerate(id_list_all[:-1]):       # loop xx-1 times
            # if the next point has no measurement, no plus w, no recording sym_x
            if df_all_horizon['sensor.type'].loc[id_i+1] == 'imu':
                sym_x_next = self.f(sym_x_next, param_u[:, i], ts_horizon[i])

            # if the next point has measurement, plus w, record sym_x
            else:
                id_j = id_list_mea.index(id_i+1)
                sym_x_next = self.f_plus_w(sym_x_next, param_u[:, i], ts_horizon[i], self.sym_w[:, id_j - 1])
                self.sym_x[:, id_j] = sym_x_next

        self.sym_sum_vrv = 0
        # Calculate N+1 symbolic ||y-Cx||_R iteratively
        for (i, id_i) in enumerate(id_list_mea):            # loop N+1 times
            sensor_typ = df_mea_horizon['sensor.type'].loc[id_i]
            if sensor_typ == 'vel':
                y_k = df_mea_horizon['velocity'].loc[id_i]
                c_matrix = self.c_vel
                r_matrix = self.r_vel
            elif sensor_typ == 'tf':
                y_k = df_mea_horizon[['tf.x', 'tf.y', 'tf.yaw_z']].loc[id_i].to_numpy().reshape(3, 1)
                c_matrix = self.c_tf
                r_matrix = self.r_tf
            else:
                raise RuntimeError("Something goes wrong by manipulating data!")
            v_k = y_k - c_matrix @ self.sym_x[:, i]
            self.sym_sum_vrv += v_k.T @ r_matrix @ v_k

        # # Arrival cost part" ||x_0 - x^_0||_P
        delta_x0 = self.sym_x0 - df_mea_horizon[['x', 'y', 'theta', 'v']].iloc[0].to_numpy().reshape(4,1)
        self.sym_arrival_cost = delta_x0.T @ self.p_matrix @ delta_x0

        # # Objective function
        self.object_function = self.sym_sum_wqw + self.sym_sum_vrv + self.sym_arrival_cost

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

    def setup_integral_function(self):
        x = ca.SX.sym("x", 4,1)
        u = ca.SX.sym("u", 2,1)
        ts = ca.SX.sym("ts", 1,1)
        w = ca.SX.sym("w", 4, 1)

        self.f = ca.Function('f', [x, u, ts], [self.f_next_x(x, u, ts)])
        self.f_plus_w = ca.Function('f_plus_w', [x, u, ts, w], [self.f_next_x_plus_w(x, u, ts, w)])
