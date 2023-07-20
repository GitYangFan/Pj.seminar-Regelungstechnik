import numpy as np
import casadi as ca
import pandas as pd
import time


def nlinear_diff_yaw_angle(x):
    # select one of the following syntax
    # x[2] = ca.tan(0.5 * x[2])
    x[2] = 2 - ca.cos(x[2])
    # x[2] = ca.fmod(x[2] + np.pi, 2*np.pi) - np.pi
    # x[2] = 2*ca.sin(x[2]/2)
    return x


def constrain_yaw_angle(x):
    """

    :param x: shape: (4, n)
    :return:
    """
    # # 0 <= theta < 2*pi
    # x[2,:] = np.fmod(x[2,:], 2 * np.pi)

    # or
    # -pi <= theta < pi
    x[2,:] = np.fmod(x[2,:] + np.pi, 2 * np.pi) - np.pi
    return x


def np_insert(arr: np.ndarray, index: int, obj: np.ndarray):
    """
    insert 'obj' to the 'index' column of 'arr'
    """
    return np.concatenate((arr[:, :index], obj, arr[:, index:]), axis=1)


class MHERealTime:
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
        self.i = None               # id number of the last item in list
        self.list_id_mea = None     # id list of measurements in a horizon
        self.list_id_all = None     # id list of all in a horizon
        self.xk0 = None
        self.dataset = {'seq':      [],                 # 0  | list[int],   sorted seq according to time
                        'time':     [],                 # 1  | list[float],
                        'sensor':   [],                 # 2  | list[str],   items in ['imu', 'vel', 'tf']
                        'data':     [],                 # 3  | list[(2,1), (1,1), (3,1)],   data from imu, vel, tf
                        'X':        np.array([[]]*4),   # 4  | (4,n) array, states updated continually
                        'X_rt':     np.array([[]]*4),   # 5  | (4,n) array, states in real time  (sent to controller)
                        'W':        np.array([[]]*4),   # 6  | (4,n) array, state noise, result of NLP
                        't_com':    []}                 # 7  | list[float], computational time

        # self.last_u = np.array([[0.], [0.]])

        # # Objective Function
        self.sym_xx = None          # x_0 ~ x_N, shape:(4, N+1)
        self.sym_xx_next = None     # x_1 ~ x_N, shape:(4, N)
        self.sym_xx_all = None      # x_0 ~ x_M, shape: (4, M+1)
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
            self.w_lb = [-ca.inf]*4
        elif isinstance(w_lb, list) and len(w_lb) == 4:
            self.w_lb = w_lb

        if w_ub is None:
            self.w_ub = [ca.inf]*4
        elif isinstance(w_ub, list) and len(w_ub) == 4:
            self.w_ub = w_ub

        self.opts = {'print_time': False,
                     'ipopt.max_iter': 50,
                     'ipopt.print_level': 0,    # 0: print nothing 3:
                     'ipopt.acceptable_tol': 1e-8,
                     'ipopt.acceptable_obj_change_tol': 1e-6}
        # run a very simple nlp to initialize, avoid a large computational time at the beginning
        x = ca.SX.sym('x', 1)
        nlp = {'x': x, 'f': x**2}
        nlp_solver = ca.nlpsol("solver", self.str_solver, nlp, self.opts)
        sol = nlp_solver(x0=0)

    def update_data(self, t_stamp: float, data_typ: str, data):
        if self.initialized:
            t0 = time.perf_counter()
            self.i += 1
            id_0 = self.i
            while t_stamp < self.dataset['time'][id_0 - 1]:
                id_0 -= 1
                if id_0 == 0:
                    break

            # fill the state X at the time stamp of data (by integrating last X or initial X0)
            # purpose to serve as initial state of NLP
            u0 = self._get_last_u(id_0)
            if id_0 != 0:   # the time stamp of data is not the smallest
                ts = t_stamp - self.dataset['time'][id_0 - 1]
                x = self.f_next_x(self.dataset['X'][:, id_0-1:id_0], u0, ts)
                x = constrain_yaw_angle(x)
            else:           # the time stamp of data is the smallest
                i0 = self.dataset['seq'].index(0)
                x = self.dataset['X'][:, i0:i0+1]

            if data_typ == 'imu':
                if id_0 == self.i:
                    x_rt = x
                else:
                    x_rt = np.array([[np.nan]] * 4)
                self.dataset['seq'].insert(id_0, self.i)  # sorted according to time stamp
                self.dataset['time'].insert(id_0, t_stamp)
                self.dataset['sensor'].insert(id_0, data_typ)
                self.dataset['data'].insert(id_0, data)
                self.dataset['X'] = np_insert(self.dataset['X'], id_0, x)
                self.dataset['X_rt'] = np_insert(self.dataset['X_rt'], id_0, x_rt)
                self.dataset['W'] = np_insert(self.dataset['W'], id_0, np.array([[np.nan]] * 4))

            else:   # data_typ == 'vel' or 'tf'
                self.dataset['seq'].insert(id_0, self.i)  # sorted according to time stamp
                self.dataset['time'].insert(id_0, t_stamp)
                self.dataset['sensor'].insert(id_0, data_typ)
                self.dataset['data'].insert(id_0, data)
                self.dataset['X'] = np_insert(self.dataset['X'], id_0, x)
                self.dataset['X_rt'] = np_insert(self.dataset['X_rt'], id_0, np.array([[np.nan]] * 4))
                self.dataset['W'] = np_insert(self.dataset['W'], id_0, np.array([[0.]] * 4))

                self.update_obj(id_0, u0)
                xx_all_est, ww_est = self.solve_nlp()   # xx_all_est: (4, M+1) array; ww_est: (4, N) array
                xx_all_est = constrain_yaw_angle(xx_all_est)

                self.dataset['X'][:, self.list_id_all] = xx_all_est     # update all X
                self.dataset['W'][:, self.list_id_mea[:-1]] = ww_est    # update W

                list_id_res = list(range(self.list_id_mea[-1], self.i+1))   # 1st item is id of the last result of NLP
                if len(list_id_res) > 1:    # the last item in 'list_id_all' is not with the largest time stamp
                    # the rest of data are all imu
                    x = xx_all_est[:, -1:]
                    u = self._get_last_u(list_id_res[0])
                    list_ts = np.diff(self.dataset['time'][list_id_res[0]:])
                    for (i, id_r) in enumerate(list_id_res[1:]):    # exclude the first item in 'list_id_res'
                        x = self.f_next_x(x, u, list_ts[i])
                        x = constrain_yaw_angle(x)
                        u = self.dataset['data'][id_r]
                        self.dataset['X'][:, id_r:id_r+1] = x
                    self.dataset['X_rt'][:, -1:] = x

            t1 = time.perf_counter()
            self.dataset['t_com'].insert(id_0, t1 - t0)

        else:   # not yet receive data from TF
            if data_typ == 'tf':
                t0 = time.perf_counter()
                self.i = 0
                self.dataset['seq'].append(self.i)
                self.dataset['time'].append(t_stamp)
                self.dataset['sensor'].append(data_typ)
                self.dataset['data'].append(data)
                self.dataset['X'] = np.append(data, [[0.]], axis=0)  # append v=0 at the end of tf data
                self.dataset['X_rt'] = np.array([[np.nan]] * 4)  # X_rt set to nan
                self.dataset['W'] = np.array([[0.]] * 4)

                self.initialized = True  # once receive data from TF, initialize X

                t1 = time.perf_counter()
                self.dataset['t_com'].append(t1 - t0)

    def _get_last_u(self, id_start):
        """
        such recent u before id_start, if there is no imu data before, u = 0
        :param id_start: index start to such
        :return: u
        """
        id_u0 = id_start - 1
        while id_u0 != -1:
            if self.dataset['sensor'][id_u0] == 'imu':
                break
            id_u0 -= 1
        # get u0
        if id_u0 != -1:  # There are imu data before
            u = self.dataset['data'][id_u0]
        else:  # There is no imu data yet, use zero u0
            u = np.array([[0.], [0.]])
        return u

    def update_obj(self, id_0, u):
        """
        self.dataset['sensor'][id_0] is definitely not 'imu'
        :param id_0:
        :param u: recent u before id_0
        :return:
        """
        # get id list of measurements and all, the last item of them is mea id and maybe not with the largest time stamp
        # len(list_id_mea) >= horizon and list_id_mea include the newest data
        self.list_id_mea = []
        id_mea = self.i
        while len(self.list_id_mea) < self.horizon or id_mea >= id_0:
            if self.dataset['sensor'][id_mea] != 'imu':
                self.list_id_mea.append(id_mea)
            if id_mea == 0:     # at beginning, id_mea can reach to 0, real horizon <= horizon
                break
            id_mea -= 1
        self.list_id_mea.reverse()
        self.list_id_all = list(range(self.list_id_mea[0], self.list_id_mea[-1] + 1))

        self.f_horizon = len(self.list_id_all) - 1                      # fake horizon, M, for integral of symbolic
        self.r_horizon = len(self.list_id_mea) - 1                      # real horizon, N， at begin < horizon, later >=

        list_t = self.dataset['time'][self.list_id_all[0]:self.list_id_all[-1]+1] # list_t for MHE, shape: (M,)
        list_ts = np.diff(list_t)                                       # Ts list for integral, len = M - 1

        self.sym_ww = ca.SX.sym('w', 4, self.r_horizon)                 # w_0 ~ w_N-1, shape: (4, N)
        self.sym_xx = ca.SX.sym('x', 4, self.r_horizon + 1)             # x_0 ~ x_N, shape: (4, N+1)
        self.sym_xx_next = ca.SX.sym('xn', 4, self.r_horizon)           # x_1 ~ x_N, shape: (4, N)
        self.sym_xx_all = ca.SX.sym('xa', 4, self.f_horizon + 1)    # x_0 ~ x_M, shape: (4, M+1), for update all X later

        # #  States noise part: SUM ||w||_Q
        sum_w_matrix = np.multiply(self.sym_ww, self.q_matrix @ self.sym_ww)  # np.multiply() element wise
        self.sym_sum_wqw = ca.SX.ones(1,4) @ sum_w_matrix @ ca.SX.ones(self.r_horizon,1)  # same as sum loop and faster

        # # Measurement noise part: SUM ||y-Cx||_R
        # Calculate N+1 symbolic x
        for (i, id_i) in enumerate(self.list_id_all[:-1]):       # loop M times, the M point is y_N and x_N, excluded
            # If the current point is measurement, record sym_xx and integrate to the next point using w in this point
            if id_i in self.list_id_mea:
                # find the same item in 'list_id_mea' and get its list id, n < N, n!=N, last point excluded
                n = self.list_id_mea.index(id_i)            # n: 0 ~ N-1
                self.sym_xx_all[:, i] = self.sym_xx[:, n]
                self.sym_xx_next[:, n] = self.f_plus_w(self.sym_xx[:, n], u, list_ts[i], self.sym_ww[:, n])

            # If the current point is imu, integrate to the next point using latest w
            else:
                # n kept in the measurement point
                self.sym_xx_all[:, i] = self.sym_xx_next[:, n]
                self.sym_xx_next[:, n] = self.f_plus_w(self.sym_xx_next[:, n], u, list_ts[i], self.sym_ww[:, n])
                u = self.dataset['data'][id_i]              # update u
        self.sym_xx_all[:, -1] = self.sym_xx[:, -1]

        # Calculate N symbolic ||y-Cx||_R iteratively, start from x0 which is also involved in arrival cost
        self.sym_sum_vrv = 0                                # reset sum_vrv
        for (i, id_i) in enumerate(self.list_id_mea):                        # loop N+1 times
            sensor_typ = self.dataset['sensor'][id_i]        # only 'vel' or 'tf'
            y_k = self.dataset['data'][id_i]
            if sensor_typ == 'vel':
                c_matrix = self.c_vel
                r_matrix = self.r_vel
                v_k = y_k - c_matrix @ self.sym_xx[:, i]
            elif sensor_typ == 'tf':
                # c_matrix = self.c_tf
                r_matrix = self.r_tf
                # the squared difference of yaw angle should be a function with period 2pi
                v_k = self.y_cx_tf(self.sym_xx[:, i], y_k)
                # v_k = y_k - c_matrix @ self.sym_xx[:, i]
            else:
                raise RuntimeError("Something goes wrong by manipulating data!")

            self.sym_sum_vrv += v_k.T @ r_matrix @ v_k

        # # Arrival cost part: ||x_0 - x^_0||_P
        self.xk0 = self.dataset['X'][:, self.list_id_mea[0]:self.list_id_mea[0]+1]
        delta_x0 = self.sym_xx[:, 0] - self.xk0
        delta_x0 = nlinear_diff_yaw_angle(delta_x0)
        self.sym_arrival_cost = delta_x0.T @ self.p_matrix @ delta_x0
        self.sym_arrival_cost = self.sym_arrival_cost * self.r_horizon  # Avoid varying weighting with length of horizon
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

        init_xx = self.dataset['X'][:, self.list_id_mea].T.reshape(-1)
        init_ww = np.array([0]*(4*self.r_horizon))
        # init_ww = self.dataset['W'][:, self.list_id_mea[:-1]].T.reshape(-1)

        # solve NLP
        nlp_solver = ca.nlpsol("solver", self.str_solver, nlp, self.opts)
        sol = nlp_solver(x0=ca.DM(np.append(init_xx, init_ww)), lbx=lb_xx_ww, ubx=ub_xx_ww,
                         lbg=[0]*4*self.r_horizon, ubg=[0]*4*self.r_horizon)
        xx_est = np.array(sol['x'][0:4 * (self.r_horizon + 1)]).reshape(self.r_horizon + 1, 4).T
        ww_est = np.array(sol['x'][4 * (self.r_horizon + 1):]).reshape(self.r_horizon, 4).T

        result = ca.Function('result', [self.sym_xx, self.sym_ww], [self.sym_xx_all])
        xx_all_est = np.array(result(xx_est, ww_est))

        return xx_all_est, ww_est

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

    def f_next_x(self, x, u, ts):
        """
        integral

        :param x:
        :param u:
        :param ts:
        :return:
        """
        # TODO: add different integration method, switch among them using class property, and initialize it before mhe
        # RK-4
        k1 = self.ode(x, u)
        k2 = self.ode(x + ts / 2 * k1, u)
        k3 = self.ode(x + ts / 2 * k2, u)
        k4 = self.ode(x + ts * k3, u)

        x_next = x + 1 / 6 * ts * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def f_next_x_plus_w(self, x, u, ts, w):
        """
        integral plus state noise

        regard state noise as constant and integrated along timestep
        :param x:
        :param u:
        :param ts:
        :param w:
        :return:
        """
        x_next = self.f_next_x(x, u, ts) + ts * w
        return x_next

    def setup_sym_function(self):
        x = ca.SX.sym("x", 4, 1)
        u = ca.SX.sym("u", 2, 1)
        ts = ca.SX.sym("ts", 1, 1)
        w = ca.SX.sym("w", 4, 1)

        y_tf = ca.SX.sym("y", 3, 1)
        y_dach = self.c_tf @ x

        y_delta = y_tf - y_dach
        # -pi < theta < pi, -2pi < delta_theta <2pi
        # actual angle difference minimized when delta_theta closes to 0 or 2*k*pi, k is integer
        y_delta = nlinear_diff_yaw_angle(y_delta)

        self.f = ca.Function('f', [x, u, ts], [self.f_next_x(x, u, ts)])
        self.f_plus_w = ca.Function('f_plus_w', [x, u, ts, w], [self.f_next_x_plus_w(x, u, ts, w)])
        self.y_cx_tf = ca.Function('y_cx_tf', [x, y_tf], [y_delta])

    @property
    def df(self):
        n = self.i + 1
        columns = ['rt_seq',        # 0
                   'time',          # 1
                   'sensor.type',   # 2
                   'U.a_z',         # 3
                   'U.w_y',         # 4
                   'velocity',      # 5
                   'tf.x',          # 6
                   'tf.y',          # 7
                   'tf.yaw_z',      # 8
                   'x',             # 9
                   'y',             # 10
                   'theta',         # 11
                   'v',             # 12
                   'rt_x',          # 13
                   'rt_y',          # 14
                   'rt_theta',      # 15
                   'rt_v',          # 16
                   'w_x',           # 17
                   'w_y',           # 18
                   'w_theta',       # 19
                   'w_v',           # 20
                   't_com']         # 21
        df = pd.DataFrame(data=[[np.nan] * 22] * n, columns=columns)

        df.iloc[:, 0] = self.dataset['seq']
        df.iloc[:, 1] = self.dataset['time']
        df.iloc[:, 2] = self.dataset['sensor']
        for i in range(n):
            sensor = self.dataset['sensor'][i]
            if sensor == 'imu':
                df.iloc[i, 3:5] = self.dataset['data'][i].reshape(2)
            elif sensor == 'vel':
                df.iloc[i, 5] = self.dataset['data'][i].reshape(1)
            elif sensor == 'tf':
                df.iloc[i, 6:9] = self.dataset['data'][i].reshape(3)
        df.iloc[:, 9:13] = self.dataset['X'].T
        df.iloc[:, 13:17] = self.dataset['X_rt'].T
        df.iloc[:, 17:21] = self.dataset['W'].T
        df.iloc[:, 21] = self.dataset['t_com']
        return df

