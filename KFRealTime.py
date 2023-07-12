import numpy as np
import pandas as pd


def ode_x_and_p(x, u, p, q):
    """
    ODE equations

    :param x: [x, y, theta, v].T  , all system states
    :param u: [a_z, w_y].T  , 'inputs' of the system
    :param p: covariance matrix of states
    :param q: covariance matrix of state noise
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
    dx = A @ x + B @ u
    dp = A @ p + p @ A.T + q

    return dx, dp


def f(x, u, ts, p, q):  # RK-4 integral
    """
    Integrate the states and matrix P,
    according to their ODE
    :param x:
    :param u:
    :param ts:
    :param p:
    :param q:
    :return: x_next, P_next
    """
    k1, kp1 = ode_x_and_p(x, u, p, q)
    k2, kp2 = ode_x_and_p(x + ts / 2 * k1, u, p + ts / 2 * kp1, q)
    k3, kp3 = ode_x_and_p(x + ts / 2 * k2, u, p + ts / 2 * kp2, q)
    k4, kp4 = ode_x_and_p(x + ts * k3, u, p + ts / 2 * kp3, q)

    x_next = x + 1 / 6 * ts * (k1 + 2 * k2 + 2 * k3 + k4)
    p_next = p + 1 / 6 * ts * (kp1 + 2 * kp2 + 2 * kp3 + kp4)

    # # 0 <= theta < 2*pi
    # x_next[2][0] = x_next[2][0] % (2 * np.pi)

    # -pi <= theta < pi
    x_next[2][0] = (x_next[2][0] + np.pi) % (2 * np.pi) - np.pi

    return x_next, p_next


class KFRealTime:
    def __init__(self, q_matrix: np.ndarray, r_vel: np.ndarray, r_tf: np.ndarray, p0_matrix: np.ndarray):
        if q_matrix.shape != (4, 4): raise ValueError("'q_matrix' should be a 4*4 numpy array!")
        if r_vel.shape != (1, 1): raise ValueError("'r_vel' should be a 1*1 numpy array!")
        if r_tf.shape != (3, 3): raise ValueError("'r_tf' should be a 3*3 numpy array!")
        if p0_matrix.shape != (4, 4): raise ValueError("'p_matrix' should be a 4*4 numpy array!")

        self.q_matrix = q_matrix                    # covariance matrix of state/system noise
        self.r_vel = r_vel                          # covariance matrix of velocity measurement noise
        self.r_tf = r_tf                            # covariance matrix of TF measurement noise
        self.p0_matrix = p0_matrix                  # initial covariance matrix of state error
        self.C_vel = np.array([[0., 0., 0., 1.]])         # C_vel output matrix for velocity data

        self.C_tf = np.array([[1., 0., 0., 0.],  # C_tf output matrix for tf  data
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.]])
        # # Info holder
        self.initialized = False
        dict_init_all = {'time':                        [],     # 0
                         'sensor.type':                 [],     # 1
                         'U.a_z':                       [],     # 2
                         'U.w_y':                       [],     # 3
                         'velocity':                    [],     # 4
                         'tf.x':                        [],     # 5
                         'tf.y':                        [],     # 6
                         'tf.yaw_z':                    [],     # 7
                         'x':                           [],     # 8
                         'y':                           [],     # 9
                         'theta':                       [],     # 10
                         'v':                           [],     # 11
                         'P11':                         [],     # 12
                         'P12':                         [],     # 13
                         'P13':                         [],     # 14
                         'P14':                         [],     # 15
                         'P21':                         [],     # 16
                         'P22':                         [],     # 17
                         'P23':                         [],     # 18
                         'P24':                         [],     # 19
                         'P31':                         [],     # 20
                         'P32':                         [],     # 21
                         'P33':                         [],     # 22
                         'P34':                         [],     # 23
                         'P41':                         [],     # 24
                         'P42':                         [],     # 25
                         'P43':                         [],     # 26
                         'P44':                         [],     # 27
                         'rt_x':                        [],     # 28
                         'rt_y':                        [],     # 29
                         'rt_theta':                    [],     # 30
                         'rt_v':                        []}     # 31
        self.df = pd.DataFrame(dict_init_all)   # empty at begin

    def __call__(self, t_stamp: float, data_typ: str, data: np.ndarray):
        """

        :param t_stamp:
        :param data_typ: 'imu', 'vel' or 'tf'
        :param data:
        :return:
        """
        self.update_data(t_stamp, data_typ, data)

    def initialization(self):
        pass

    def update_data(self, t_stamp: float, data_typ: str, data):
        if self.initialized:
            if data_typ == 'imu':
                dict_init = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [data[0][0]],
                             'U.w_y':           [data[1][0]]}
                self.update_df_by_dict(dict_init)  # update imu data, states are so far empty

            elif data_typ == 'vel':
                dict_init = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'velocity':        [data[0][0]]}
                self.update_df_by_dict(dict_init)  # update imu data, states are so far empty

            elif data_typ == 'tf':
                dict_init = {'time': [t_stamp],
                             'sensor.type':     [data_typ],
                             'tf.x':            [data[0][0]],
                             'tf.y':            [data[1][0]],
                             'tf.yaw_z':        [data[2][0]]}
                self.update_df_by_dict(dict_init)  # update imu data, states are so far empty

            else:
                raise ValueError("'data_typ' must be in ['imu','vel','tf']!")
            # Do Simulation
            self.update_states()

        else:   # not yet receive data from TF
            if data_typ == 'tf':
                dict_init = {'time':            [t_stamp],
                             'sensor.type':     [data_typ],
                             'U.a_z':           [0],
                             'U.w_y':           [0],
                             'velocity':        [0],
                             'tf.x':            [data[0][0]],
                             'tf.y':            [data[1][0]],
                             'tf.yaw_z':        [data[2][0]],
                             'x':               [data[0][0]],
                             'y':               [data[1][0]],
                             'theta':           [data[2][0]],
                             'v':               [0]}

                self.update_df_by_dict(dict_init)  # update data, states are so far empty
                self.df.iloc[0, 12:28] = self.p0_matrix.reshape(1, 16)
                self.initialized = True

    def update_df_by_dict(self, dict_data):
        df_all_curr = pd.DataFrame(dict_data)
        self.df.sort_index(inplace=True, kind='stable') # sort according to received order
        self.df = pd.concat((self.df, df_all_curr), axis=0, ignore_index=True)  # append the new row to 2 df
        self.df.sort_values(by=['time'], inplace=True, kind='stable')   # sort according to ascending time stamp

    def update_states(self):
        df_time = self.df.iloc[:, 0]            # self.df['time']
        df_sensor_typ = self.df.iloc[:, 1]      # self.df['sensor.type']
        df_u = self.df.iloc[:, 2:4]             # self.df[['U.a_z', 'U.w_y']]
        df_x = self.df.iloc[:, 8:12]            # self.df[['x', 'y', 'theta', 'v']]
        df_vel = self.df.iloc[:, 4]             # self.df['velocity']
        df_tf = self.df.iloc[:, 5:8]            # self.df[['tf.x', 'tf.y', 'tf.yaw_z']]

        last_seq = len(self.df) - 1             # the largest seq number in df
        list_seq = self.df.index.to_list()      # all id list of df
        id_last = list_seq.index(last_seq)      # id of the last seq number (id of the newest data 0~inf)
        id_0 = id_last - 1  # -1~inf            # sim start from id_last-1, whose state is initial state of this loop
        id_u0 = id_0                            # id of U start from id_last - 1 , cuz the last u is useless

        # such id of recent u of newest data until id_u0 = -1, if there is no imu data
        while id_u0 != -1:
            if df_sensor_typ.iloc[id_u0] == 'imu':
                break
            id_u0 -= 1
        # get u0
        if id_u0 != -1:                                             # There are imu data before
            u = df_u.iloc[id_u0].to_numpy().reshape(2,1)
        else:                                                       # There is no imu data yet, use initial u0
            u = df_u.loc[0].to_numpy().reshape(2,1)

        # get x0, P matrix and Ts seq
        if id_0 != -1:                                              # There are data before the newest data
            x = df_x.iloc[id_0].to_numpy().reshape(4, 1)
            p = self.df.iloc[id_0, 12:28].to_numpy().astype(float).reshape(4, 4)
            ts_seq = np.diff(df_time.iloc[id_0:])
        else:   # id_0==-1                                          # There is no data before the newest data
            x = df_x.loc[0].to_numpy().reshape(4, 1)
            p = self.df.loc[0].iloc[12:28].to_numpy().astype(float).reshape(4, 4)
            ts_seq = np.append(0., np.diff(df_time.iloc[0:]))            # append the first ts = 0 to the Ts seq

        # Simulation
        for (i, ts) in enumerate(ts_seq):
            # < Prediction Step >
            x, p = f(x, u, ts, p, self.q_matrix)

            data_typ = df_sensor_typ.iloc[id_last + i]
            if data_typ == 'imu':
                u = df_u.iloc[id_last + i].to_numpy().reshape(2, 1)  # if imu, update u

            # < Correction Step >
            elif data_typ == 'vel':
                y = df_vel.iloc[id_last + i]
                k = p @ self.C_vel.T @ np.linalg.inv(self.C_vel @ p @ self.C_vel.T + self.r_vel)
                x = x + k @ (y - self.C_vel @ x)            # correction
                p = (np.eye(4) - k @ self.C_vel) @ p

            elif data_typ == 'tf':
                y = df_tf.iloc[id_last + i].to_numpy().reshape(3, 1)
                k = p @ self.C_tf.T @ np.linalg.inv(self.C_tf @ p @ self.C_tf.T + self.r_tf)
                y_delta = y - self.C_tf @ x
                # -pi <= delta_theta < pi
                y_delta[2][0] = (y_delta[2][0] + np.pi) % (2 * np.pi) - np.pi
                x = x + k @ y_delta                         # correction
                p = (np.eye(4) - k @ self.C_tf) @ p
            # update or store all x and p in the loop to df
            self.df.iloc[id_last + i, 8:28] = np.append(x.reshape(4), p.reshape(16))
        # store the real time x with the largest time stamp
        self.df.iloc[-1, 28:32] = x.reshape(4)




