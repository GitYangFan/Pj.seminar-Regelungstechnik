import numpy as np

from model import *
from model_kf import *
from model_all import *
from get_data import *

from bokeh.io import show, output_file
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def plot_config(width: int, height: int, title=None, x_label: str = None,
                y_label: str = None):
    p = figure(title=title, background_fill_color="#fafafa", width=width, height=height)
    p.output_backend = "svg"  # save plot in HTML as SVG
    p.title.text_font_size = "15pt"
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    return p


def simulation():
    # ==========================================Simulation================================================
    a_z = imu['a_z'] - 0.33  # compensate the data offset of linear acceleration
    w_y = imu['w_y']
    U_k = np.append(a_z, w_y, axis=0)  # 'input' of simulation system

    timestamp = imu['time'] - imu['time'][0][0]  # global time stamp start from 0s
    Ts = np.diff(timestamp)  # time steps of time stamps, number of data will be -1

    x0 = [0, 0, 0, 0]  # initial states
    x_all = sim(x0, U_k, timestamp)
    print("shape of result", x_all.shape)

    # ============================================ Plot =============================================
    p_total = []

    # simulation curve
    p = plot_config(width=600, height=600, title='imu curve', x_label='x [ unknown ]', y_label='y [ unknown ]')
    p.circle(x_all[0][0], x_all[1][0], fill_color="red", legend_label='start', size=10)
    p.line(x=x_all[0], y=x_all[1], legend_label='curve', line_width=1, line_color='blue')
    p.asterisk(x_all[0][-1], x_all[1][-1], line_color="green", legend_label='end', size=10)

    p.circle(tf['x'][0][0], tf['y'][0][0], fill_color="red", legend_label='start', size=10)
    p.line(x=tf['x'][0], y=tf['y'][0], legend_label='curve', line_width=1, line_color='black')
    p.asterisk(tf['x'][0][-1], tf['y'][0][-1], line_color="green", legend_label='end', size=10)
    p_total.append(p)

    # simulation states
    p2 = plot_config(width=900, height=600, title='system states', x_label='t [ s ]', y_label='states [ unknown ] ')
    p2.line(x=timestamp[0], y=x_all[0], legend_label='x', line_width=2, line_color='green')
    p2.line(x=timestamp[0], y=x_all[1], legend_label='y', line_width=2, line_color='black')
    p2.line(x=timestamp[0], y=x_all[2], legend_label='theta', line_width=2, line_color='royalblue')
    p2.line(x=timestamp[0], y=x_all[3], legend_label='v', line_width=2, line_color='crimson')
    p_total.append(p2)

    show(gridplot(p_total, ncols=2))


def simulation_kf():
    # ==========================================Simulation================================================
    df_raw = pd.read_csv(r'data/data_2023_03_01-16_48_29/data_2023_03_01-16_48_29.csv')

    df = df_raw.iloc[:, 0:31]  # take the data from imu and 'velocity', include global time stamp

    # drop the rows where both imu and 'velocity' are NaN
    df_imu_vel = df[df.iloc[:, 29].notna() | df.iloc[:, 30].notna()].reset_index(drop=True)
    data = df_imu_vel.to_numpy()

    # needed data for simulation
    u_k = np.append(data[:, 13].reshape(1, -1), data[:, 2].reshape(1, -1), axis=0)  # a_z, w_y, with nan, shape: (2, n)
    y_k = - data[:, 30] / 1000  # unit of data is mm/s and data is inverse
    switch = np.isnan(data[:, -1]) # switch, to determine prediction or correction in KF, if False,then correction
    Ts = np.diff(data[:, 0])

    x0 = [2.668, -3.346, 0, 0]
    Q = np.diag([1, 1, 1, 1])
    R = np.array([[0.1]])
    P0 = np.diag([10, 10, 10, 10])

    x_all = sim_kf(x0, u_k, y_k, switch, Ts, Q, R, P0)
    print("shape of result", x_all.shape)

    # ============================================ Plot =============================================
    p_total = []

    # simulation curve
    p = plot_config(width=600, height=600, title='curve', x_label='x [ unknown ]', y_label='y [ unknown ]')
    p.circle(x_all[0][0], x_all[1][0], fill_color="red", legend_label='start_imu', size=10)
    p.circle(x=x_all[0], y=x_all[1], legend_label='curve', line_width=1, line_color='blue')
    p.asterisk(x_all[0][-1], x_all[1][-1], line_color="green", legend_label='end_imu', size=10)

    # tf curve
    p.circle(tf['x'][0][0], tf['y'][0][0], fill_color="red", legend_label='start_tf', size=10)
    p.circle(x=tf['x'][0], y=tf['y'][0], legend_label='curve', line_width=1, line_color='black')
    p.asterisk(tf['x'][0][-1], tf['y'][0][-1], line_color="green", legend_label='end_tf', size=10)
    p_total.append(p)

    # simulation states
    p2 = plot_config(width=900, height=600, title='system states', x_label='t [ s ]', y_label='states [ unknown ] ')
    p2.line(x=data[:, 0], y=x_all[0], legend_label='x', line_width=2, line_color='green')
    p2.line(x=data[:, 0], y=x_all[1], legend_label='y', line_width=2, line_color='black')
    p2.line(x=data[:, 0], y=x_all[2], legend_label='theta', line_width=2, line_color='royalblue')
    p2.line(x=data[:, 0], y=x_all[3], legend_label='v', line_width=2, line_color='crimson')
    p_total.append(p2)
    output_file(filename="./simulation_kf.html", title="KF_imu_velocity vs tf")
    show(gridplot(p_total, ncols=2))


def simulation_kf_all_data():
    # ==========================================Simulation================================================
    df_raw = pd.read_csv(r'data/data_2023_03_01-16_48_29/data_2023_03_01-16_48_29.csv')

    # take the data from imu, velocity and tf, including global time stamp
    df = df_raw.iloc[:, [i for i in range(31)] + [t for t in range(32, 40)]]
    df = df[df.iloc[:, 29].notna() | df.iloc[:, 30].notna() | df.iloc[:, 31].notna()].reset_index(
        drop=True)  # drop the rows of voltage
    data = df.to_numpy()

    # calculate the yaw angle of tf
    wxyz = data[:, 32:36]  # wxyz = [w, x, y, z]
    print(wxyz.shape[0])
    yaw_z = np.zeros((1, wxyz.shape[0]))    # shape: (1, n)
    for i in range(wxyz.shape[0]):
        _, _, yaw_z[0][i] = euler_from_quaternion(wxyz[i][1], wxyz[i][2], wxyz[i][3], wxyz[i][0])

    u_k = np.append(data[:, 13].reshape(1, -1), data[:, 2].reshape(1, -1), axis=0)  # a_z, w_y, with nan, shape: (2, n)
    y_k_vel = - data[:, 30] / 1000  # unit of data is mm/s and data is inverse
    y_k_tf = np.concatenate((data[:, 36].reshape(1, -1), data[:, 37].reshape(1, -1), yaw_z), axis=0)  # [x, y, yaw_angle].T , shape: (3, n)
    Ts = np.diff(data[:, 0])    # get time steps
    # Switch, if the data are NaN, True. To determine prediction or correction in KF
    s_imu = np.isnan(data[:, 29])
    s_vel = np.isnan(data[:, 30])
    s_tf = np.isnan(data[:, 31])

    # Initial
    x0 = [2.669, -3.347, 0, 0]
    Q = np.diag([1, 1, 1, 1])
    R_v = np.array([[0.1]])
    R_t = np.diag([0.1, 0.1, 0.1])
    P0 = np.diag([10, 10, 10, 10])

    x_all = sim_all(x0, u_k, Ts, y_k_vel, y_k_tf, s_imu, s_vel, s_tf, Q, R_v, R_t, P0)
    print(x_all.shape)

    # ============================================ Plot =============================================
    p_total = []

    # simulation curve
    p = plot_config(width=600, height=600, title='curve', x_label='x [ unknown ]', y_label='y [ unknown ]')
    p.circle(x_all[0][0], x_all[1][0], fill_color="red", legend_label='start_imu', size=2)
    p.circle(x=x_all[0][0:3000], y=x_all[1][0:3000], legend_label='curve', line_width=1, line_color='blue')
    p.asterisk(x_all[0][-1], x_all[1][-1], line_color="black", legend_label='end_imu', size=20)

    # tf curve
    p.circle(tf['x'][0][0], tf['y'][0][0], fill_color="red", legend_label='start_tf', size=10)
    p.circle(x=tf['x'][0], y=tf['y'][0], legend_label='curve', line_width=1, line_color='black')
    p.asterisk(tf['x'][0][-1], tf['y'][0][-1], line_color="green", legend_label='end_tf', size=20)
    p_total.append(p)

    # simulation states
    p2 = plot_config(width=900, height=600, title='system states', x_label='t [ s ]', y_label='states [ unknown ] ')
    p2.line(x=data[:, 0], y=x_all[0], legend_label='x', line_width=2, line_color='green')
    p2.line(x=data[:, 0], y=x_all[1], legend_label='y', line_width=2, line_color='black')
    p2.line(x=data[:, 0], y=x_all[2], legend_label='theta', line_width=2, line_color='royalblue')
    p2.line(x=data[:, 0], y=x_all[3], legend_label='v', line_width=2, line_color='crimson')
    p_total.append(p2)
    output_file(filename="./simulation_all.html", title="KF_all vs tf")
    show(gridplot(p_total, ncols=2))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # simulation_kf()
    simulation_kf_all_data()