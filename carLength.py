import numpy as np
import matplotlib.pyplot as plt
import interpolation
from model import *
from model_kf import *
from get_data import *
from bokeh.io import show, output_file
from bokeh.layouts import gridplot
from bokeh.plotting import figure
import pandas as pd


def carLength_calculation():
    df_raw = pd.read_csv(r'.\data\data_new_recombine.csv')  # get the data from imu
    # df_imu = df_raw.iloc[:,4:9]
    # df_command = df_raw.iloc[:,2:3]

    # df_imu_noNaN = df_imu.dropna()  # drop all rows with NaN data
    # df_command_noNaN = df_command.dropna()  # drop all rows with NaN data

    # transform data to numpy.ndarray
    data = df_raw.to_numpy()
    # data_imu = df_imu.to_numpy()
    # data_command = df_command.to_numpy()

    # interpolation
    # data[data == 0] = np.nan
    df = pd.DataFrame(data)
    df_interpolate = df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None,
                   downcast=None)
    df_interpolate.to_csv(r'.\data\data_new_recombine_interpolate.csv', index=False)
    data_interpolate = df_interpolate.values

    # take out useful data from data_imu and data_command
    imu = {}
    command = {}
    imu['time'] = data_interpolate[:, 1].reshape(1, -1)
    imu['w_y'] = data_interpolate[:, 5].reshape(1, -1)
    imu['a_z'] = data_interpolate[:, 9].reshape(1, -1)
    command['time'] = data_interpolate[:, 1].reshape(1, -1)
    command['angle_steering'] = data_interpolate[:, 2].reshape(1, -1)
    command['wheel_speed'] = data_interpolate[:, 3].reshape(1, -1)

    # simulation
    a_z = imu['a_z'] - 0.33  # compensate the data offset of linear acceleration
    w_y = imu['w_y']
    U_k = np.append(a_z, w_y, axis=0)  # 'input' of simulation system
    timestamp = imu['time'] - imu['time'][0][0]  # time start from 0s
    Ts = np.diff(timestamp)  # time steps of time stamps, number of data will be -1
    x0 = [0, 0, 0, 0]  # initial states
    # x_all = sim(x0, U_k, timestamp)
    #
    # v_z = x_all[3,1:]    # get the linear velocity from simulation result

    # find corresponding linear volocity based on timestamp
    car_length = {}
    car_length['time'] = command['time']
    car_length['length'] = np.zeros((1,len(command['time'][0,:])))
    # w_y = interpolation.previous_interpolation(w_y)
    for i in range(10,len(command['time'][0,:])):
        if w_y[0,i] != 0:
            car_length['length'][0,i] = np.tan(command['angle_steering'][0,i]) * command['angle_steering'][0,i] / w_y[0,i]

    car_length_mean = np.mean(car_length['length'])
    print('the mean car length is: ',car_length_mean)
    plt.plot(car_length['time'][0,:], car_length['length'][0,:])
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    carLength_calculation()