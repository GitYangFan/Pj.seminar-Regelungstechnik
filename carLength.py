import numpy as np
import matplotlib.pyplot as plt
import interpolation
from model import *
from model_kf import *
from get_data import *
from bokeh.io import show, output_file
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def carLength_calculation():
    df_raw_imu = pd.read_csv(r'.\data\data.hamster2__imu.csv')  # get the data from imu
    df_raw_command = pd.read_csv(r'.\data\data.hamster2__command.csv')  # get the data from command
    df_imu = df_raw_imu.iloc[:,0:33]
    df_command = df_raw_command.iloc[:,0:10]

    # df_imu_noNaN = df_imu.dropna()  # drop all rows with NaN data
    # df_command_noNaN = df_command.dropna()  # drop all rows with NaN data

    # transform data to numpy.ndarray
    data_imu = df_imu.to_numpy()
    data_command = df_command.to_numpy()

    # take out useful data from data_imu and data_command
    imu = {}
    command = {}
    imu['time'] = data_imu[:, 3].reshape(1, -1)
    imu['w_y'] = data_imu[:, 19].reshape(1, -1)
    imu['a_z'] = data_imu[:, 32].reshape(1, -1)
    command['time'] = data_command[:, 3].reshape(1, -1)
    command['angle_steering'] = data_command[:, 5].reshape(1, -1)

    # simulation
    a_z = imu['a_z'] - 0.33  # compensate the data offset of linear acceleration
    w_y = imu['w_y']
    U_k = np.append(a_z, w_y, axis=0)  # 'input' of simulation system
    timestamp = imu['time'] - imu['time'][0][0]  # time start from 0s
    Ts = np.diff(timestamp)  # time steps of time stamps, number of data will be -1
    x0 = [0, 0, 0, 0]  # initial states
    x_all = sim(x0, U_k, timestamp)

    v_z = x_all[3,1:]    # get the linear velocity from simulation result

    # find corresponding linear volocity based on timestamp
    car_length = {}
    car_length['time'] = command['time']
    car_length['length'] = np.zeros((1,len(command['time'][0,:])))
    # w_y = interpolation.previous_interpolation(w_y)
    for i in range(4,len(command['time'][0,:])):
        for j in range(len(imu['time'][0,:])):
            # print(imu['time'][0,j])
            if imu['time'][0,j] <= command['time'][0,i] and command['time'][0,i] <= imu['time'][0,j+1] and w_y[0,j]!=0:
                # print(car_length['length'][0,i])
                # print(w_y[0,j])
                # print(np.tan(command['angle_steering'][0,i]) * v_z[j])
                car_length['length'][0,i] = np.tan(command['angle_steering'][0,i]) * v_z[j] / w_y[0,j]

    print('finish!')
    plt.plot(car_length['time'][0,:], car_length['length'][0,:])
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    carLength_calculation()