from typing import List, Tuple
import numpy as np
import os
import pandas as pd


def load_data(csv_path: str = './data/data_2023_03_01-16_48_29.csv') -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[
        np.ndarray]]:
    """

    :param
        csv_path: the location where you place the data
    :return:
        time_imu:
        angular_velocity_x:
        angular_velocity_y:
        angular_velocity_z:
        linear_accel_x:
        linear_accel_y:
        linear_accel_z:
    """
    # check the target path
    assert os.path.exists(csv_path), 'Target path does not exist!'

    df = pd.read_csv(csv_path)
    df_imu = df.iloc[:, 0:30]  # take the data from imu only, include global time stamp
    # df_imu_nonan = df_imu.dropna()  # drop all rows with NaN data
    # df_imu_nonan = df_imu_nonan.reset_index(drop=True)

    columns = df_imu.columns.tolist()
    print('Name list of the data:\n')
    for i in range(len(columns)):
        print(columns[i])

    # transform data as numpy.ndarray
    data_imu = df_imu.to_numpy()
    print('\nThe shape of data:', data_imu.shape)

    # time stamp for imu
    time_imu = data_imu[:, 0]
    time_imu_diff = np.diff(time_imu)
    print('Average time step of imu data:', np.average(time_imu_diff))
    # angular_velocity
    angular_velocity_x = data_imu[:, 1]
    angular_velocity_y = data_imu[:, 2]
    angular_velocity_z = data_imu[:, 3]
    # linear_acceleration
    linear_accel_x = data_imu[:, 11]
    linear_accel_y = data_imu[:, 12]
    linear_accel_z = data_imu[:, 13]

    return time_imu, angular_velocity_x, angular_velocity_y, angular_velocity_z, linear_accel_x, linear_accel_y, linear_accel_z
