import numpy as np
import pandas as pd

df_raw = pd.read_csv(r'.\data\data_2023_03_01-16_48_29.csv')

# Data of IMU sensor
# ======================================================================================================================
df_imu = df_raw.iloc[:, 0:30]  # take the data from imu only, include global time stamp
df_imu_noNaN = df_imu.dropna()  # drop all rows with NaN data
df_imu_noNaN = df_imu_noNaN.reset_index(drop=True)

# imucolumns = df_imu.columns.tolist()
# print('Name list of the data:\n')
# for i in range(len(imucolumns)):
#     print(imucolumns[i])

data_imu = df_imu_noNaN.to_numpy()  # transform data as numpy.ndarray

imu = {}

# time stamp for imu-----------------------------------
imu['time'] = data_imu[:, 0].reshape(1,-1)
# time_imu_diff = np.diff(imu['time'])
# print('Average time step of imu data:', np.average(time_imu_diff))

# angular_velocity--------------------------------------
imu['w_x'] = data_imu[:, 1].reshape(1,-1)
imu['w_y'] = data_imu[:, 2].reshape(1,-1)
imu['w_z'] = data_imu[:, 3].reshape(1,-1)

# linear_acceleration--------------------------------------
imu['a_x'] = data_imu[:, 11].reshape(1,-1)
imu['a_y'] = data_imu[:, 12].reshape(1,-1)
imu['a_z'] = data_imu[:, 13].reshape(1,-1)

# orientation----------------------------------------(all 0, useless)
# imu['o_x'] = data_imu[:, 21].reshape(1,-1)
# imu['o_y'] = data_imu[:, 22].reshape(1,-1)
# imu['o_z'] = data_imu[:, 23].reshape(1,-1)
# imu['o_w'] = data_imu[:, 20].reshape(1,-1)

print('Data of imu: ', list(imu.keys()))


# Data of TF sensor
# ======================================================================================================================
df_tf = pd.concat([df_raw.iloc[:, 0], df_raw.iloc[:, 33:40]], axis=1)  # take the data from tf only, include global time stamp
df_tf_noNaN = df_tf.dropna()  # drop all rows with NaN data
df_tf_noNaN = df_tf_noNaN.reset_index(drop=True)

# tfcolumns = df_tf.columns.tolist()
# print('Name list of the data:\n')
# for i in range(len(tfcolumns)):
#     print(tfcolumns[i])

data_tf = df_tf_noNaN.to_numpy()  # transform data as numpy.ndarray

tf = {}

# time stamp for tf
tf['time'] = data_tf[:, 0].reshape(1,-1)
# time_tf_diff = np.diff(tf['time'])
# print('Average time step of tf data:', np.average(time_tf_diff))

# rotation
tf['r_w'] = data_tf[:, 1].reshape(1,-1)
tf['r_x'] = data_tf[:, 2].reshape(1,-1)
tf['r_y'] = data_tf[:, 3].reshape(1,-1)
tf['r_z'] = data_tf[:, 4].reshape(1,-1)

# translation
tf['x'] = data_tf[:, 5].reshape(1,-1)
tf['y'] = data_tf[:, 6].reshape(1,-1)
tf['z'] = data_tf[:, 7].reshape(1,-1)

print('Data of tf: ', list(tf.keys()))

# Data of velocity
# ======================================================================================================================
df_vel = df_raw.iloc[:, [0, 30]]  # take the data from velocity only, include global time stamp
df_vel_noNaN = df_vel.dropna()  # drop all rows with NaN data
df_vel_noNaN = df_vel_noNaN.reset_index(drop=True)

data_vel = df_vel_noNaN.to_numpy()

vel = {}

# time stamp for velocity-----------------------------
vel['time'] = data_vel[:, 0]
# time_vel_diff = np.diff(time_vel)
# print('Average time step of velocity data:', np.average(time_vel_diff))

# velocity---------------------------------------
vel['velocity'] = data_vel[:, 1]

print('Data of velocity: ', list(vel.keys()))