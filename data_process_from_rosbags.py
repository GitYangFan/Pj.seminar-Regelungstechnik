from euler_from_quaternion import *
import os
import pandas as pd
import numpy as np

global df_command
global df_imu
global df_velocity
global df_tf
global df_merge


try:
    [data_subfold, save_merged_data]
except ValueError:
    print("Please define the following variables:\n"
          "    data_fold = <subfolder>    # './data/<subfolder>/...'\n"
          "    save_merged_data = bool    # True, if you want to save the merged data along '__time' under './data/<subfolder>\n"
          "                               #'data_merge' is merged from 'df_command', 'df_velocity', 'df_imu' and 'df_tf'\n")

# Load CSV data converted from ROSBAGS in a folder
# Attention: if there is file
try:
    # df_command = pd.read_csv('./data/' + data_subfold + '/data.hamster2__command.csv')
    # df_command = df_command.iloc[:, [0, 3, 5, 7]]
    # or
    df_command = pd.read_csv('./data/' + data_subfold + '/data.hamster2__twist_command.csv')
    df_command = df_command.iloc[:, [0, 3, 4, 5, 6, 7, 8]]

    df_imu = pd.read_csv('./data/' + data_subfold + '/data.hamster2__imu.csv')
    df_imu = df_imu.iloc[:, [0, 3, 18, 19, 20, 30, 31, 32]]

    df_velocity = pd.read_csv('./data/' + data_subfold + '/data.hamster2__velocity.csv')
    df_velocity = df_velocity.iloc[:, [0, 3]]

    df_tf = pd.read_csv('./data/' + data_subfold + '/data.tf.csv')
    df_tf = df_tf.iloc[:, [0, 3, 6, 7, 8, 9, 10, 11, 12]]
except RuntimeError:
    print("Please check if there are the following files under './data/<subfolder> :\n"
          "    data.hamster2__twist_command.csv  #or\n"
          "    data.hamster2__command.csv\n"
          "\n"
          "    data.hamster2__imu.csv\n"
          "    data.hamster2__velocity.csv\n"
          "    data.tf.csv\n"
          "if not, rename the related file or modify this py file 'data_process_from_rosbags.py'\n")


# rename the data
df_command.rename(columns={'/hamster2/command/header.stamp': 'command.header.stamp',
                           '/hamster2/command/drive.steering_angle': 'command.steering_angle',
                           '/hamster2/command/drive.speed': 'command.speed',

                           '/hamster2/twist_command/linear.x': 'command.linear.x',
                           '/hamster2/twist_command/linear.y': 'command.linear.y',
                           '/hamster2/twist_command/linear.z': 'command.linear.z',
                           '/hamster2/twist_command/angular.x': 'command.angular.x',
                           '/hamster2/twist_command/angular.y': 'command.angular.y',
                           '/hamster2/twist_command/angular.z': 'command.angular.z'
                           }, inplace=True)

df_imu.rename(columns={'/hamster2/imu/header.stamp': 'imu.header.stamp',
                       '/hamster2/imu/angular_velocity.x': 'imu.w_x',
                       '/hamster2/imu/angular_velocity.y': 'imu.w_y',
                       '/hamster2/imu/angular_velocity.z': 'imu.w_z',
                       '/hamster2/imu/linear_acceleration.x': 'imu.a_x',
                       '/hamster2/imu/linear_acceleration.y': 'imu.a_y',
                       '/hamster2/imu/linear_acceleration.z': 'imu.a_z',
                       }, inplace=True)

df_velocity.rename(columns={'/hamster2/velocity/data': 'velocity.v'}, inplace=True)

df_tf.rename(columns={'/tf/transforms.0.header.stamp': 'tf.header.stamp',
                      '/tf/transforms.0.transform.translation.x': 'tf.x',
                      '/tf/transforms.0.transform.translation.y': 'tf.y',
                      '/tf/transforms.0.transform.translation.z': 'tf.z',
                      '/tf/transforms.0.transform.rotation.x': 'tf.r_x',
                      '/tf/transforms.0.transform.rotation.y': 'tf.r_y',
                      '/tf/transforms.0.transform.rotation.z': 'tf.r_z',
                      '/tf/transforms.0.transform.rotation.w': 'tf.r_w'
                      }, inplace=True)

# transform quaternion rotation to euler angle, and add it to the data
xyzw = df_tf.iloc[:, 5:9].to_numpy()
euler = np.zeros((xyzw.shape[0], 3))

for i in range(xyzw.shape[0]):
    euler[i][0], euler[i][1], euler[i][2] = euler_from_quaternion(xyzw[i][0], xyzw[i][1], xyzw[i][2], xyzw[i][3])
df_euler = pd.DataFrame(euler, columns=['tf.roll_x', 'tf.pitch_y', 'tf.yaw_z'])

df_tf = pd.concat([df_tf, df_euler], axis=1)

# merge 'df_command', 'df_velocity', 'df_imu' and 'df_tf'
df_merge = df_command.merge(df_velocity, how='outer', on='__time')
df_merge = df_merge.merge(df_imu, how='outer', on='__time')
df_merge = df_merge.merge(df_tf, how='outer', on='__time')

df_merge.sort_values(by=['__time'], inplace=True)
df_merge.reset_index(drop=True, inplace=True)

if save_merged_data is True:
    os.makedirs(r'./data/' + data_subfold, exist_ok=True)
    df_merge.set_index('__time').to_csv(r'./data/' + data_subfold + '/data_merge.csv')
    print("./data/" + data_subfold + "/data_merge.csv is saved\n")

print("Defined Variables:\n"
      "    df_command\n"
      "    df_imu\n"
      "    df_velocity\n"
      "    df_tf\n"
      "    df_merge    # merged from 'df_command', 'df_velocity', 'df_imu' and 'df_tf'\n")

print("Finish!")
