from euler_from_quaternion import *
import os
import pandas as pd
import numpy as np
"""
# Template:
data_subfold = "rosbag2_2023_06_16-16_50_38"
merge_command = False
save_merged_data = False        # optional
"""


# Make sure that 'data_subfold', 'save_merged_data' are defined
try:
    [data_subfold, save_merged_data]
except ValueError:
    print("Please define the following variables:\n"
          "    data_fold = <subfolder>    # './data/<subfolder>/...'\n"
          "    save_merged_data = bool    # True, if you want to save the merged data along '__time' under './data/<subfolder>\n"
          "                               #'data_merge' is merged from 'df_command', 'df_velocity', 'df_imu' and 'df_tf'\n")

# # Load CSV data converted from ROSBAGS in a folder
df_command = None
df_imu = None
df_velocity = None
df_tf = None
list_df_origin = [df_command, df_imu, df_velocity, df_tf]
# Attention: command file might be different in different rosbags
try:
    df_command = pd.read_csv('./data/' + data_subfold + '/data.hamster2__command.csv')
    df_command = df_command.iloc[:, [0, 3, 5, 7]]
    print("'data.hamster2__command.csv' loaded to 'df_command'")
    # rename the data
    df_command.rename(columns={'/hamster2/command/header.stamp': 'command.header.stamp',
                               '/hamster2/command/drive.steering_angle': 'command.steering_angle',
                               '/hamster2/command/drive.speed': 'command.speed'
                               }, inplace=True)
except:

    try:
        df_command = pd.read_csv('./data/' + data_subfold + '/data.hamster2__twist_command.csv')
        df_command = df_command.iloc[:, [0, 3, 4, 5, 6, 7, 8]]
        print("'data.hamster2__twist_command.csv' loaded to 'df_command'")
        # rename the data
        df_command.rename(columns={'/hamster2/twist_command/linear.x': 'command.linear.x',
                                   '/hamster2/twist_command/linear.y': 'command.linear.y',
                                   '/hamster2/twist_command/linear.z': 'command.linear.z',
                                   '/hamster2/twist_command/angular.x': 'command.angular.x',
                                   '/hamster2/twist_command/angular.y': 'command.angular.y',
                                   '/hamster2/twist_command/angular.z': 'command.angular.z'
                                   }, inplace=True)
    except:
        print("Neither 'data.hamster2__command.csv'\n"
              "Nor 'data.hamster2__twist_command.csv' exist!!!!!")

try:
    df_imu = pd.read_csv('./data/' + data_subfold + '/data.hamster2__imu.csv')
    df_imu = df_imu.iloc[:, [0, 3, 18, 19, 20, 30, 31, 32]]
    print("'data.hamster2__imu.csv' loaded to 'df_imu'")
    # rename the data
    df_imu.rename(columns={'/hamster2/imu/header.stamp': 'imu.header.stamp',
                           '/hamster2/imu/angular_velocity.x': 'imu.w_x',
                           '/hamster2/imu/angular_velocity.y': 'imu.w_y',
                           '/hamster2/imu/angular_velocity.z': 'imu.w_z',
                           '/hamster2/imu/linear_acceleration.x': 'imu.a_x',
                           '/hamster2/imu/linear_acceleration.y': 'imu.a_y',
                           '/hamster2/imu/linear_acceleration.z': 'imu.a_z',
                           }, inplace=True)
except:
    print("'data.hamster2__imu.csv' doesn't exist!!!!!")

try:
    df_velocity = pd.read_csv('./data/' + data_subfold + '/data.hamster2__velocity.csv')
    df_velocity = df_velocity.iloc[:, [0, 3]]
    print("'data.hamster2__velocity.csv' loaded to 'df_velocity'")
    # rename the data
    df_velocity.rename(columns={'/hamster2/velocity/data': 'velocity.v'}, inplace=True)
except:
    print("'data.hamster2__velocity.csv' doesn't exist!!!!!")

try:
    df_tf = pd.read_csv('./data/' + data_subfold + '/data.tf.csv')
    df_tf = df_tf.iloc[:, [0, 3, 6, 7, 8, 9, 10, 11, 12]]
    print("'data.tf.csv' loaded to 'df_tf'")
    # rename the data
    df_tf.rename(columns={'/tf/transforms.0.header.stamp': 'tf.header.stamp',
                          '/tf/transforms.0.transform.translation.x': 'tf.x',
                          '/tf/transforms.0.transform.translation.y': 'tf.y',
                          '/tf/transforms.0.transform.translation.z': 'tf.z',
                          '/tf/transforms.0.transform.rotation.x': 'tf.r_x',
                          '/tf/transforms.0.transform.rotation.y': 'tf.r_y',
                          '/tf/transforms.0.transform.rotation.z': 'tf.r_z',
                          '/tf/transforms.0.transform.rotation.w': 'tf.r_w'
                          }, inplace=True)
except:
    print("'data.tf.csv' doesn't exist!!!!!")


# transform quaternion rotation to euler angle, and add it to the data
xyzw = df_tf.iloc[:, 5:9].to_numpy()            # extract quaternion rotation data: x, y, z, w, shape:(n,3)
euler = np.zeros((xyzw.shape[0], 3))            # initialize euler angles, shape:(n,3)

for i in range(xyzw.shape[0]):                  # loop n times to calculate euler angles and store them in 'df_euler'
    euler[i][0], euler[i][1], euler[i][2] = euler_from_quaternion(xyzw[i][0], xyzw[i][1], xyzw[i][2], xyzw[i][3])
df_euler = pd.DataFrame(euler, columns=['tf.roll_x', 'tf.pitch_y', 'tf.yaw_z'])

df_tf = pd.concat([df_tf, df_euler], axis=1)    # concatenate 'df_tf' and 'df_euler'

# merge 'df_command', 'df_velocity', 'df_imu' and 'df_tf'
list_df_origin = [df_command, df_imu, df_velocity, df_tf]

try:
    if not merge_command:
        list_df_origin = [df_imu, df_velocity, df_tf]
except:
    pass

list_df = [item_df for item_df in list_df_origin if item_df is not None]  # filter the empty data, (especially command)

df_merge = list_df[0]                           # initialize 'df_merge' with the first item of 'list_df'
for i in range(1, len(list_df)):                # loop 'list_df' to merge all df in it
    df_merge = df_merge.merge(list_df[i], how='outer', on='__time')

df_merge.sort_values(by=['__time'], inplace=True)
df_merge.reset_index(drop=True, inplace=True)

if save_merged_data is True:
    os.makedirs(r'./data/' + data_subfold, exist_ok=True)
    df_merge.set_index('__time').to_csv(r'./data/' + data_subfold + '/data_merge.csv')
    print("./data/" + data_subfold + "/data_merge.csv is saved\n")

print("\nDefined Variables:\n"
      "    df_command\n"
      "    df_imu\n"
      "    df_velocity\n"
      "    df_tf\n"
      "    df_merge    # merged from 'df_command', 'df_velocity', 'df_imu' and 'df_tf'\n")

print("Finish!")
