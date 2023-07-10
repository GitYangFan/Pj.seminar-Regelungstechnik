import os
import casadi as ca
from MheSingleShooting import MheSingleShooting
from MheMultipleShooting import MheMultipleShooting
from KFRealTime import KFRealTime

# load data files
# PLEASE MODIFY THE PATH OF DATA!!!!!!!!!!!!!!!!!!!!!!!!!
# ===========================================================================================
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# # select data
data_subfold = "data_2023_03_01-16_48_29"  # the data from Lukas

# merge_command = False
save_merged_data = False
# execfile("data_process_from_rosbags.py")
exec(open("data_process_from_rosbags.py").read())
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# ===========================================================================================

# correct imu data using a rotation matrix
# o_R_e = np.genfromtxt("o_R_e.csv", delimiter=",")
# df_imu[["imu.a_x", "imu.a_y", "imu.a_z"]] = df_imu[["imu.a_x", "imu.a_y", "imu.a_z"]] @ o_R_e.T
# df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] = df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] @ o_R_e.T
# df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] = df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] * 1.1


Dimu = df_imu.iloc[:, [0, 1, -1, 3]].rename(columns={'imu.header.stamp': 'time'})
Dvelocity = df_velocity.assign(time=df_velocity['__time'])
Dtf = df_tf.iloc[:, [0, 1, 2, 3, -1]].rename(columns={'tf.header.stamp': 'time'})

# time difference between header.stamp and __time of TF in the first data point
d_time = df_tf.iloc[0, 1] - df_tf.iloc[0, 0]
Dtf["time"] = Dtf["time"] - d_time

Dvelocity['velocity.v'] = -Dvelocity['velocity.v']/1000 # change the unit of velocity data, only for old data

df_data = Dimu.merge(Dvelocity, how='outer', on=['__time', 'time'])
df_data = df_data.merge(Dtf, how='outer', on=['__time', 'time'])
df_data.sort_values(by=['__time'], inplace=True)        # sort by global time:'__time', else 'time'
df_data.reset_index(inplace=True, drop=True)
data = df_data.to_numpy()

s_imu = np.isnan(data[:, 2])
s_vel = np.isnan(data[:, 4])
s_tf = np.isnan(data[:, 5])

Q = np.diag([1., 1., 1., 1.])   # weight matrix for state noise
R_vel = np.array([[1.]])        # weight matrix for velocity measurement noise
R_tf = np.diag([0.1, 0.1, 0.1])    # weight matrix for TF measurement noise
P = np.diag([1., 1., 1., 1.])   # weight matrix for arrival cost

horizon = 20

# sim = MheSingleShooting(horizon, Q, R_vel, R_tf, P)
# sim = MheMultipleShooting(horizon, Q, R_vel, R_tf, P)
sim = KFRealTime(Q, R_vel, R_tf, P)

sim.initialization()
for i in range(len(data)):
    if not s_imu[i]:
        data_typ = 'imu'
        t_stamp = data[i, 1]    # header time stamp
        value = data[i, 2:4].reshape(2, 1)
    elif not s_vel[i]:
        data_typ = 'vel'
        t_stamp = data[i, 1]
        value = data[i, 4].reshape(1, 1)
    elif not s_tf[i]:
        data_typ = 'tf'
        t_stamp = data[i, 1]
        value = data[i, 5:8].reshape(3, 1)
    print("Iteration:", i+1, '/', len(data))
    sim(t_stamp, data_typ, value)
    # if i == 4600:
    #     sim.df.to_csv(r'sim_KF_halfway.csv')
    #     print("saved")

# file_name = 'sim_mhe_multiple_correctedimu'
# file_name = 'sim_mhe_multiple_cos'
# file_name = 'sim_mhe_multiple_tan'
# file_name = 'sim_mhe_multiple_sin'
# file_name = 'sim_mhe_multiple_mod'
file_name = 'sim_KF'

sim.df.to_csv(file_name + r'.csv')

