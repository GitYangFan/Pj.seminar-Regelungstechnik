import os
import casadi as ca
from MheSingleShooting import MheSingleShooting
from MheMultipleShooting import MheMultipleShooting
from MHERealTime import MHERealTime
from KFRealTime import KFRealTime

# # Load data files ====================================================
# select data
# data_subfold = "data_2023_03_01-16_48_29"  # the data from Lukas
data_subfold = "rosbag2_2023_07_24-14_59_34"    # new data
merge_command = False
save_merged_data = False
# execfile("data_process_from_rosbags.py")
exec(open("data_process_from_rosbags.py").read())
# ====================================================================

# # Correct imu data using a rotation matrix =====================================================
o_R_e = np.genfromtxt("o_R_e.csv", delimiter=",")
df_imu[["imu.a_x", "imu.a_y", "imu.a_z"]] = df_imu[["imu.a_x", "imu.a_y", "imu.a_z"]] @ o_R_e.T
df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] = df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] @ o_R_e.T
# df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] = df_imu[["imu.w_x", "imu.w_y", "imu.w_z"]] * 1.1
# ================================================================================================

Dimu = df_imu[['__time', 'imu.header.stamp', 'imu.a_z', 'imu.w_y']].rename(columns={'imu.header.stamp': 'time'})
Dtf = df_tf[['__time', 'tf.header.stamp', 'tf.x', 'tf.y', 'tf.yaw_z']].rename(columns={'tf.header.stamp': 'time'})
if 'vel.header.stamp' in df_velocity.columns.to_list():
    Dvelocity = df_velocity.rename(columns={'vel.header.stamp': 'time'})
else:
    Dvelocity = df_velocity.assign(time=df_velocity['__time'])  # vel has no header.stamp

# # Only for old data ===========================================================================================
# compensate time difference between header.stamp and __time of TF in the first data point
# d_time = df_tf.iloc[0, 1] - df_tf.iloc[0, 0]
# Dtf["time"] = Dtf["time"] - d_time
#
# Dtf["__time"] = Dtf["__time"] - 0.8                         # compensate transport delay in __time of TF
#
# Dvelocity['velocity.v'] = -Dvelocity['velocity.v']/1000     # change the unit of velocity data, only for old data
# ===============================================================================================================

# merge imu vel tf raw data
df_data = Dimu.merge(Dvelocity, how='outer', on=['__time', 'time'])
df_data = df_data.merge(Dtf, how='outer', on=['__time', 'time'])


# # Real Time Simulation by=['__time'], else by=['time'] ============================================
df_data.sort_values(by=['__time'], inplace=True)        # sort by global time:'__time', else 'time'
# ===================================================================================================
df_data.reset_index(inplace=True, drop=True)
data = df_data.to_numpy()

# switch
s_imu = np.isnan(data[:, 2])
s_vel = np.isnan(data[:, 4])
s_tf = np.isnan(data[:, 5])

# # Fine Tune the Simulation Parameters
# for mhe
# horizon = 20
# Q = np.diag([10, 10, 10, 10])     # covariance matrix for state noise
# R_vel = np.array([[0.1]])           # covariance matrix for velocity measurement noise
# R_tf = np.diag([0.01, 0.01, 0.01])    # covariance matrix for TF measurement noise
# P = np.diag([10, 10, 10, 10])       # covariance matrix for arrival cost
# # for kf
Q = np.diag([10, 10, 10, 10])         # process noise
R_vel = np.array([[0.1]])        # covariance matrix for velocity measurement noise
R_tf = np.diag([0.1, 0.1, 0.1])    # covariance matrix for TF measurement noise
P = np.diag([10, 10, 10, 10])         # initial covariance matrix for current state

# # Choose the Algorithm
# sim = MheSingleShooting(horizon, Q, R_vel, R_tf, P)
# sim = MheMultipleShooting(horizon, Q, R_vel, R_tf, P)
# sim = MHERealTime(horizon, Q, R_vel, R_tf, P)
sim = KFRealTime(Q, R_vel, R_tf, P)

# # Set the file name in which the simulation result is saved
# file_name = 'sim_mhe_multiple_correctedimu'
# file_name = 'sim_mhe_multiple_cos'
# file_name = 'sim_mhe_multiple_tan'
# file_name = 'sim_mhe_multiple_sin'
# file_name = 'sim_mhe_multiple_mod'
# file_name = 'MHE'
# file_name = 'MHE_nodelay'
file_name = 'KF'
# file_name = 'KF_nodelay'


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
    # Record simulation status in halfway
    if i == 4600:
        sim.df.to_csv('./sim_result/' + file_name + '_halfway.csv')
        print("saved halfway")

sim.df.to_csv('./sim_result/' + file_name + '.csv')

