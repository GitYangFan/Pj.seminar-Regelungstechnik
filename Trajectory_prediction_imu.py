import numpy as np
import interpolation
from load_write_data import load_data
import matplotlib.pyplot as plt

# load the imu data from csv
imu_time, imu_angular_v_x, imu_angular_v_y, imu_angular_v_z, imu_linear_a_x, imu_linear_a_y, imu_linear_a_z = load_data()

# interpolation of the missing values
imu_angular_v_x = interpolation.previous_interpolation(imu_angular_v_x)
imu_angular_v_y = interpolation.previous_interpolation(imu_angular_v_y)
imu_angular_v_z = interpolation.previous_interpolation(imu_angular_v_z)
imu_linear_a_x = interpolation.previous_interpolation(imu_linear_a_x)
imu_linear_a_y = interpolation.previous_interpolation(imu_linear_a_y)
imu_linear_a_z = interpolation.previous_interpolation(imu_linear_a_z)

# initialization of position and velocity result of vehicle
sz = len(imu_time)
position_hamster = np.zeros((sz,2))
velocity_hamster = np.zeros((sz,2))

# initialization of intermediate variables(only consider horizon level)
dt = 0  # duration time step
position_hamster[0,:] = np.array([2.668912, -3.346767])   # initial position of Hamster in the world coordinate system
angular = 0  # the initial body gesture of Hamster in the world coordinate system
linear_a = 0  # average linear acceleration in dt
angular_v = 0  # average angular velocity in dt

for idx in range(1, sz):
    dt = imu_time[2]-imu_time[1]
    linear_a = 0.5 * (imu_linear_a_z[idx-1] + imu_linear_a_z[idx])  # average linear acceleration using median method
    angular_v = 0.5 * (imu_angular_v_y[idx-1] + imu_angular_v_y[idx])   # average angular velocity using median method
    # calculate the gesture after dt using integral
    angular = angular + angular_v * dt
    # calculate the displacement in the direction of new gesture after dt using integral
    displacement = 0.5 * linear_a * dt**2
    # calculate corresponding position after dt based on angular and displacement
    position_hamster[idx,:] = position_hamster[idx-1,:] + [displacement * np.cos(angular), displacement * np.sin(angular)]


# plotting
plt.plot(position_hamster[:,0],position_hamster[:,1])
plt.show()