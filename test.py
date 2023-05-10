import numpy as np
from load_write_data import load_data
from Kalman_filter import kalman_filter
import interpolation
import matplotlib.pyplot as plt

time_imu, angular_velocity_x, angular_velocity_y, angular_velocity_z, linear_accel_x, linear_accel_y, linear_accel_z = load_data()
angular_velocity_x = interpolation.previous_interpolation(angular_velocity_x)
angular_velocity_y = interpolation.previous_interpolation(angular_velocity_y)
angular_velocity_z = interpolation.previous_interpolation(angular_velocity_z)
angular_velocity_x_filtered = kalman_filter(angular_velocity_x, 1e-8, 1e-5, 0, 1)
angular_velocity_y_filtered = kalman_filter(angular_velocity_y, 1e-8, 1e-5, 0, 1)
angular_velocity_z_filtered = kalman_filter(angular_velocity_z, 1e-8, 1e-5, 0, 1)

plt.plot(time_imu, angular_velocity_z_filtered)
plt.show()

test = np.sin(30)
print(test)