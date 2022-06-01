
import scipy.integrate
import numpy as np

lins = []
with open('./data.csv','r') as the_in:
    lins = the_in.readlines()

sensor_readings = []
for line in lins:
    if line != '':
        sensor_read_line = line.strip('\n').strip(',').split(',')
        sensor_readings += [sensor_read_line]

print(sensor_readings)

time_nums = []
acc_x_list = []
acc_y_list = []
acc_z_list = []

for p in range(len(sensor_readings)):
    time_val = int(sensor_readings[p][0]) / 1000  # convert to seconds
    numx = float(sensor_readings[p][1])
    numy = float(sensor_readings[p][2])
    numz = float(sensor_readings[p][3])
    time_nums.append(time_val)
    acc_x_list.append(numx)
    acc_y_list.append(numy)
    acc_z_list.append(numz)

acc_x_list = np.array(acc_x_list)
acc_y_list = np.array(acc_y_list)
acc_z_list = np.array(acc_z_list)


# calibrate
avg_x = np.sum(acc_x_list[:10]) / 10
avg_y = np.sum(acc_y_list[:10]) / 10
avg_z = np.sum(acc_z_list[:10]) / 10

acc_x_list -= avg_x
acc_y_list -= avg_y
acc_z_list -= avg_z

# integrate

vel_x_list = scipy.integrate.cumulative_trapezoid(y=acc_x_list, x=time_nums)
dist_x_list = scipy.integrate.cumulative_trapezoid(y=vel_x_list,x=time_nums[:-1])
dist_x = scipy.integrate.trapezoid(y=vel_x_list,x=time_nums[:-1])
print('acc x list:', acc_x_list[-10:])
print("vel x list:", vel_x_list[-10:])
print('dist x list:', dist_x_list[-10:])
print('dist x val:', dist_x)

print('------')

vel_y_list = scipy.integrate.cumulative_trapezoid(y=acc_y_list, x=time_nums)
dist_y_list = scipy.integrate.cumulative_trapezoid(y=vel_y_list,x=time_nums[:-1])
dist_y = scipy.integrate.trapezoid(y=vel_y_list,x=time_nums[:-1])
print('acc_y list:', acc_y_list[-10:])
print("vel y list':", vel_y_list[-10:])
print('dist y list:', dist_y_list[-10:])
print('dist y val:', dist_y)

# vel_z_list = scipy.integrate.cumulative_trapezoid(y=acc_z_list, x=time_nums)
# dist_z_list = scipy.integrate.cumulative_trapezoid(y=vel_z_list,x=time_nums[:-1])
# dist_z = scipy.integrate.trapezoid(y=vel_z_list,x=time_nums[:-1])
# print('acc_z list:', acc_z_list[-10:])
# print("vel z list':", vel_z_list[-10:])
# print('dist z list:', dist_z_list[-10:])
# print('dist z val:', dist_z)
