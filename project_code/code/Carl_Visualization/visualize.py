# this is a script that worked when tested with sample data below to plot and get max index
from matplotlib import markers
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib import cm

def uuid_lookup(uuid):
    manufacturers = []
    fp = '/Users/carlguo/Documents/gradclasses/22_2_SP/CS598CG/finalproject/BlueVision/bleuuids/UUIDs.txt'
    f = open(fp)
    lines = f.read().splitlines()

    for line in lines:
        if line.split(':')[1] == uuid:
            device_manu = line.split(":")[0]
            manufacturers.append(device_manu)

    return manufacturers

'''
Given some coordinates and RSSI of device packets at those coordinates, 
we use interpolation to predict the location of a device. 
- xs contains the x coordinates of where packet was detected  
- ys contains the y coordinates of where packet was detected  
- rssis is a list of signal strength values of the packets corresponding to coords.
'''
def localize_device(xs_device, ys_device, rssis, pos_xs, pos_ys):
    # TODO: change these
    minX = 0
    maxX = 240
    minY = 0
    maxY = 40

    xsteps = np.arange(minX, maxX, 0.1) # lets put as decimeters
    ysteps = np.arange(minY, maxY, 0.1)  # pretty sure this should be y

    xgrid, ygrid = np.meshgrid(xsteps, ysteps, indexing='xy')
    zgrid = griddata((xs_device, ys_device), rssis, (xgrid, ygrid), method='cubic')

    # clean out nan data
    if ~np.count_nonzero(zgrid[~np.isnan(zgrid)]):
        zgrid[np.isnan(zgrid)] = np.nanmin(zgrid[:])
        print(np.argmax(zgrid))
        print('x:', xsteps.shape)
        print('y:', ysteps.shape)
        print('z:', zgrid.shape)
        peak2d = np.unravel_index(zgrid.argmax(), zgrid.shape)
    x_peak = round(peak2d[1] * 0.1, 1)
    y_peak = round(peak2d[0] * 0.1, 1)
    rssi_peak = np.amax(rssis)
    print('peak:', '(', x_peak, ',', y_peak, ')', rssi_peak)
    print(np.stack((xs_device, ys_device), axis=1))
    print(rssis)

    manufacturers = uuid_lookup(test_uuid)

    # plot mesh grid
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.set_xlabel("X Position (feet)")
    ax1.set_ylabel("Y Posiiton (feet)")
    ax1.set_zlabel("RSSI Value (dBm)")
    ax1.set_title('BLE Device Localization')
    ax1.text(-60, -25, -15, s='UUID lookups:\n' + ',\n'.join(manufacturers), fontsize=6)

    surf = ax1.plot_surface(xgrid, ygrid, zgrid, rstride=100, cstride=100, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.scatter(xs_device, ys_device, rssis, marker='o', c='r')
    ax1.scatter([x_peak], [y_peak], [np.amax(zgrid)], marker='X', c='black')
    ax1.text(x_peak, y_peak, rssi_peak, '(' + str(x_peak) + ',' + str(y_peak) + ',' + str(rssi_peak) + ')')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    ax2.plot(pos_xs, pos_ys, '-')
    ax2.plot(xs_device, ys_device, '.')
    ax2.plot(0, 0, '.')
    ax2.plot(x_peak, y_peak, 'o')

    ax2.text(x_peak, y_peak, 'predicted device', fontsize=7, va='top')
    ax2.text(0, 0, 'initial point', fontsize=7, va='top')
    ax2.set_xlabel('x position (feet)')
    ax2.set_ylabel('y position (feet)')
    ax2.set_title('Path of route with localized device')

    plt.show()

# input is three filepaths: 
# one is for time: []
# other is for position: []
# last is for rssi: [time, uuid, rssi]
try:
    position_fp = sys.argv[1]
    rssi_fp = sys.argv[2]
except IndexError:
    print('Please add command line arguments position_file_path, rssi_file_path')
    exit()

TIME_THRESHOLD = 0.25#seconds
DIST_SCALE = 2

device_info = {} # {uuid: [[x,y,rssi]]} need to correlate timestamps to get x,y,rssi with uuid
# TODO: add some processing here where we merge the information in the two files together
rssi_data= pd.read_csv(filepath_or_buffer=rssi_fp, header=None, names=['timestamp', 'rssi', 'uuid'], skipinitialspace=True)
pos_data = pd.read_csv(filepath_or_buffer=position_fp, header=None, names=['timestamp', 'x', 'y'], skipinitialspace=True)

pos_xs = np.array(pos_data['x'], dtype=float) * DIST_SCALE
pos_ys = np.array(pos_data['y'], dtype=float) * DIST_SCALE


for rssi_ind in range(len(rssi_data)):
    for pos_ind in range(len(pos_data)):
        if np.abs(float(rssi_data['timestamp'][rssi_ind]) - float(pos_data['timestamp'][pos_ind])) < TIME_THRESHOLD:
            uuid = rssi_data['uuid'][rssi_ind]
            x = round(pos_data['x'][pos_ind], 1) * DIST_SCALE
            y = round(pos_data['y'][pos_ind], 1) * DIST_SCALE
            rssi = rssi_data['rssi'][rssi_ind]
            if uuid in device_info:
                device_info[uuid].append((x, y, rssi))
            else:
                device_info[uuid] = [(x, y, rssi)]
            break
# test
test_uuid = '0000feed-0000-1000-8000-00805f9b34fb'
test_res = np.array(device_info[test_uuid])


xs_device = test_res[:,0]
ys_device = test_res[:,1]
rssis = test_res[:,2]
localize_device(xs_device, ys_device, rssis, pos_xs, pos_ys)