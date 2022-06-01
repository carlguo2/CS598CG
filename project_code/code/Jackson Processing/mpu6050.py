'''
        Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
	http://www.electronicwings.com
'''
import smbus			#import SMBus module of I2C
from time import sleep          #import
import sys
import time


class MPU6050:
    #some MPU6050 Registers and their Address
    PWR_MGMT_1   = 0x6B
    SMPLRT_DIV   = 0x19
    CONFIG       = 0x1A
    GYRO_CONFIG  = 0x1B
    INT_ENABLE   = 0x38
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    GYRO_XOUT_H  = 0x43
    GYRO_YOUT_H  = 0x45
    GYRO_ZOUT_H  = 0x47
    def __init__(self) -> None:
        self.device_address = 0x68   # MPU6050 device address
        self.bus = smbus.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
        self.PRINT_RESULT = True
        #write to sample rate register
        self.bus.write_byte_data(self.device_address, MPU6050.SMPLRT_DIV, 7)
        
        #Write to power management register
        self.bus.write_byte_data(self.device_address, MPU6050.PWR_MGMT_1, 1)
        
        #Write to Configuration register
        self.bus.write_byte_data(self.device_address, MPU6050.CONFIG, 0)
        
        #Write to Gyro configuration register
        self.bus.write_byte_data(self.device_address, MPU6050.GYRO_CONFIG, 24)
        
        #Write to interrupt enable register
        self.bus.write_byte_data(self.device_address, MPU6050.INT_ENABLE, 1)
        

    def read_raw_data(self, addr):
        #Accelero and Gyro value are 16-bit
            high = self.bus.read_byte_data(self.device_address, addr)
            low = self.bus.read_byte_data(self.device_address, addr+1)
        
            #concatenate higher and lower value
            value = ((high << 8) | low)
            
            #to get signed value from mpu6050cg
            if(value > 32768):
                    value = value - 65536
            return value

    def read_output(self):
        #Read Accelerometer raw value
        acc_x = self.read_raw_data(self.ACCEL_XOUT_H)
        acc_y = self.read_raw_data(self.ACCEL_YOUT_H)
        acc_z = self.read_raw_data(self.ACCEL_ZOUT_H)
        
        #Read Gyroscope raw value
        gyro_x = self.read_raw_data(self.GYRO_XOUT_H)

        gyro_y = self.read_raw_data(self.GYRO_YOUT_H)
        gyro_z = self.read_raw_data(self.GYRO_ZOUT_H)
        #Full scale range +/- 250 degree/C as per sensitivity scale factor
        Ax = acc_x/16384.0
        Ay = acc_y/16384.0
        Az = acc_z/16384.0
        
        Gx = gyro_x/131.0
        Gy = gyro_y/131.0
        Gz = gyro_z/131.0
        #if self.PRINT_RESULT:
        #    print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)
        return (Gx, Gy, Gz,Ax, Ay, Az)

    def read_output2(self):
                #Read Accelerometer raw value
        acc_x = self.read_raw_data(self.ACCEL_XOUT_H)
        acc_y = self.read_raw_data(self.ACCEL_YOUT_H)
        acc_z = self.read_raw_data(self.ACCEL_ZOUT_H)
        
        #Read Gyroscope raw value
        gyro_x = self.read_raw_data(self.GYRO_XOUT_H)

        gyro_y = self.read_raw_data(self.GYRO_YOUT_H)
        gyro_z = self.read_raw_data(self.GYRO_ZOUT_H)

        return [gyro_x,gyro_y, gyro_z, acc_x, acc_y, acc_z]

    def get_smoothed_values(self, n_samples=10):
        result = [0,0,0,0,0,0]
        for _ in range(n_samples):
            data = self.read_output2()
            
            for k in range(6):
                result[k] = result[k] + data[k]/n_samples
        return result

    def calibrate(self, threshold=250, n_samples=100):
        while True:
            v1=[]
            v2=[]
            for _ in range(n_samples):
                v1.append(self.read_output2())
                v2.append(self.read_output2())

            #print(v1[0],v2[0])
            if all(abs(v1[x][k]-v2[x][k]) < threshold for k in range(6) for x in range(n_samples)):
                return v1
def find_offsets(f):
    with open(f,'r') as the_in:
        lins = the_in.readlines()
    lins = lins[1:]
    meas = [[],[],[],[],[],[]]
    for lin in lins:
        split_lin = lin.split(',')
        for i in range(0,len(split_lin)):
            meas[i] += [float(split_lin[i].strip().strip('\n'))]
    ranges = []

    for i in meas:
        ranges.append(tuple((min(min(i)/1.5,min(i)*1.5),max(max(i)*1.5,max(i)/1.5))))
    # averages = []
    # for k in meas:
    #     averages.append(k/len(lins))
    # return averages
    return ranges 

if __name__ == '__main__':
    mpu = MPU6050()
    l=''
    with open('cs598cg/code/Jackson Processing/nomu.txt','w') as the_in:
        for _ in range(1000):
            l += str(mpu.read_output()).strip('(').strip(')') + '\n'
        the_in.write(l)
    ranges = find_offsets('cs598cg/code/Jackson Processing/nomu.txt')
    #We use ranges from max and min of 10000 test values without movement
    #print(ranges)
    #ranges = [(-0.3893129770992366, -0.2748091603053435), (-0.48091603053435117, -0.2824427480916031), (0.061068702290076333, 0.17557251908396945), (-0.03662109375, 0.02587890625), (-0.03125, -0.000244140625), (0.954833984375, 1.077880859375)]
    print("Now Callibrated: offsets", ranges)
    #Now we try movement
    ll = []
    try:
        while True:
            ll += [list(mpu.read_output()) + [time.time()]]
            sleep(0.001)
    except KeyboardInterrupt:
        #print(ll)
        with open('cs598cg/code/Jackson Processing/mo.txt','w') as the_out:
            l_write = ''
            for elem in ll:
                for i in range(len(elem)-1):
                    if elem[i] >= ranges[i][0] and elem[i]  <= ranges[i][1]:
                        l_write += '0,'#str(elem[i]-sum(ranges[i])/2)+','#'0,'
                    elif elem[i]  < ranges[i][0]:
                        l_write += str(elem[i]-ranges[i][0])+','
                    elif elem[i]  > ranges[i][1]:
                        l_write += str(elem[i]-ranges[i][1])+','
                l_write+= str(elem[-1]) + ',' + '\n' 
            the_out.write(l_write)