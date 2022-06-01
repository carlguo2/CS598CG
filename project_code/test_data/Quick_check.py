import sys
#Finds all the data in the file an writes it to a csv
with open('output.txt') as ff:
    lines = ff.readlines()

all = ''
info = ''
prnt = False
for line in lines:
    if 'time' in line:
        info += line[6:-1] + ', '
    if 'RSSI' in line:
        info += line[line.find('=') +2:-1] + ', '
    if "UUIDs = dbus.Array" in line:
        if "UUIDs = dbus.Array([dbus.String(" in line:
            info += line[37:73]
            all += info + '\n'
            info = ''
        else:
            info = ''
    

with open('output.csv', 'w') as aa:
    aa.write(all)
