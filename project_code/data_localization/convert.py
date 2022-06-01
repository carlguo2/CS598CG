from scapy.all import *
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from scapy2dict import to_dict
from collections import ChainMap
from parse import *
import csv

f1 = sys.argv[1]

fname=f1+'.pcap'

kop =[]
def method_filter_HTTP(pkt):
    global kop
    try:
        srcmac = pkt.addr2
        if srcmac in vals:
            ctime = pkt.time
            rssi = pkt.dBm_AntSignal
            # print(srcmac,ctime,rssi)
            # if srcmac not in kop:
            #     kop[srcmac]=[]
            # else:
            tpr = [valid_macs[srcmac],float(ctime),float(rssi)]
            kop.append(tpr)

            # ()+1
    except Exception as e:
        print(e)

vals = list(valid_macs.keys())
print(vals)
sniff(offline=fname,prn=method_filter_HTTP,store=0)
# print(kop)

# for x1,v1 in kop.items():
#     print(x1,v1)
OG = f1+'.csv'
with open(OG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(kop)
# ()+1