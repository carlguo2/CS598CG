import sys
import socket
import select
import time

TCP_IP = '10.193.9.118'
TCP_PORT = 2055
BUFFER_SIZE = 1024
param = []

print('Listening for client...')
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((TCP_IP,TCP_PORT))
server.listen(1)
rxset = [server]
txset = []
tcpdata = []

try:
    while 1:
            rxfds, txfds, exfds = select.select(rxset, txset, rxset)
            for sock in rxfds:
                if sock is server:
                    conn, addr = server.accept()
                    conn.setblocking(0)
                    rxset.append(conn)
                    print('Connection from address:', addr)
                else:
                    data = sock.recv(BUFFER_SIZE)
                    if data == ";" :
                        print("Received all the data")
                        for x in param:
                            print(x)
                        param = []
                        rxset.remove(sock)
                        sock.close()
                    else:
                        param.append(data)
                        tcpdata.append(data)
except KeyboardInterrupt:
    with open('./data.csv', 'w') as f:
        for imudata in tcpdata:
            f.write(imudata.decode())
    print("Connection closed by remote end")
    param = []
    rxset.remove(sock)
    sock.close()