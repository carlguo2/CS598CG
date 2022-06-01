import pandas as pd
import numpy as np
import scipy

def main():
    df = pd.read_csv('data.csv', names=['time', 'accel_x', 'accel_y', 'accel_z'])
    t = (df['time']/1000).to_numpy()
    a = df['accel_x'].to_numpy()
    a -= np.average(a[:50])
    v = [np.trapz(a[0:i], t[0:i]) for i in range(len(t))]
    d = [np.trapz(v[0:i], t[0:i]) for i in range(len(t))]
    print(f't:{t}')
    print(f'a:{a}\n')
    print(f'v:{v}')
    print(f'displacement is: {d[-1]}')
    with open('v.csv', 'w') as f:
        f.write('\n'.join(list(map(str, v))))
    with open('d.csv', 'w') as f:
        f.write('\n'.join(list(map(str, d))))

if __name__ == '__main__':
    main()