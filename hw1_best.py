import numpy as np
import pandas as pd
import sys
import csv

def load_data(path):
    name = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    chosen = ['CH4', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR']
    raw_test_data = pd.read_csv('data/test.csv', header=None, encoding='big5')
    raw_test_data = raw_test_data.replace(({'NR': 0})).to_numpy()[:, 2:].astype('float64')
        
    test_X = []
    for i in range(0, raw_test_data.shape[0], 18):
        test_X.append(raw_test_data[i:i + 18, :])
    test_X = np.hstack(test_X)
    df = pd.DataFrame(test_X.transpose(), columns=name)
    my = df[chosen].to_numpy().transpose()

    Xtest = []
    for i in range(0, my.shape[1], 9):
        Xtest.append(my[:, i:i + 9].flatten())
    Xtest = np.array(Xtest)
    Xtest = np.concatenate([np.ones([Xtest.shape[0], 1]), Xtest], axis=1)
    return Xtest

def to_csv(y, path):
    with open(path, mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        csv_writer.writerow(header)
        for i in range(y.shape[0]):
            row = ['id_' + str(i), y[i][0]]
            csv_writer.writerow(row)

if __name__ == '__main__':
    infile, outfile = sys.argv[1], sys.argv[2]
    X = load_data(infile)
    W = np.load('hw1_best/W.npy')
    to_csv(np.maximum(0, np.dot(X, W)), outfile)
