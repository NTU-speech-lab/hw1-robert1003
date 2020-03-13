import numpy as np
import pandas as pd
import sys
import csv

def load_data(path, avg, std):
    raw_test_data = pd.read_csv(path, header=None, encoding='big5')
    raw_test_data = raw_test_data.replace(({'NR': 0})).to_numpy()[:, 2:].astype('float64')

    test_X = []
    for i in range(0, raw_test_data.shape[0], 18):
        test_X.append(raw_test_data[i:i + 18, :])
    test_X = np.array(test_X)
    test_X = test_X.reshape(test_X.shape[0], -1)
    test_X = (test_X - avg) / std
    test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)

    return test_X
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
    avg, std = np.load('hw1_regression/avg.npy'), np.load('hw1_regression/std.npy')
    X = load_data(infile, avg, std)
    W = np.load('hw1_regression/W.npy')
    to_csv(np.maximum(0, np.dot(X, W)), outfile)
