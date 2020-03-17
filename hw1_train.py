import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split

def to_csv(y_test):
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), y_test[i]]
            csv_writer.writerow(row)

# load train & test data
raw_train_data = pd.read_csv('data/train.csv', encoding='big5')
raw_train_data = raw_train_data.replace({'NR': 0})
raw_train_data['Month'] = raw_train_data['日期'].apply(lambda x: int(x.split('/')[1]) - 1)

month_train_data = []
for i in range(12):
    temp = raw_train_data.groupby('Month').get_group(i).iloc[:, 3:-1].to_numpy().astype('float64')
    final = np.array([[] for _ in range(18)])
    for j in range(0, temp.shape[0], 18):
        final = np.concatenate((final, temp[j:j + 18, :]), axis=1)
    month_train_data.append(final)

tot_train_data = []
for i in range(18):
    tmp = np.array([])
    for j in range(12):
        tmp = np.concatenate((tmp, month_train_data[j][i]))
    tot_train_data.append(tmp)
tot_train_data = np.array(tot_train_data)

raw_test_data = pd.read_csv('data/test.csv', header=None, encoding='big5')
raw_test_data = raw_test_data.replace(({'NR': 0})).to_numpy()[:, 2:].astype('float64')

for i in range(0, raw_test_data.shape[0], 18):
    tot_train_data = np.concatenate((tot_train_data, raw_test_data[i:i + 18, :]), axis=1)

name = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
df = pd.DataFrame(tot_train_data.transpose(), columns=name)
df['PM2.5true'] = df['PM2.5']

def get_data(data, subset, num):
    pm25 = data['PM2.5true'].to_numpy()
    my = data[subset].to_numpy().transpose()
    
    X, y = [], []
    for i in range(9, 5760):
        if i % 480 < 9:
            continue
        X.append(my[:, i - 9:i].flatten())
        y.append(pm25[i])
    X, y = np.array(X), np.array(y)
    
    tX, ty, vX, vy = [], [], [], []
    for i in range(12):
        tX.append(X[471 * i:471 * (i + 1) - num])
        ty.append(y[471 * i:471 * (i + 1) - num])
        vX.append(X[471 * (i + 1) - num:471 * (i + 1)])
        vy.append(y[471 * (i + 1) - num:471 * (i + 1)])
    
    Xtest = []
    for i in range(5760, my.shape[1], 9):
        Xtest.append(my[:, i:i + 9].flatten())
    Xtest = np.array(Xtest)

    return np.vstack(tX), np.vstack(vX), np.hstack(ty).reshape(-1, 1), np.hstack(vy).reshape(-1, 1), Xtest # X_train, X_valid, y_train, y_valid, X_test

class SGD:
    def __init__(self):
        self.W = None
    def rmse(self, y_pred, y_true):
        return np.sqrt(np.sum(np.power(y_pred - y_true, 2)) / y_pred.shape[0])
    def fit(self, X, y, Xv=None, yv=None, epochs=10000, lr=0.01, opt='adam', print_every=1000, lamb=0, l1=0):
        X_train, y_train = X.copy(), y.copy()
        X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        bestv, bestt, ep = 10 ** 9, 10 ** 9, 0
        X_valid, y_valid = None, None
        if Xv is not None:
            X_valid, y_valid = Xv.copy(), yv.copy()
            X_valid = np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis=1)
        W = np.zeros((X_train.shape[1], 1))
        train_loss, valid_loss = [], None
        m, v, a, eps, b1, b2 = np.zeros((X_train.shape[1], 1)), np.zeros((X_train.shape[1], 1)), np.zeros((X_train.shape[1], 1)), 1e-9, 0.9, 0.999
        if Xv is not None:
            valid_loss = []
        for epoch in range(epochs):
            # calculate loss
            loss, vloss = self.rmse(np.dot(X_train, W), y_train), None
            train_loss.append(loss)
            if Xv is not None:
                vloss = self.rmse(np.dot(X_valid, W), y_valid)
                valid_loss.append(vloss)
                if vloss < bestv:
                    ep = epoch
                    bestv = vloss
                    bestt = loss
                    self.W = W
            # print_loss
            if epoch % print_every == 0:
                print(f'{epoch}: {loss} {vloss}')
            # calculate gradient
            grad = np.dot(X_train.transpose(), np.dot(X_train, W) - y_train) * 2 + lamb * np.sqrt(W ** 2) + l1 * np.sign(W)
            # update W by opt
            if opt == 'adam':
                m = m * b1 + grad * (1 - b1)
                v = v * b2 + grad ** 2 * (1 - b2)
                mhat = m / (1 - b1)
                vhat = v / (1 - b2)
                W -= lr * mhat / (np.sqrt(vhat) + eps)
            elif opt == 'adagrad':
                a += grad ** 2
                W -=  grad * lr / np.sqrt(a + eps)
            else:
                W -= grad * lr
        if Xv is None:
            self.W = W
        print(f'Best train loss: {bestt}, best valid loss: {bestv}, ep: {ep}')
        return train_loss, valid_loss
    def predict(self, X):
        X_test = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.dot(X_test, self.W)

chosen = ['CH4', 'CO', 'NO', 'NO2', 'O3', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR', 'PM10']

model = SGD()

X_train, X_valid, y_train, y_valid, X_test = get_data(df, chosen, 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=32)

model.fit(X_train, y_train, X_valid, y_valid, epochs=500000, lr=0.01, opt='adagrad', print_every=1000, lamb=0, l1=0)

to_csv(np.maximum(0, model.predict(X_test).flatten()))

np.save('hw1/W.npy', model.W)

