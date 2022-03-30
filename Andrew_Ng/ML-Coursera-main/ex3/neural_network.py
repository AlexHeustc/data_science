import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


if __name__ == '__main__':
    #theta1.shape: 25, 401 theta2.shape: 10,26
    theta1, theta2 = load_weight('ex3weights.mat')
    # X.shape: 5000, 401 y.shape: 5000,1
    X, y = load_data('ex3data1.mat',transpose=False)
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
    a1 = X
    z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
    z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
    a2 = sigmoid(z2)
    # z3.shape: 5000, 10
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
    print(classification_report(y, y_pred))








