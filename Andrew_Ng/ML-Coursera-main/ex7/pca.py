import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

if __name__ == '__main__':
    data = loadmat('data/ex7data1.mat')
    X = data['X']

    U, S, V = pca(X)

    Z = project_data(X, U, 1)

    X_recovered = recover_data(Z, U, 1)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
    plt.show()





