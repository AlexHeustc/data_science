import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

from scipy import stats

def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    
    return mu, sigma

# f1 score 最高的就是threshold
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    
    step = (pval.max() - pval.min()) / 1000
    
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    
    return best_epsilon, best_f1

if __name__ == '__main__':
    data = loadmat('data/ex8data1.mat')
    X = data['X']

    mu, sigma = estimate_gaussian(X)

    Xval = data['Xval']
    yval = data['yval']

    p = np.zeros((X.shape[0], X.shape[1]))
    p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
    p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

    pval = np.zeros((Xval.shape[0], Xval.shape[1]))
    pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
    pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

    epsilon, f1 = select_threshold(pval, yval)

    outliers = np.where(p < epsilon)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(X[:,0], X[:,1])
    ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
    plt.show()



