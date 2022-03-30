import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mat = sio.loadmat('./data/ex6data1.mat')
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')

    svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
    svc100.fit(data[['X1', 'X2']], data['y'])

    data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
    ax.set_title('SVM (C=100) Decision Confidence')
    plt.show()
