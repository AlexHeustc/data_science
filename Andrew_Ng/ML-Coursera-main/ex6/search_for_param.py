from sklearn import svm
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio


if __name__ == '__main__':
    mat = sio.loadmat('./data/ex6data3.mat')
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    # len: 81
    combination = [(C, gamma) for C in candidate for gamma in candidate]

    search = []

    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(training[['X1', 'X2']], training['y'])
        search.append(svc.score(cv[['X1', 'X2']], cv['y']))

    best_score = search[np.argmax(search)]
    best_param = combination[np.argmax(search)]

    best_svc = svm.SVC(C=100, gamma=0.3)
    best_svc.fit(training[['X1', 'X2']], training['y'])
    ypred = best_svc.predict(cv[['X1', 'X2']])

    print(metrics.classification_report(cv['y'], ypred))
