from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio

def svm_model(X, y, test_X, test_y):
    svc = svm.SVC()
    svc.fit(X, y)
    pred = svc.predict(test_X)
    print(metrics.classification_report(test_y, pred))


def logisitic_reg(X, y, test_X, test_y):
    logit = LogisticRegression()
    logit.fit(X, y)

    pred = logit.predict(test_X)
    print(metrics.classification_report(test_y, pred))


if __name__ == '__main__':
    mat_tr = sio.loadmat('data/spamTrain.mat')
    X, y = mat_tr.get('X'), mat_tr.get('y').ravel()

    mat_test = sio.loadmat('data/spamTest.mat')
    test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()

    svm_model(X, y, test_X, test_y)
