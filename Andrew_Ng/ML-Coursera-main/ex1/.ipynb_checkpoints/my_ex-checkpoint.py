import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def computeCost(X,y,theta):
    inner = np.array(X * theta.T - y)
    return np.sum(np.power(inner,2)) / (2*len(X))


#批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    #初始化
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T)-y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost



if __name__ == '__main__':
    path = './ex1data1.txt'
    data = pd.read_csv(path,header=None,names=['Population','Profit'])
    data.head()

    #x0=1数据向量化
    data.insert(0,'Ones',1)
    print(data.head())
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    plt.show()

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]


    #矩阵化
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))

    alpha = 0.01 #步长
    iters = 1000 #步数

    g, cost = gradientDescent(X, y, theta, alpha, iters)
    res = computeCost(X, y, g)

    print('theta:',g)
    print('cost',theta)

# data = pd.DataFrame(data)
# print(data)






