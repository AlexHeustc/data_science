import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:, 0]
y = data[:, 1]
m = y.size

#数据可视化
plt.figure(0)
fg1 = plt.scatter(x, y, marker='o', c='b', label='Training data')

'''正规方程法'''
a = np.ones((m, 1)) #将x矩阵第一列赋值为1
x = np.column_stack((a, x)) #矩阵合并
theta = np.linalg.inv(x.T@x)@x.T@y #求矩阵的逆

#增加拟合出的曲线
a = np.linspace(5.0, 22.5, num=5) #等差数列设点，其实可以直接用x
i = 0
b = np.ones(5)
while i<5:
    b[i] = np.array([1,a[i]])@theta.T
    i = i+1
fg2 = plt.plot(a, b, c='r', label='Prediction')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')
plt.legend(loc=2)
plt.show()

#预测值
x_predict = float(input('请输入预测人口：'))
predict1 = np.array([1,x_predict])@theta.T
print(predict1)

