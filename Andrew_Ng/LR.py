import numpy as np
from sympy import symbols,diff

# def f(a,b):
#     return a**2 + b**3
#
# x,y = symbols('x y',read=True)
#
# print(diff(f(x,y),x,2))
import pandas as pd

# def Costfun(X,y,theta):
#     inner = np.power((X * theta.T) - y,2)
#     return np.sum(inner) / 2*len(X)

X=[[1,0],[0,1]]
# y=[[1,0],[0,1]]
# theta = [[1,0],[0,1]]
# print(Costfun(X,y,theta))
# X=np.array(X)
# print(X.T)
# def normalEqn(X,y):
#     theta = np.linalg.inv(X.T @ X)@ X.T @ y
#     return theta

def cost(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply(1-y,np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X)
