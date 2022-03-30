from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

wine = load_wine()

wine.data.shape
wine.target

pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
wine.feature_names
wine.target_names

Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

