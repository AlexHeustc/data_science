{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7562460",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid character in identifier (<ipython-input-1-285da064b93c>, line 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-1-285da064b93c>\"\u001b[1;36m, line \u001b[1;32m2\u001b[0m\n\u001b[1;33m    clf = tree.DecisionTreeClassifier()     #实例化\u001b[0m\n\u001b[1;37m                                        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid character in identifier\n"
     ]
    }
   ],
   "source": [
    "from sklearn import tree #导入需要的模块\n",
    "clf = tree.DecisionTreeClassifier()     #实例化\n",
    "clf = clf.fit(X_train,y_train) #用训练集数据训练模型\n",
    "result = clf.score(X_test,y_test) #导入测试集，从接口中调用需要的信息"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "411bb6de",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import tree\n",
    "from sklearn.datasets import load_wine\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "53c6e8ef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['class_0', 'class_1', 'class_2'], dtype='<U7')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine = load_wine()\n",
    "wine.data.shape\n",
    "wine.target\n",
    "import pandas as pd\n",
    "pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)\n",
    "wine.feature_names\n",
    "wine.target_names\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9213bdb8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
