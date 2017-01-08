# -*- coding: utf-8 -*-
"""
程序说明：
fit函数
基于X和y作线性回归拟合，求出系数coef_，shape为(1, _)，X为矩阵，y为向量

predict函数
基于coef_对X进行预测，预测值的shape为(1, _)，X为矩阵

程序验证部分参考sklearn中的Linear Regression Example
"""

import numpy as np
from sklearn import datasets

class LinearRegressor:

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.array(X).reshape((len(X), -1))
        X = np.c_[np.ones(len(X)), X]   #add a constant column
        y = np.array(y).reshape((len(y), -1))
        if X.shape[0] != y.shape[0] or y.shape[1] != 1:
            raise Exception('Input Error')
        self.cofe_= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)[:, 0]

    def predict(self, X):
        X = np.c_[np.ones(len(X)), np.array(X)]
        y = self.coef_.dot(X.T)
        return y

if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    regr = LinearRegressor()
    regr.fit(diabetes_X_train, diabetes_y_train)

    #The coefficients
    print('Coefficients: \n', regr.coef_)
    #The mean squared error
    print('Mean squared error: %.2f' % np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
