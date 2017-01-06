"""
����˵����
fit����
����X��Y�����Իع���ϣ����ϵ��coef_��shapeΪ(1, _)

predict����
����coef_����Ԥ�⣬Ԥ��ֵ��shapeΪ(1, _)
"""

import numpy as np
from sklearn import datasets

class LinearRegressor:

    def __init__(self):
        self.coef_ = None

    def fit(self, X, Y):
        X = np.array(X).reshape((len(X), -1))
        X = np.c_[np.ones(len(X)), X]   #add a constant column
        Y = np.array(Y).reshape((len(Y), -1))
        if Y.shape[1] != 1:
            Y = Y.T
        if X.shape[0] != Y.shape[0] or Y.shape[1] != 1:
            raise Exception('Input Error')
        self.coef_= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)[:, 0]

    def predict(self, x):
        x = np.c_[np.ones(len(x)), np.array(x)]
        y = self.coef_.dot(x.T)
        return y

if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_Y_train = diabetes.target[:-20]
    diabetes_Y_test = diabetes.target[-20:]

    regr = LinearRegressor()
    regr.fit(diabetes_X_train, diabetes_Y_train)

    #The coefficients
    print('Coefficients: \n', regr.coef_)
    #The mean squared error
    print('Mean squared error: %.2f' % np.mean((regr.predict(diabetes_X_test)-diabetes_Y_test)**2))
