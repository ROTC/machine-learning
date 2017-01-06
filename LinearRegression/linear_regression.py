import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class LinearRegressor:

    def __init__(self):
        self.coef_ = 0

    def fit(self, X, Y):
        X = np.matrix(X)
        Y = np.matrix(Y)
        if Y.shape[1] != 1:
            Y = Y.T
        if X.shape[0] != Y.shape[0] or Y.shape[1] != 1:
            raise Exception('Input Error')
        self.coef_= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, x):
        x = np.matrix(x)
        y = self.coef_.T.dot(x)
        return y

if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_tes = diabetes_X[-20:]
    diabetes_Y_train = diabetes.target[:-20]
    diabetes_Y_test = diabetes.target[-20:]

    regr = LinearRegressor()
    regr.fit(diabetes_X_train, diabetes_Y_train)
    print('Coefficients: \n', regr.coef_)
