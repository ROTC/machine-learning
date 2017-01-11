import numpy as np
from sklearn import datasets

class LinearRegression(object):
    """Linear regression model.

    Attributes
    -----------
    coef_ : array, shape = (1, m),
            coefficients of linear regression problems

    """

    def fit(self, X, y):
        """Fit linear model.

        Parameters
        ----------
        X : list or array, shape = (m, n),
            Training data

        y : list or array, shape = (m, )
            Target values

        Returns
        ----------
        self : returns an instance of self.
        """

        X = np.array(X).reshape((len(X), -1))
        X = np.c_[np.ones(len(X)), X]   #add a constant column
        y = np.array(y).reshape((len(y), -1))
        if X.shape[0] != y.shape[0] or y.shape[1] != 1:
            raise Exception('Input Error')
        self.coef_= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)[:, 0]
        return self

    def predict(self, X):
        """Predict target values for samples in X.

        Parameters
        ----------
        X : list or array, shape = (1, n) or (m, n)

        returns
        ----------
        C : array, shape = (1, ) or (m, )
            Predicted target value per sample.
        """

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

    regr = LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)

    #The coefficients
    print('Coefficients: \n', regr.coef_)
    #The mean squared error
    print('Mean squared error: %.2f' % np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
