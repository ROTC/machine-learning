import numpy as np
from sklearn import datasets, linear_model

class LogisticRegression(object):
    """Logistic regression model.

    Attributes
    -----------
    coef_ : array, shape = (m, )
            Estimated coefficients for the linear classification problem

    """

    def fit(self, X, y, solver = 'gd'):
        """Fit logistic model

        Parameters
        ----------
        X : list or array, shape = (n_samples, n_features + 1),
            Training data, adding a constant column(ones)

        y : list or array, shape = (n_samples, ),
            Sample labels

        solver : {'gd', 'sgd'}
            Algorithm to use in the optimization problem

        Returns
        ----------
        self : returns an instance of self.
        """

        n_samples = len(X)
        X = np.array(X).reshape((n_samples, -1))
        n_features = X.shape[1]
        y = np.array(y).reshape((n_samples, -1))

        w_old = np.ones(n_features)#/1000			# 系数初始值
        w_old = w_old.reshape((-1, 1))
        if solver == 'gd':
            alpha = 0.001
            precision = 0.0001
            #while True:
            for r in range(500):
                #print(-X.dot(w_old))
                p = 1 / (1 + np.exp(-X.dot(w_old)))
                g = X.T.dot(y - p)
                w_new = w_old + alpha * g
                #print(np.linalg.norm(w_new - w_old))
                #if np.linalg.norm(w_new - w_old) < precision:
                #    print('cnt:', cnt)
                #    break
                w_old = w_new
            self.coef_ = w_new
        elif solver == 'sgd':
            num_iter = 150
            data = np.c_[X, y]
            for j in range(num_iter):
                np.random.shuffle(data)
                X = data[:, :-1]
                #print(X.shape)
                y = data[:, -1].reshape((-1, 1))
                #print(y.shape)
                for i in range(n_samples):
                    alpha = 4 / (1.0 + j + i) + 0.01
                    Xi = X[i, :].reshape((1, -1))
                    #print(Xi.shape)
                    yi = y[i, :].reshape((-1, 1))
                    #print(yi.shape)
                    pi = 1 / (1 + np.exp(-Xi.dot(w_old)))
                    gi = Xi.T.dot(yi - pi)
                    w_new = w_old + alpha * gi
                    w_old = w_new
            self.coef_ = w_new
        else:
            raise ValueError("solver must be one of {'gd', 'sgd'}, got {} instead".format(solver))
        return self

    def predict_proba(self, X):
        X = np.c_[np.ones(len(X)), X]
        p = np.exp(X.dot(self.coef_))/(1+np.exp(X.dot(self.coef_)))
        return p.reshape((p.shape[0], ))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5).astype(int)

def load_data_set():
    data_mat, label_mat = [], []
    fr = open('test_set.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


if __name__ == '__main__':
    data_arr, label_mat = load_data_set()
    lr = LogisticRegression()
    lr.fit(data_arr, label_mat, 'sgd')
    #print(lr.coef_)

