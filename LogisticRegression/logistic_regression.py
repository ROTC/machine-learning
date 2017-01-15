import numpy as np
from sklearn import datasets, linear_model

class LogisticRegression(object):
    """Logistic regression model.

    Attributes
    -----------
    coef_ : array, shape = (m, )
            Estimated coefficients for the linear classification problem

    """

    def fit(self, X, y):
        """Fit logistic model

        Parameters
        ----------
        X : list or array, shape = (m, n),
            Training data

        y : list or array, shape = (m, ),
            Sample labels

        Returns
        ----------
        self : returns an instance of self.
        """

        n_samples = len(X)
        X = np.array(X).reshape((n_samples, -1))
        print(X)
        n_features = X.shape[1]
        X = np.c_[np.ones(n_samples), X]   #add a constant column
        y = np.array(y).reshape((n_samples, -1))

        w_old = np.zeros(n_features+1)#/1000			# 系数初始值
        w_old = w_old.reshape((-1, 1))
        #print(w_old)
        #r = 100		# 迭代次数上限
        #d = 1e-10	# 梯度模值界限
        ## 牛顿法
        #for n in range(r):
        #    p = np.exp(X.dot(w_old))/(1+np.exp(X.dot(w_old)))
        #    #print(p.shape)
        #    g = X.T.dot(y - p)
        #    gm = np.sqrt(g.T.dot(g))[0, 0]
        #    print(gm)
        #    if gm < d:
        #        break
        #    D = p*(1-p)
        #    D = np.diag(D.reshape((-1,)))
        #    #print(T.shape)
        #    H = -X.T.dot(D).dot(X)
        #    H_inv = np.linalg.inv(H)
        #    w_new = w_old - H_inv.dot(g)
        #    w_old = w_new
        #self.coef_ = w_old
        # 随机梯度下降法
        #print(X)
        #print(y)
        data = np.c_[X, y]
        #print(data)
        r = 100
        c = 0.01
        for n in range(r):
            np.random.shuffle(data)     #shuffle rows inplace
            #print(data)
            X = data[:, :-1]
            y = data[:, -1:]
            p = np.exp(X.dot(w_old))/(1+np.exp(X.dot(w_old)))
            g = X.T.dot(y - p)
            gm = np.sqrt(g.T.dot(g))[0, 0]
            print(gm)
            for i in range(n_samples):
                xi = X[i, :].reshape((1, -1))
                #print(X)
                #print(xi)
                yi = y[i, :].reshape((1, 1))
                pi = np.exp(xi.dot(w_old))/(1+np.exp(xi.dot(w_old)))
                gi = xi.T.dot(yi - pi)
                w_new = w_old - c * gi
                w_old = w_new

    def predict_proba(self, X):
        X = np.c_[np.ones(len(X)), X]
        p = np.exp(X.dot(self.coef_))/(1+np.exp(X.dot(self.coef_)))
        return p.reshape((p.shape[0], ))

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5).astype(int)

if __name__ == '__main__':
    X = [[2],[1],[0]]
    y = [1, 1, 0]
    lr = LogisticRegression()
    lr.fit(X, y)
    X1 = [[1.5], [0.5]]
    print(lr.predict_proba(X1))
    print(lr.predict(X1))

    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    n_samples = len(X_digits)
    X_train = X_digits[:.9 * n_samples]
    y_train = y_digits[:.9 * n_samples]
    X_test = X_digits[.9 * n_samples:]
    y_test = y_digits[.9 * n_samples:]

    lr_skl = linear_model.LogisticRegression()
    lr_skl.fit(X_train, y_train)
    p_skl = lr_skl.predict_proba(X_test)
    lr_r = LogisticRegression()
    lr_r.fit(X_train, y_train)
    p_r = lr_r.predict_proba(X_test)
