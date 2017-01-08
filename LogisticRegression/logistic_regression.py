import numpy as np 
from sklearn import datasets

class LogisticRegressor:

	def __init__(self):
		self.coef_ = None

	def fit(self, X, y):
		# 求概率函数
		def prob(w, x):
			return np.exp(w.T.dot(x))/(1+np.exp(w.T.dot(x)))
		# 求p(1-p)，用来求W矩阵
		def pprob(w, x):
			p1 = np.exp(w.T.dot(x))/(1+np.exp(w.T.dot(x)))
			return p1*(1 - p1)

		X = np.array(X).reshape((len(X), -1))
		X = np.c_[np.ones(len(X)), X]   #add a constant column
		y = np.array(y).reshape((len(y), -1))

		w_old = np.ones(X.shape[1])#/1000			# 系数初始值
		w_old = w_old.reshape((-1, 1))
		r = 100		# 迭代次数上限
		d = 1e-10	# 梯度模值界限
		for n in range(r):
			p = np.apply_along_axis(prob, 0, w_old, X.T)
			#print(p.shape)
			g = X.T.dot(y - p)
			gm = np.sqrt(g.T.dot(g))[0, 0]
			#print(gm)
			if gm < d:
				break
			D = np.apply_along_axis(pprob, 0, w_old, X.T)
			D = np.diag(D.reshape((-1,)))
			#print(T.shape)
			H = -X.T.dot(D).dot(X)
			H_inv = np.linalg.inv(H)
			w_new = w_old - H_inv.dot(g)
			w_old = w_new	
		self.coef_ = w_old

	def predict_proba(X):
		#求概率函数
		def prob(w, x):
			return np.exp(w.T.dot(x))/(1+np.exp(w.T.dot(x)))


if __name__ == '__main__':
	X = [[2],
		 [1],
		 [0]]
	y = [1, 1, 0]
	lr = LogisticRegressor()
	lr.fit(X, y)


