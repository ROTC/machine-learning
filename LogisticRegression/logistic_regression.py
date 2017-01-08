import numpy as np 
from sklearn import datasets

class LogisticRegressor:

	def __init__(self):
		self.coef_ = None
		
	def fit(self, X, y):
		X = np.array(X).reshape((len(X), -1))
		X = np.c_[np.ones(len(X)), X]   #add a constant column
		y = np.array(y).reshape((len(y), -1))

		w_old = np.ones(X.shape[1])#/1000			# 系数初始值
		w_old = w_old.reshape((-1, 1))
		r = 100		# 迭代次数上限
		d = 1e-10	# 梯度模值界限
		for n in range(r):
			p = np.exp(X.dot(w_old))/(1+np.exp(X.dot(w_old)))#np.apply_along_axis(prob, 0, w_old, X.T)
			#print(p.shape)
			g = X.T.dot(y - p)
			gm = np.sqrt(g.T.dot(g))[0, 0]
			print(gm)
			if gm < d:
				break
			D = p*(1-p)
			D = np.diag(D.reshape((-1,)))
			#print(T.shape)
			H = -X.T.dot(D).dot(X)
			H_inv = np.linalg.inv(H)
			w_new = w_old - H_inv.dot(g)
			w_old = w_new	
		self.coef_ = w_old

	def predict_proba(self, X):
		return np.exp(X.dot(self.coef_))/(1+np.exp(X.dot(self.coef_)))

	def predict(self, X):
		p = predict_proba()
		
if __name__ == '__main__':
	X = [[2],
		 [1],
		 [0]]
	y = [1, 1, 0]
	lr = LogisticRegressor()
	lr.fit(X, y)


