import numpy as np 
import cvxopt
import cvxopt.solvers
from sklearn.base import BaseEstimator

cvxopt.solvers.options['show_progress'] = False

class SVM(BaseEstimator):
    
    def __init__(self,kernel="linear",polyconst=1,gamma=10,degree=2, c = None):
        
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.gamma = float(gamma)
        self.degree = degree
        self.c = c
        
        if self.c is not None: self.c = float(self.c)
        
        self.kf = {
			"linear":self.linear,
			"rbf":self.rbf,
			"polynomial":self.polynomial
		}
        self._support_vectors = None
        self._alphas = None
        self.intercept = None
        self._n_support = None
        self.weights = None
        self._support_labels = None
        self._indices = None
        
    def linear(self,x,y):
        return np.dot(x.T,y)

    def polynomial(self,x,y):
        return (np.dot(x.T,y) + self.polyconst)**self.degree

    def rbf(self,x,y):
        return np.exp(-1.0*self.gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))

    def transform(self,X):
        K = np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i,j] = self.kf[self.kernel](X[i],X[j])
        return K

    def fit(self,data,labels):
        num_data, num_features = data.shape
        labels = labels.astype(np.double)
        K = self.transform(data)
        P = cvxopt.matrix(np.outer(labels,labels)*K)
        q = cvxopt.matrix(np.ones(num_data)*-1)
        A = cvxopt.matrix(labels,(1,num_data))
        b = cvxopt.matrix(0.0)
        if self.c is not None:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(num_data) * -1),np.diag(np.ones(num_data)))))
            h = cvxopt.matrix(np.hstack((np.zeros(num_data),np.ones(num_data)*self.c)))
        else:
            G = cvxopt.matrix(np.diag(np.ones(num_data) * -1))
            h = cvxopt.matrix(np.zeros(num_data))
        
        alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        is_sv = np.where(alphas>1e-5)[0]
        if self.c is not None:
            S2 = np.where(alphas < self.c)[0]
            M = [val for val in is_sv if val in S2] # intersection of two lists
        else:
            M = is_sv
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = alphas[is_sv]
        self._alphasb = alphas[M]
        self._support_labels = labels[is_sv]
        self._support_labelsb = labels[M]
        self._indices = np.arange(num_data)[is_sv]
        self._indices = np.arange(num_data)[M]
        self.intercept = 0
        for i in range(len(M)):
            self.intercept += self._support_labelsb[i] 
            self.intercept -= np.sum(self._alphas*self._support_labels*K[self._indices[i],is_sv])
        self.intercept /= self._alphasb.shape[0]
        self.weights = np.dot(self._support_labels*self._alphas,self._support_vectors) if self.kernel == "linear" else None

    def signum(self,X):
        return np.where(X>0,1,-1)

    def project(self,X):
        if self.kernel=="linear":
            score = np.dot(X,self.weights.T)+self.intercept
        else:
            score = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha,label,sv in zip(self._alphas,self._support_labels,self._support_vectors):
                    s += alpha*label*self.kf[self.kernel](X[i],sv)
                score[i] = s
            score = score + self.intercept
        return score

    def predict(self,X):
        return self.signum(self.project(X))
    
    def score(self, X, y):
        return 100*np.mean(self.predict(X)==y)
    