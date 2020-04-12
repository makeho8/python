# needed imports
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator





class TWSVM(BaseEstimator):
    
    def __init__(self, kernel = None, polyconst =1,degree =2,gamma=10,c=None, c_=None):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.degree = degree
        self.gamma = float(gamma)
        self.c = c
        self.c_ = c_
        if self.c is not None: self.c = float(self.c)
        if self.c_ is not None: self.c_ = float(self.c_)
        
        self.kf = {'linear':self.linear, 'polynomial':self.polynomial, 
                   'rbf':self.rbf}
        self.u = None
        self.v = None
        self.w_A = None
        self.b_A = None
        self.w_B = None
        self.b_B = None
        
    def linear(self, x, y):
        return np.dot(x.T, y)

    def polynomial(self, x, y):
        return (self.polyconst + np.dot(x.T, y))**self.degree
    
    def rbf(self,x,y):
        return np.exp(-1.0*self.gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))
    
    def transform(self, X, C):
        K = np.zeros((X.shape[0],C.shape[0]))
        for i in range(X.shape[0]):
            for j in range(C.shape[0]):
                K[i,j] = self.kf[self.kernel](X[i],C[j])
        return K

    def fit(self, X, y):
        
        A = X[np.where(y==1)]
        B = X[np.where(y==-1)]    
        self.C = np.vstack((A,B))
        
        
        m_A = A.shape[0]
        e_A = np.ones((m_A, 1))
        m_B = B.shape[0]
        e_B = np.ones((m_B, 1))    
        m = self.C.shape[0]
            
        if self.kernel==None:
            HA = np.hstack((A, e_A))
            GB = np.hstack((B, e_B))
            I = np.identity(A.shape[1]+1)
        else:
            HA = np.hstack((self.transform(A,self.C), e_A))
            GB = np.hstack((self.transform(B,self.C), e_B))
            I = np.identity(m+1)
            
        # finding w_A, b_A
        K = matrix(GB.dot(np.linalg.inv(HA.T.dot(HA) + 0.0001*I)).dot(GB.T))
        q = matrix((-e_B))
        
        if self.c == None:
            G = matrix(-np.eye(m_B))
            h = matrix(np.zeros((m_B,1)))
        else:
            G = matrix(np.vstack((-np.eye(m_B), np.eye(m_B))))
            h = matrix(np.vstack((np.zeros((m_B,1)), self.c*np.ones((m_B,1)))))
            
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, q, G, h)
        alpha = np.array(sol['x'])
        self.u = -np.linalg.inv(((HA.T).dot(HA) + 0.0001*I)).dot(GB.T).dot(alpha)
        self.b_A = self.u[-1]
        
        if self.kernel == None:
            self.w_A = self.u[:-1] 
        else: 
            self.w_A = None
            
    
        # finding w_B, b_B
        K = matrix(HA.dot(np.linalg.inv(GB.T.dot(GB) + 0.0001*I)).dot(HA.T))
        q = matrix(-e_A)
        
        if self.c_ == None:
            G = matrix(-np.eye(m_A))
            h = matrix(np.zeros((m_A,1)))   
        else:
            G = matrix(np.vstack((-np.eye(m_A),np.eye(m_A))))
            h = matrix(np.vstack((np.zeros((m_A,1)), self.c_*np.ones((m_A,1)))))
            
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, q, G, h)
        beta = np.array(sol['x'])
        self.v = -np.linalg.inv(GB.T.dot(GB) + 0.0001*I).dot(HA.T).dot(beta)
        self.b_B = self.v[-1]
        
        if self.kernel == None:
            self.w_B = self.v[:-1]
        else: 
            self.w_B = None

    def signum(self,X):
        return np.ravel(np.where(X>=0,1,-1))

    def project(self,X):
        if self.kernel== None:
            scoreA = np.abs(np.dot(X,self.w_A)+self.b_A)
            scoreB = np.abs(np.dot(X,self.w_B)+self.b_B)
        else:
            scoreA = np.zeros(X.shape[0])
            scoreB = np.zeros(X.shape[0])
            
            for i in range(X.shape[0]):
                sA=0
                sB=0
                for uj, vj, ct in zip(self.u[:-1], self.v[:-1], self.C):
                    sA += self.kf[self.kernel](X[i],ct)*uj
                    sB += self.kf[self.kernel](X[i],ct)*vj
                scoreA[i] = sA
                scoreB[i] = sB
            scoreA = np.abs(scoreA + self.b_A)
            scoreB = np.abs(scoreB + self.b_B)
        
        score = scoreB - scoreA
        return score
    
    def predict(self,X):
        return self.signum(self.project(X))
    
    def score(self, X, y):
        return 100*np.mean(self.predict(X)==y)

