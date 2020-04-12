# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator


class HMTWSVM2(BaseEstimator):
    
    def __init__(self, kernel = None,polyconst =1,degree=2,gamma = 1,ci=None, ci_=None, c=None, c_=None):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.degree = degree
        self.gamma = float(gamma)
        self.ci = ci
        self.ci_ = ci_
        self.c = c
        self.c_ = c_
        if self.ci is not None: self.ci = float(self.ci)
        if self.ci_ is not None: self.ci_ = float(self.ci_)
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
        self.k = None
        self.l = None
        
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
    
    def sigma(self, X,C):
        if self.kernel == None:
            return np.cov(X.T)
        else:
            p = X.shape[0]
            M = np.zeros((X.shape))
            for j in range(len(M)):
                M[j] = np.mean(M, axis = 0)                    
            return ((self.transform(X,C)-self.transform(M,C))/np.sqrt(p)).T.dot(((self.transform(X,C)-self.transform(M,C))/np.sqrt(p)))
               
    def fit(self, X, y):
        ### Clustering class A, B.
        A = X[np.where(y!=-1)]
        B = X[np.where(y==-1)]
        self.C = np.vstack((A,B))
        # generate the linkage matrix
        L_A = linkage(A, 'ward')
        L_B = linkage(B, 'ward')
        # number of clusters
        last_A = L_A[-10:, 2]
        last_B = L_B[-10:, 2]
        #last_rev = last_A[::-1]
        #idxs = np.arange(1, len(last_A) + 1)
        #plt.plot(idxs, last_rev)
        
        acceleration_A = np.diff(last_A, 2)  # 2nd derivative of the distances
        acceleration_rev_A = acceleration_A[::-1]
        acceleration_B = np.diff(last_B, 2)
        acceleration_rev_B = acceleration_B[::-1]
        #plt.plot(idxs[:-2] + 1, acceleration_rev_A)
        #plt.show()
        self.k = acceleration_rev_A.argmax() + 1  # if idx 0 is the max of this we want 2 clusters
        self.l = acceleration_rev_B.argmax() + 1
        #print ("clusters_A:", self.k)
        # Retrieve the clusters_A, clusters_B
        from scipy.cluster.hierarchy import fcluster
        clusters_A = fcluster(L_A, self.k, criterion='maxclust')
        clusters_B = fcluster(L_B, self.l, criterion='maxclust')
        #print(clusters_A, clusters_B)
        
        # Visualizing clusters_A
        #plt.figure(figsize=(10, 8))
        #plt.scatter(A[:,0], A[:,1], c=clusters_A, cmap='prism')  # plot points with cluster dependent colors
        #plt.show()
        
        

        self.labels_A = np.unique(clusters_A)
        self.Z_A = []
        if self.k != 1:
            for i in range(self.k):
                Ai = A[np.where(clusters_A == self.labels_A[i])]
                self.Z_A.append(Ai)
        else:
            self.Z_A.append(A)

        self.labels_B = np.unique(clusters_B)
        self.Z_B = []
        if self.l != 1:
            for i in range(self.l):
                Bi = B[np.where(clusters_B == self.labels_B[i])]
                self.Z_B.append(Bi)
        else:
            self.Z_B.append(B)
        
        n = A.shape[1]
        m = self.C.shape[0]
            
        m_A = A.shape[0]
        e_A = np.ones((m_A, 1))
        m_B = B.shape[0]
        e_B = np.ones((m_B, 1))
        
        if self.kernel == None:
            HA = np.hstack((A, e_A))
            GB = np.hstack((B, e_B))
            I = np.identity(n+1)
        else:
            HA = np.hstack((self.transform(A,self.C),e_A))
            GB = np.hstack((self.transform(B,self.C),e_B))
            I = np.identity(m+1)
        #caculate covariance matrix of class A
        sigma_A = np.zeros((I.shape[0]-1, I.shape[0]-1))
        for i in range(self.k):
            sigmaA_i = self.sigma(self.Z_A[i],self.C)
            sigma_A += sigmaA_i
            
        zer_0 = np.zeros((I.shape[0]-1,1))
        zer_1 = np.zeros((1,I.shape[0]))
        J_ = np.hstack((sigma_A, zer_0))
        J = np.vstack((J_, zer_1))
        #caculate covariance matrix of class B
        sigma_B = np.zeros((I.shape[0]-1, I.shape[0]-1))
        for i in range(self.l):
            sigmaB_i = self.sigma(self.Z_B[i],self.C)
            sigma_B += sigmaB_i
            
        zer_0 = np.zeros((I.shape[0]-1,1))
        zer_1 = np.zeros((1,I.shape[0]))
        F_ = np.hstack((sigma_B, zer_0))
        F = np.vstack((F_, zer_1))
        #finding k hyperplanes of class B equivalent to k clusters of A
        self.WB = []
        self.bB = []
        for i in range(self.k):
            mAi = self.Z_A[i].shape[0]
            eAi = np.ones((mAi, 1))
            if self.kernel == None:
                H_i = np.hstack((self.Z_A[i], eAi))
            else:
                H_i = np.hstack((self.transform(self.Z_A[i],self.C),eAi))
            
            if self.c_==None:
                K = matrix(H_i.dot(np.linalg.inv(GB.T.dot(GB) + 0.0001*I)).dot(H_i.T))
            else:
                K = matrix(H_i.dot(np.linalg.inv(GB.T.dot(GB)+ self.c_ * F + 0.0001*I)).dot(H_i.T))
            
            q = matrix((-eAi))
            if self.c==None:
                G = matrix(-np.eye(mAi))
                h = matrix(np.zeros((mAi,1)))
            else:
                G = matrix(np.vstack((-np.eye(mAi), np.eye(mAi))))
                h = matrix(np.vstack((np.zeros((mAi,1)), self.c*np.ones((mAi,1)))))
            
            solvers.options['show_progress'] = False
            sol = solvers.qp(K, q, G, h)
            gam = np.array(sol['x'])
            if self.c_==None:
                self.vi = -np.linalg.inv(GB.T.dot(GB) + 0.0001*I).dot(H_i.T).dot(gam)
            else:
                self.vi = -np.linalg.inv(GB.T.dot(GB)+ self.c_*F + 0.0001*I).dot(H_i.T).dot(gam)
            
            wi = self.vi[:-1]
            bi = self.vi[-1]
            self.WB.append(wi)
            self.bB.append(bi)
        
        self.wB_mean = np.mean(self.WB, axis = 0)
        self.bB_mean = np.mean(self.bB)
        #finding l hyperplanes of class A equivalent to l clusters of B
        self.WA = []
        self.bA = []
        for i in range(self.l):
            mBi = self.Z_B[i].shape[0]
            eBi = np.ones((mBi, 1))
            if self.kernel==None:
                G_i = np.hstack((self.Z_B[i], eBi))
            else:
                G_i = np.hstack((self.transform(self.Z_B[i],self.C),eBi))
            
            if self.ci_==None:
                K = matrix(G_i.dot(np.linalg.inv(HA.T.dot(HA) + 0.0001*I)).dot(G_i.T))
            else:
                K = matrix(G_i.dot(np.linalg.inv(HA.T.dot(HA)+ self.ci_*J + 0.0001*I)).dot(G_i.T))
            
            q = matrix(-eBi)
            if self.ci==None:
                G = matrix(-np.eye(mBi))
                h = matrix(np.zeros((mBi,1)))
            else:
                G = matrix(np.vstack((-np.eye(mBi),np.eye(mBi))))
                h = matrix(np.vstack((np.zeros((mBi,1)), self.ci*np.ones((mBi,1)))))
            
            solvers.options['show_progress'] = False
            sol = solvers.qp(K, q, G, h)
            alpha = np.array(sol['x'])
            if self.ci_==None:
                self.ui = -np.linalg.inv(HA.T.dot(HA) + 0.0001*I).dot(G_i.T).dot(alpha)
            else:
                self.ui = -np.linalg.inv(HA.T.dot(HA)+ self.ci_*J + 0.0001*I).dot(G_i.T).dot(alpha)
            
            wi = self.ui[:-1]
            bi = self.ui[-1]
            self.WA.append(wi)
            self.bA.append(bi)
            
        self.wA_mean = np.mean(self.WA, axis =0)
        self.bA_mean = np.mean(self.bA)

    def signum(self,X):
        return np.ravel(np.where(X>=0,1,-1))

    def project(self,X):
        if self.kernel== None:
            scoreA = np.abs(np.dot(X,self.wA_mean)+self.bA_mean)
            scoreB = np.abs(np.dot(X,self.wB_mean)+self.bB_mean)
        else:
            scoreA = np.zeros(X.shape[0])
            scoreB = np.zeros(X.shape[0])
            
            for i in range(X.shape[0]):
                sA=0
                sB=0
                for uj, vj, ct in zip(self.wA_mean, self.wB_mean, self.C):
                    sA += self.kf[self.kernel](X[i],ct)*uj
                    sB += self.kf[self.kernel](X[i],ct)*vj
                scoreA[i] = sA
                scoreB[i] = sB
            scoreA = np.abs(scoreA + self.bA_mean)
            scoreB = np.abs(scoreB + self.bB_mean)
        
        score = scoreB - scoreA
        return score
    
    def predict(self,X):
        return self.signum(self.project(X))
    
    def score(self, X, y):
        return 100*np.mean(self.predict(X)==y)
    