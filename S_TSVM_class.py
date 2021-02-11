# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator


class S_TSVM(BaseEstimator):
    
    def __init__(self, kernel = None, polyconst =1, degree = 2, gamma = 1,c1=None, c2=None, c3=None):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.degree = degree
        self.gamma = float(gamma)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        if self.c1 is not None: self.c1 = float(self.c1)
        if self.c2 is not None: self.c2 = float(self.c2)
        if self.c3 is not None: self.c3 = float(self.c3)
        
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
        self.k = acceleration_rev_A.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        self.l = acceleration_rev_B.argmax() + 2
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
        
        if self.kernel == None:
            sigma_A = np.zeros((n, n))
            for i in range(self.k):
                sigma_i = np.cov(self.Z_A[i].T)
                sigma_A += sigma_i
            
            zer_0 = np.zeros((n,1))
            zer_1 = np.zeros((1,n+1))
            J_ = np.hstack((sigma_A, zer_0))
            J = np.vstack((J_, zer_1))
        else:
            sigma_A = np.zeros((m,m))
            for i in range(self.k):
                pi = self.Z_A[i].shape[0]
                Mi = np.zeros((self.Z_A[i].shape))
                for j in range(len(Mi)):
                    Mi[j] = np.mean(self.Z_A[i], axis = 0)                    
                sigma_i = ((self.transform(self.Z_A[i],self.C)-self.transform(Mi,self.C))/np.sqrt(pi)).T.dot(((self.transform(self.Z_A[i],self.C)-self.transform(Mi,self.C))/np.sqrt(pi)))
                sigma_A += sigma_i
            zer_0 = np.zeros((m,1))
            zer_1 = np.zeros((1,m+1))
            J_ = np.hstack((sigma_A, zer_0))
            J = np.vstack((J_, zer_1))
            
        if self.kernel == None:
            sigma_B = np.zeros((n, n))
            for i in range(self.l):
                sigma_i = np.cov(self.Z_B[i].T)
                sigma_B += sigma_i
            
            zer_0 = np.zeros((n,1))
            zer_1 = np.zeros((1,n+1))
            F_ = np.hstack((sigma_B, zer_0))
            F = np.vstack((F_, zer_1))
        else:
            sigma_B = np.zeros((m,m))
            for i in range(self.l):
                pi = self.Z_B[i].shape[0]
                Mi = np.zeros((self.Z_B[i].shape))
                for j in range(len(Mi)):
                    Mi[j] = np.mean(self.Z_B[i], axis = 0)                    
                sigma_i = ((self.transform(self.Z_B[i],self.C)-self.transform(Mi,self.C))/np.sqrt(pi)).T.dot(((self.transform(self.Z_B[i],self.C)-self.transform(Mi,self.C))/np.sqrt(pi)))
                sigma_B += sigma_i
            zer_0 = np.zeros((m,1))
            zer_1 = np.zeros((1,m+1))
            F_ = np.hstack((sigma_B, zer_0))
            F = np.vstack((F_, zer_1))
        
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
        # weights and bias of class A
        if self.c3 == None:
            K = matrix(GB.dot(np.linalg.inv(HA.T.dot(HA) + self.c2*I)).dot(GB.T))
        else:
            K = matrix(GB.dot(np.linalg.inv(HA.T.dot(HA)+ self.c3*J + self.c2*I)).dot(GB.T))
        
        q = matrix((-e_B))
        if self.c1 == None:
            G = matrix(-np.eye(m_B))
            h = matrix(np.zeros((m_B,1)))
        else: 
            G = matrix(np.vstack((-np.eye(m_B), np.eye(m_B))))
            h = matrix(np.vstack((np.zeros((m_B,1)), self.c1*np.ones((m_B,1)))))
            
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, q, G, h)
        alpha = np.array(sol['x'])
        if self.c3 == None:
            self.u = -np.linalg.inv(((HA.T).dot(HA) + self.c2*I)).dot(GB.T).dot(alpha)
        else:
            self.u = -np.linalg.inv(((HA.T).dot(HA)+ self.c3*J + self.c2*I)).dot(GB.T).dot(alpha)
        
        self.b_A = self.u[-1]
        if self.kernel == None:
            self.w_A = self.u[:-1]
        else:
            self.w_A = None
    
        # weights and bias of class B
        if self.c3 == None:
            K = matrix(HA.dot(np.linalg.inv(GB.T.dot(GB) + self.c2*I)).dot(HA.T))
        else:
            K = matrix(HA.dot(np.linalg.inv(GB.T.dot(GB)+ self.c3*F + self.c2*I)).dot(HA.T))
        
        q = matrix(-e_A)
        if self.c1 == None:
            G = matrix(-np.eye(m_A))
            h = matrix(np.zeros((m_A,1)))
        else:
            G = matrix(np.vstack((-np.eye(m_A),np.eye(m_A))))
            h = matrix(np.vstack((np.zeros((m_A,1)), self.c1*np.ones((m_A,1)))))
            
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, q, G, h)
        beta = np.array(sol['x'])
        if self.c3 == None:
            self.v = -np.linalg.inv(GB.T.dot(GB) + self.c2*I).dot(HA.T).dot(beta)
        else:
            self.v = -np.linalg.inv(GB.T.dot(GB)+ self.c3*F + self.c2*I).dot(HA.T).dot(beta)
        
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
    
