# needed imports

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn.base import BaseEstimator


class LSW_SVM_cluster_vs_class(BaseEstimator):
    
    def __init__(self, kernel = None,polyconst =1,degree=2,gamma = 1,c1=None,c2=None,c3=None, c4 = None):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.degree = degree
        self.gamma = float(gamma)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        if self.c1 is not None: self.c1 = float(self.c1)
        if self.c2 is not None: self.c2 = float(self.c2)
        if self.c3 is not None: self.c3 = float(self.c3)
        if self.c4 is not None: self.c4 = float(self.c4)
        self.kf = {'linear':self.linear, 'polynomial':self.polynomial, 'rbf':self.rbf}
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
    
    def fit(self, X, y):
        ### Clustering class A, B.
        A = X[np.where(y!=-1)]
        B = X[np.where(y==-1)]
        self.C = np.vstack((A,B))
        # generate the linkage matrix
        L_A = linkage(A, 'ward')
        # number of clusters
        last_A = L_A[-10:, 2]
        #last_Arev = last_A[::-1]
        #idxsA = np.arange(1, len(last_A) + 1)
        #last_Brev = last_B[::-1]
        #idxsB = np.arange(1, len(last_B)+1)
        #plt.plot(idxsA, last_Arev)
        #plt.plot(idxsB, last_Brev)
        acceleration_A = np.diff(last_A, 2)  # 2nd derivative of the distances
        acceleration_rev_A = acceleration_A[::-1]
        #plt.plot(idxsA[:-2] + 1, acceleration_rev_A)
        #plt.show()
        self.k = acceleration_rev_A.argmax() +2  # if idx 0 is the max of this we want 2 clusters
        #print ("clusters_A:", self.k)
        #print('clusters_B:', self.l)
        # Retrieve the clusters_A, clusters_B
        clusters_A = fcluster(L_A, self.k, criterion='maxclust')
        #print(clusters_A, clusters_B)
        L_B = linkage(B, method = 'ward')
        last_B = L_B[-10:, 2]
        acceleration_B = np.diff(last_B, 2)
        acceleration_rev_B = acceleration_B[::-1]
        self.l = acceleration_rev_B.argmax() +2
        clusters_B = fcluster(L_B, self.l, criterion = 'maxclust')
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
            
        self.m_A = A.shape[0]
        e_A = np.ones((self.m_A, 1))
        IA = np.identity(self.m_A)
        self.m_B = B.shape[0]
        e_B = np.ones((self.m_B, 1))
        IB = np.identity(self.m_B)
        
        if self.kernel == None:
            HA = np.hstack((A, e_A))
            GB = np.hstack((B, e_B))
            I = np.identity(n+1)
        else:
            HA = np.hstack((self.transform(A,self.C),e_A))
            GB = np.hstack((self.transform(B,self.C),e_B))
            I = np.identity(m+1)
            Y = (self.c1/self.c2)*I - (self.c1/self.c2)*GB.T.dot(np.linalg.inv(IB + (self.c1/self.c2)*GB.dot(GB.T))).dot(GB)
            Z = (self.c3/self.c4)*I - (self.c3/self.c4)*HA.T.dot(np.linalg.inv(IA +(self.c3/self.c4)*HA.dot(HA.T))).dot(HA)
        # class A
        self.WA = []
        self.bA = []
        for i in range(self.k):
            mAi = self.Z_A[i].shape[0]
            eAi = np.ones((mAi, 1))
            IAi = np.identity(mAi)
            if self.kernel == None:
                H_i = np.hstack((self.Z_A[i], eAi))
                self.ui = np.linalg.inv((1/self.c1)*H_i.T.dot(H_i) + GB.T.dot(GB) + (self.c2/self.c1)*I).dot(GB.T).dot(e_B)
            else:
                H_i = np.hstack((self.transform(self.Z_A[i],self.C), eAi))
                self.ui = (Y - Y.dot(H_i.T).dot(np.linalg.inv(self.c1*IAi + H_i.dot(Y).dot(H_i.T))).dot(H_i).dot(Y)).dot(GB.T).dot(e_B)
            bi = self.ui[-1]
            wi = self.ui[:-1]
            self.WA.append(wi)
            self.bA.append(bi)
        
        # class B
        self.WB = []
        self.bB = []
        for j in range(self.l):
            mBj = self.Z_B[j].shape[0]
            eBj = np.ones((mBj, 1))
            IBj = np.identity(mBj)
            if self.kernel == None:
                G_j = np.hstack((self.Z_B[j], eBj))
                self.vj = np.linalg.inv((1/self.c3)*G_j.T.dot(G_j) + HA.T.dot(HA) + (self.c4/self.c3)*I).dot(HA.T).dot(e_A)
            else:
                G_j = np.hstack((self.transform(self.Z_B[j],self.C), eBj))
                self.vj = (Z - Z.dot(G_j.T).dot(np.linalg.inv(self.c3*IBj + G_j.dot(Z).dot(G_j.T))).dot(G_j).dot(Z)).dot(HA.T).dot(e_A)
            wj = self.vj[:-1]
            bj = self.vj[-1]
            self.WB.append(wj)
            self.bB.append(bj)    

    def signum(self,X):
        return np.ravel(np.where(X>=0,1,-1))

    def project(self,X):
        scoreA = np.zeros(X.shape[0])
        scoreB = np.zeros(X.shape[0])
        score_arrayA = np.zeros((self.k,X.shape[0]))
        score_arrayB = np.zeros((self.l,X.shape[0]))
        if self.kernel== None:
            for i in range(self.k):
                scoreAi = ((self.Z_A[i].shape[0])/(self.m_A))*(np.dot(X,self.WA[i]) + self.bA[i]).ravel()
                score_arrayA[i] = scoreAi
            scoreA = np.sum(score_arrayA, axis = 0)
            for j in range(self.l):
                scoreBj = ((self.Z_B[j].shape[0])/(self.m_B))*(np.dot(X, self.WB[j]) + self.bB[j]).ravel()
                score_arrayB[j] = scoreBj
            scoreB = np.sum(score_arrayB, axis = 0)
        else:
            for i in range(self.k):
                scoreAi = np.zeros(X.shape[0])
                for j in range(X.shape[0]):
                    sA=0
                    for uj, ct in zip(self.WA[i], self.C):
                        sA += self.kf[self.kernel](X[j],ct)*uj
                    scoreAi[j] = sA + self.bA[i]
                scoreA += ((self.Z_A[i].shape[0])/(self.m_A))*scoreAi
            for i in range(self.l):
                scoreBi = np.zeros(X.shape[0])
                for j in range(X.shape[0]):
                    sB=0
                    for vj, ct in zip(self.WB[i], self.C):
                        sB += self.kf[self.kernel](X[j],ct)*vj
                    scoreBi[j] = sB + self.bB[i]
                scoreB += ((self.Z_B[i].shape[0])/(self.m_B))*scoreBi
        
        score = scoreB - scoreA
        return score
    
    def predict(self,X):
        return self.signum(self.project(X))
    
    def score(self, X, y):
        return 100*np.mean(self.predict(X)==y)
