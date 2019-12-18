import numpy as np
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

### Load data
def load_data(name):
    name = name
    data_old = np.genfromtxt(name)
    #print(data)
    labels = data_old[:,-1].reshape((270,))
    idx = np.where(labels == 2)
    labels[idx] = -1
    data = np.delete(data_old, np.s_[-1], axis = 1)
    #print(data, labels)
    data_train = data[:240]
    data_test = data[240:]
    labels_train = labels[:240]
    labels_test = labels[240:]
    return data_train, labels_train, data_test, labels_test

def split_train():
    idx1 = np.where(labels_train == 1)
    idx2 = np.where(labels_train == -1)
    A_train = data_train[idx1]
    yA_train = labels_train[idx1]
    B_train = data_train[idx2]
    yB_train = labels_train[idx2]
    C_train = np.vstack((A_train,B_train))
    y_train = np.hstack((yA_train, yB_train))
    return A_train, yA_train, B_train, yB_train, C_train, y_train

def split_test():
    idx3 = np.where(labels_test == 1)
    idx4 = np.where(labels_test == -1)
    A_test = data_test[idx3]
    yA_test = labels_test[idx3]
    B_test = data_test[idx4]
    yB_test = labels_test[idx4]
    C_test = np.vstack((A_test,B_test))
    y_test = np.hstack((yA_test, yB_test))
    return A_test, yA_test, B_test, yB_test, C_test, y_test

data_train, labels_train, data_test, labels_test = load_data('heart.dat')
A_train, yA_train, B_train, yB_train, C_train, y_train = split_train()
A_test, yA_test, B_test, yB_test, C_test, y_test = split_test()

###solving by dual_TWSVM
def TSVM_solver(A, B, c1, c2):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2

    ## DTWSVM1
    # Build K1, q1, G1, h1
    e1 = np.ones((m1,1))
    e2 = np.ones((m2,1))
    H = np.hstack((A,e1))
    G = np.hstack((B,e2))
    K1 = matrix(G.dot(np.linalg.inv(H.T.dot(H))).dot(G.T))
    q1 = matrix(-e2)
    G1 = matrix(np.vstack((-np.eye(m2),np.eye(m2))))
    h1 = matrix(np.vstack((np.zeros((m2,1)),c1*np.ones((m2,1)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(K1,q1, G1, h1)

    al = np.array(sol['x'])
    #print('alpha = \n', al.T)

    ## DTWSVM2
    # build K2, q2, G2, h2
    K2 = matrix(H.dot(np.linalg.inv(G.T.dot(G))).dot(H.T))
    q2 = matrix(-e1)
    G2 = matrix(np.vstack((-np.eye(m1),np.eye(m1))))
    h2 = matrix(np.vstack((np.zeros((m1,1)),c2*np.ones((m1,1)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(K2,q2,G2,h2)
    gam = np.array(sol['x'])
    #print('gamma = \n', gam.T)

    S1 = np.where(al > 1e-5)[0]
    S2 = np.where(gam > 1e-5)[0]
    alS1 = al[S1]
    gamS2 = gam[S2]
    GS1 = G[S1,:]
    HS2 = H[S2,:]
    u = -np.linalg.inv(H.T.dot(H)).dot(GS1.T).dot(alS1)
    v = np.linalg.inv(G.T.dot(G)).dot(HS2.T).dot(gamS2)
    #print(u,v)
    w1 = u[:-1]
    b1 = u[-1]
    w2 = v[:-1]
    b2 = v[-1]
    return w1, b1, w2, b2

def predict_class(x,w1,b1,w2,b2):
    y1 = np.abs(x.T.dot(w1) + b1)
    y2 = np.abs(x.T.dot(w2) + b2)
    y_pred_x = None
    if y1 <= y2:
        y_pred_x = 1.
    else:
        y_pred_x = -1.
        
    return y_pred_x

def predict_TSVM(X,w1,b1,w2,b2):
    y_pred = []
    for i in range(X.shape[0]):
        y_pred_i = predict_class(X[i,:],w1,b1,w2,b2)
        y_pred.append(y_pred_i)
    
    y_pred_TSVM = np.array(y_pred)
    return y_pred_TSVM

import time
start_time = time.time()
w1_tsvm, b1_tsvm, w2_tsvm, b2_tsvm = TSVM_solver(A_train,B_train, c1=1000, c2=1000)
end_time = time.time()
print('total run-time of TSVM: %f ms' %((end_time - start_time)*1000))
y_pred_train = predict_TSVM(C_train,w1_tsvm,b1_tsvm,w2_tsvm,b2_tsvm)
accuracy_train = np.mean(y_train == y_pred_train)
print('training accuracy of TSVM: ', 100*accuracy_train)

y_pred_test = predict_TSVM(C_test,w1_tsvm,b1_tsvm,w2_tsvm,b2_tsvm)
accuracy_test = np.mean(y_test == y_pred_test)
print('testing accuracy of TSVM: ', 100*accuracy_test)

#Solving by dual problems of soft SVM

def SVM_solver(A, B, c):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2    
    
    #build K
    V = np.concatenate((A, -B), axis = 0)
    K = matrix(V.dot(V.T))    
    e = matrix(-np.ones((m, 1)))
    
    #build A, b, G, h
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.vstack((np.zeros((m, 1)), c*np.ones((m, 1)))))
    Y = matrix(y_train.reshape((-1, m)))
    b = matrix(np.zeros((1, 1)))
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(K, e, G, h, Y, b)
    al = np.array(sol['x'])
    #print('alpha = \n', al.T)

    S = np.where(al > 1e-5)[0]
    S2 = np.where(al < .99*c)[0]
    M = [val for val in S if val in S2] # intersection of two lists
    VS = V[S, :]
    alS = al[S]
    yM = y_train[M]
    CM = C_train[M,:]
    w_dual = VS.T.dot(alS).reshape(-1,1)
    b_dual = np.mean(yM - CM.dot(w_dual))
    #print(w_dual, b_dual)
    return w_dual, b_dual



def predict_SVM(X,w,b):
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if np.sign(X[i,:].dot(w) + b) == 1:
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    return y_pred

start_time = time.time()
w_dual, b_dual = SVM_solver(A_train, B_train, c=100)
end_time = time.time()
print('total run-time of SVM: %f ms' %((end_time - start_time)*1000))
#print(w_sklearn, b_sklearn)
y_train_pred = predict_SVM(C_train,w_dual,b_dual)
print('training accuracy of softSVM: %f' %(100*np.mean(y_train == y_train_pred)))
y_test_pred = predict_SVM(C_test,w_dual,b_dual)
print('testing accuracy of softSVM: %f' %(100*np.mean(y_test == y_test_pred)))

### TSVM with kernel (KTWSVM)
def rbf(x,y,gam=0.1):
    return np.exp(-gam*((np.linalg.norm(x-y))**2))

def Kernel_1(x,Y,gam=0.1):
    n = Y.shape[1]
    K = np.zeros((1,n))
    for j in range(n):
        K[0,j] = rbf(x,Y[:,j],gam=0.1)
    return K

def Kernel(X,Y, gam=0.1):
    m = X.shape[0]
    n = Y.shape[1]
    K = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            K[i,j] = rbf(X[i,:],Y[:,j],gam=0.1)
            
    return K

def KTSVM_solver(A,B,c1,c2):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2
    C = np.vstack((A, B))
    ## DKTWSVM1
    # Build K1, q1, G1, h1
    e1 = np.ones((m1,1))
    e2 = np.ones((m2,1))
    S = np.hstack((Kernel(A,C.T,1),e1))
    R = np.hstack((Kernel(B,C.T,1),e2))
    I = np.eye(m+1)
    K1 = matrix(R.dot(np.linalg.inv(S.T.dot(S)+0.001*I)).dot(R.T))
    q1 = matrix(-e2)
    G1 = matrix(np.vstack((-np.eye(m2),np.eye(m2))))
    h1 = matrix(np.vstack((np.zeros((m2,1)),c1*np.ones((m2,1)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(K1,q1, G1, h1)

    al = np.array(sol['x'])
    #print('alpha = \n', al.T)

    ## DKTWSVM2
    # build K2, q2, G2, h2
    K2 = matrix(S.dot(np.linalg.inv(R.T.dot(R)+0.001*I)).dot(S.T))
    q2 = matrix(-e1)
    G2 = matrix(np.vstack((-np.eye(m1),np.eye(m1))))
    h2 = matrix(np.vstack((np.zeros((m1,1)),c2*np.ones((m1,1)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(K2,q2,G2,h2)
    gam = np.array(sol['x'])
    #print('gamma = \n', gam.T)

    S1 = np.where(al > 1e-5)[0]
    S2 = np.where(gam > 1e-5)[0]
    alS1 = al[S1]
    gamS2 = gam[S2]
    RS1 = R[S1,:]
    SS2 = S[S2,:]
    z1 = -np.linalg.inv(S.T.dot(S)+0.001*I).dot(RS1.T).dot(alS1)
    z2 = np.linalg.inv(R.T.dot(R)+0.001*I).dot(SS2.T).dot(gamS2)
    #print(z1,z2)
    u1 = z1[:-1]
    b1 = z1[-1][0]
    u2 = z2[:-1]
    b2 = z2[-1][0]
    return u1, b1, u2, b2

#w1_train, b1_train, w2_train, b2_train = KTSVM_solver(A_train,B_train,A_train_labels,B_train_labels,c1=1000,c2=1000)

### Load data

#import pandas as pd
#data_old = pd.read_csv('hepatitis.data', sep = ',', header = None).values

u1_Ktsvm, b1_Ktsvm, u2_Ktsvm, b2_Ktsvm = KTSVM_solver(A_train,B_train,c1=1000,c2=1000)

def predict_KTSVM_1(x,C,u1,b1,u2,b2):
    
    y1 = np.abs((Kernel_1(x, C.T, 1)).dot(u1)+b1)
    y2 = np.abs((Kernel_1(x, C.T, 1)).dot(u2)+b2)
    if y1 < y2:
        y_i = 1
    else:
        y_i = -1
    
    return y_i

def predict_KTSVM(X,C,u1,b1,u2,b2):
    y_pred = []
    for i in range(X.shape[0]):
        y_i = None
        y1 = np.abs((Kernel_1(X[i,:], C.T, gam=0.1)).dot(u1)+b1)
        y2 = np.abs((Kernel_1(X[i,:], C.T, gam=0.1)).dot(u2)+b2)
        if y1 < y2:
            y_i = 1
        else:
            y_i = -1
    
        y_pred.append(y_i)
    return y_pred
    

y_train_pred_KTSVM = predict_KTSVM(C_train,C_train,u1_Ktsvm,b1_Ktsvm,u2_Ktsvm,b2_Ktsvm)
accuracy_train = np.mean(y_train == y_train_pred_KTSVM)
print('training accuracy of KTSVM: ', 100*accuracy_train)

y_test_pred_KTSVM = predict_KTSVM(C_test,C_train,u1_Ktsvm,b1_Ktsvm,u2_Ktsvm,b2_Ktsvm)
accuracy_test = np.mean(y_test == y_test_pred_KTSVM)
print('testing accuracy of KTSVM: ', 100*accuracy_test)