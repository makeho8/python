
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(21)
from matplotlib.backends.backend_pdf import PdfPages

def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0,2])
    mean2 = np.array([2,0])
    cov = np.array([[0.8, 0.6],[0.6, 0.8]])
    A = np.random.multivariate_normal(mean1, cov, 100)
    A_labels = np.ones(len(A))
    B = np.random.multivariate_normal(mean2, cov, 100)
    B_labels = np.ones(len(B))*(-1)
    return A, A_labels, B, B_labels

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.8, 1.0], [1.0, 1.8]])
    A = np.random.multivariate_normal(mean1, cov, 100)
    A_labels = np.ones(len(A))
    B = np.random.multivariate_normal(mean2, cov, 100)
    B_labels = np.ones(len(B)) * (-1)
    return A, A_labels, B, B_labels
    
def gen_non_lin_separable_data():
    mean1 = np.array([-1.1, 2])
    mean2 = np.array([1.1, -1])
    mean3 = np.array([4.1, -4])
    mean4 = np.array([-4.1, 4])
    cov = np.array([[1.0,0.8],[0.8,1.0]])
    A = np.random.multivariate_normal(mean1, cov, 50)
    A = np.vstack((A, np.random.multivariate_normal(mean3, cov, 50)))
    A_labels = np.ones(len(A))
    B = np.random.multivariate_normal(mean2, cov, 50)
    B = np.vstack((B, np.random.multivariate_normal(mean4, cov, 50)))
    B_labels = np.ones(len(B))*(-1)
    return A, A_labels, B, B_labels

#A, A_labels, B, B_labels = gen_lin_separable_data()
A, A_labels, B, B_labels = gen_lin_separable_overlap_data()
#A, A_labels, B, B_labels = gen_non_lin_separable_data()

# Minh hoa du lieu
with PdfPages('data.pdf') as pdf:
    plt.plot(A[:,0], A[:,1], 'bs', markersize = 8, alpha = 1)
    plt.plot(B[:,0], B[:,1], 'ro', markersize = 8, alpha = 1)
    plt.axis('tight')
    plt.title('Dữ liệu gần tách được tuyến tính', fontsize = 20)
    
    #hide ticks
    #cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_ticks([])
    #cur_axes.axes.get_yaxis().set_ticks([])

    plt.xlabel('$A^1$', fontsize = 20)
    plt.ylabel('$A^2$', fontsize = 20)
    pdf.savefig()
    plt.show()

#Solving by dual problems of soft SVM
from cvxopt import matrix, solvers

def SVM_solver(A, B, y1, y2, c):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2    
    C = np.vstack((A,B))
    y = np.hstack((y1,y2))
    #build K
    V = np.concatenate((A, -B), axis = 0)
    K = matrix(V.dot(V.T))    
    e = matrix(-np.ones((m, 1)))
    
    #build A, b, G, h
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.vstack((np.zeros((m, 1)), c*np.ones((m, 1)))))
    Y = matrix(y.reshape((-1, m)))
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
    yM = y[M]
    CM = C[M,:]
    w_dual = VS.T.dot(alS).reshape(-1,1)
    b_dual = np.mean(yM - CM.dot(w_dual))
    #print(w_dual, b_dual)
    return w_dual, b_dual, y

import time
start_time = time.time()
w_dual, b_dual, y = SVM_solver(A, B, A_labels, B_labels, c=1)
end_time = time.time()
print('total run-time of SVM: %f ms' %((end_time - start_time)*1000))

def pred_class(x,w,b):
    return np.sign(x.T.dot(w) + b)

def pred_SVM(X,w,b):
    y_pred = []
    for i in range(X.shape[0]):
        yi = pred_class(X[i,:], w_dual,b_dual)
        y_pred.append(yi)
        
    y_pred_SVM = np.array(y_pred).reshape((X.shape[0],))
    return y_pred_SVM

C = np.vstack((A,B))
y_pred_SVM = pred_SVM(C, w_dual, b_dual)
print('accuracy of SVM: ', 100*np.mean(y_pred_SVM == y))

def myplot(A, B, w, b, filename, tit):
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots()
        w0 = w[0]
        w1 = w[1]
        x1 = np.arange(-10,10,0.1)
        y1 = -w0/w1*x1 - b/w1
        y2 = -w0/w1*x1 - (b-1)/w1
        y3 = -w0/w1*x1 - (b+1)/w1
        plt.plot(x1, y1, 'k', linewidth = 3)
        plt.plot(x1, y2, 'k')
        plt.plot(x1, y3, 'k')
        
        #plt.axis('tight')
        plt.axis('equal')
        plt.xlim([-3,5])
        plt.ylim([-3,5])

        # hide ticks
        #cur_axes = plt.gca()
        #cur_axes.axes.get_xaxis().set_ticks([])
        #cur_axes.axes.get_yaxis().set_ticks([])

        # fill two regions
        plt.fill_between(x1,y1,y2, color = 'blue', alpha = '0.1')
        plt.fill_between(x1,y1,y3, color = 'red', alpha = '0.1')

        plt.xlabel('$A^1$', fontsize = 20)
        plt.ylabel('$A^2$', fontsize = 20)
        plt.title('Nghiệm tìm bởi ' + tit, fontsize = 20)
        plt.plot(A[:,0],A[:,1], 'bs', markersize = 8, alpha = .8)
        plt.plot(B[:,0],B[:,1], 'ro', markersize = 8, alpha = .8)
        pdf.savefig()
        plt.show()

myplot(A, B, w_dual, b_dual, 'svm_dual.pdf', 'SVM')

### Solving by dual problem of twinSVM

def TSVM_solver(A,B,y1,y2,c1,c2):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2
    AB = np.vstack((A,B))
    y = np.hstack((y1,y2))
    
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

start_time = time.time()
w1_tsvm, b1_tsvm, w2_tsvm, b2_tsvm = TSVM_solver(A,B,A_labels,B_labels,c1=1.5,c2=1.5)
end_time = time.time()
print('total run-time of TSVM: %f ms' %((end_time - start_time)*1000))

def myplot12(A, B, w1, b1, w2, b2, filename, tit):
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots()
        w10 = w1[0]
        w11 = w1[1]
        x11 = np.arange(-10,10,0.1)
        y11 = (-w10/w11)*x11 - b1/w11
        y12 = (-w10/w11)*x11 - (b1+1)/w11
        w20 = w2[0]
        w21 = w2[1]
        x21 = np.arange(-10,10,0.1)
        y21 = (-w20/w21)*x21 - b2/w21
        y22 = (-w20/w21)*x21 - (b2-1)/w21
        
        plt.plot(x11, y11, 'k', linewidth = 3)
        plt.plot(x11, y12, 'k')
        plt.plot(x21, y21, 'k', linewidth = 3)
        plt.plot(x21, y22, 'k')

        plt.axis('equal')
        plt.xlim([-3,5])
        plt.ylim([-3,5])

        # hide ticks
        #cur_axes = plt.gca()
        #cur_axes.axes.get_xaxis().set_ticks([])
        #cur_axes.axes.get_yaxis().set_ticks([])

        # fill two regions
        #plt.fill_between(x11,y11,y12, color = 'blue', alpha = '0.1')
        #plt.fill_between(x21,y21,y22, color = 'red', alpha = '0.1')

        plt.xlabel('$A^1$', fontsize = 20)
        plt.ylabel('$A^2$', fontsize = 20)
        plt.title('Nghiệm tìm bởi ' + tit, fontsize = 20)
        plt.plot(A[:,0],A[:,1], 'bs', markersize = 8, alpha = .8)
        plt.plot(B[:,0],B[:,1], 'ro', markersize = 8, alpha = .8)
        pdf.savefig()
        plt.show()

myplot12(A, B, w1_tsvm, b1_tsvm,w2_tsvm,b2_tsvm, 'tsvm_dual.pdf', 'TSVM')

def predict_class(x,w1,b1,w2,b2):
    y1 = np.abs(x.T.dot(w1) + b1)
    y2 = np.abs(x.T.dot(w2) + b2)
    y_pred_x = None
    if y1 < y2:
        y_pred_x = 1
    else:
        y_pred_x = -1
        
    return y_pred_x

def predict_TSVM(X,w1,b1,w2,b2):
    y_pred = []
    for i in range(X.shape[0]):
        y_pred_i = predict_class(X[i,:],w1,b1,w2,b2)
        y_pred.append(y_pred_i)
    
    y_pred_TSVM = np.array(y_pred).reshape((X.shape[0],))
    return y_pred_TSVM

y_pred_TSVM = predict_TSVM(C, w1_tsvm,b1_tsvm,w2_tsvm,b2_tsvm)
accuracy_train = np.mean(y == y_pred_TSVM)

print('accuracy of SVM: ', 100*np.mean(y == y_pred_SVM))
print('accuracy of TSVM: ', 100*accuracy_train)

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

def KTSVM_solver(A,B,y1,y2,c1,c2):
    m1 = A.shape[0]
    m2 = B.shape[0]
    m = m1 + m2
    AB = np.vstack((A,B))
    C = AB
    y = np.hstack((y1,y2))
    ## DKTWSVM1
    # Build K1, q1, G1, h1
    e1 = np.ones((m1,1))
    e2 = np.ones((m2,1))
    S = np.hstack((Kernel(A,C.T,1),e1))
    I = np.eye(m+1)
    R = np.hstack((Kernel(B,C.T,1),e2))
    K1 = matrix(R.dot(np.linalg.inv((S.T).dot(S)+0.001*I)).dot(R.T))
    q1 = matrix(-e2)
    G1 = matrix(np.vstack((-np.eye(m2),np.eye(m2))))
    h1 = matrix(np.vstack((np.zeros((m2,1)),c1*np.ones((m2,1)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(K1,q1, G1, h1)

    al = np.array(sol['x'])
    #print('alpha = \n', al.T)

    ## DKTWSVM2
    # build K2, q2, G2, h2
    K2 = matrix(S.dot(np.linalg.inv((R.T).dot(R)+0.001*I)).dot(S.T))
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

u1_Ktsvm, b1_Ktsvm, u2_Ktsvm, b2_Ktsvm = KTSVM_solver(A,B,A_labels,B_labels,c1=.1,c2=.1)

def predict_KTSVM_1(x,C,u1,b1,u2,b2):
    y_i = None
    y1 = np.abs((Kernel_1(x, C.T, 1)).dot(u1)+b1)
    y2 = np.abs((Kernel_1(x, C.T, 1)).dot(u2)+b2)
    if y1[0] < y2[0]:
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
        if y1[0] < y2[0]:
            y_i = 1
        else:
            y_i = -1
    
        y_pred.append(y_i)
    return y_pred
    
y_train_pred_KTSVM = predict_KTSVM(C,C,u1_Ktsvm,b1_Ktsvm,u2_Ktsvm,b2_Ktsvm)
accuracy_train = np.mean(y == y_train_pred_KTSVM)
print('accuracy of KTSVM: ', 100*accuracy_train)

