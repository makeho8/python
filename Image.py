import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
col_names = ["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", 
                   "short-line-density-2", "vedge-mean", "vegde-sd", 
                   "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", 
                   "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", 
                   "value-mean", "saturatoin-mean", "hue-mean"]
df_image = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test', names = col_names)
df_image1 = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data', names = col_names)
#print('class labels', np.unique(df_image['region-centroid-col']))

X = np.vstack((df_image.iloc[3:,1:].values,df_image1.iloc[3:,1:].values)) 
y = np.hstack((df_image.iloc[3:,0].values, df_image1.iloc[3:,0]))
print(np.unique(y))
A1 = X[np.where(y == 'BRICKFACE')]
A2 = X[np.where(y == 'CEMENT')]
A3 = X[np.where(y == 'FOLIAGE')]
A4 = X[np.where(y == 'GRASS')]
A1234 = np.vstack((A1,A2,A3,A4))
B1 = X[np.where(y == 'PATH')]
B2 = X[np.where(y == 'SKY')]
B3 = X[np.where(y == 'WINDOW')]
B123 = np.vstack((B1,B2,B3))
A = A1234.astype(np.float)
B = B123.astype(np.float)
y_A = np.ones(len(A))
y_B = -np.ones(len(B))
AB = np.vstack((A,B))
y_AB = np.hstack((y_A, y_B))
AB_train, AB_test, y_train, y_test = train_test_split(AB, y_AB, test_size = 0.1, random_state =42)



from sklearn.preprocessing import StandardScaler 
stdsc = StandardScaler()
AB_train_std = stdsc.fit_transform(AB_train)
AB_test_std = stdsc.transform(AB_test)

from sklearn.preprocessing import MinMaxScaler 
mms = MinMaxScaler()
AB_train_mms = mms.fit_transform(AB_train)
AB_test_mms = mms.transform(AB_test)

###WS_SVM best params of WS_SVM {'c1': 0.0001, 'c2': 0.1, 'c3': 100.0}
from WS_SVM_class import WS_SVM
start_time = time.time()
clf2 = WS_SVM(c1 = 0.0001, c2 = 0.1, c3 = 100)
clf2.fit(AB_train_mms, y_train)
end_time = time.time()
print('Total runtime of WS_SVM: %s' %((end_time - start_time)))
y_pred_WS_SVM = clf2.predict(AB_test_mms)
print('accuracy of WS_SVM: %s' %(100*np.mean(y_pred_WS_SVM==y_test)), clf2.score(AB_test_mms, y_test))
###Cross validation score of WS_SVM
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = clf2, X = AB_train_mms, y = y_train, cv = 10, n_jobs =1)
#print('CV accuracy scores of WS_SVM: %s' %scores)
print('CV accuracy of WS_SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

### S_TWSVM best params of S_TWSVM {'c1': 1.0, 'c2': 0.0001, 'c3': 100.0}
from S_TWSVM_class import S_TWSVM
start_time = time.time()
clf3 = S_TWSVM(c1 = 1, c2 = 0.0001, c3 = 100)
clf3.fit(AB_train_mms, y_train)
end_time = time.time()
print('Total runtime of S_TWSVM: %s' %((end_time - start_time)))
y_S_TWSVM = clf3.predict(AB_test_mms)
print('Accuracy of S_TWSVM %.3f' %(100*np.mean(y_S_TWSVM == y_test)))
###Cross validation score of S_TWSVM
#from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = clf3, X = AB_train_mms, y = y_train, cv = 10, n_jobs =1)
#print('CV accuracy scores of S_TWSVM: %s' %scores)
print('CV accuracy of S_TWSVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# TWSVM best params of TWSVM {'c': 100.0, 'c_': 10.0}
from TWSVM_class import TWSVM
start_time = time.time()
clf4 = TWSVM(c = 100, c_ = 10)
clf4.fit(AB_train_mms, y_train)
end_time = time.time()
print('total run time of TWSVM: %f ' %((end_time - start_time)))
y_TWSVM = clf4.predict(AB_test_mms)
print('Accuracy of TWSVM %.3f ' %(100*np.mean(y_TWSVM == y_test)))
###Cross validation score of TWSVM
#from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = clf4, X = AB_train_mms, y = y_train, cv = 10, n_jobs =1)
#print('CV accuracy scores of TWSVM: %s' %scores)
print('CV accuracy of TWSVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# SVM best params of SVM 
#from SVM_class import SVM
#start_time = time.time()
#clf5 = SVM(c = 1)
#clf5.fit(AB_train, y_train)
#end_time = time.time()
#print('Total runtime of SVM: %s' %((end_time - start_time)*1000))
#y_svm = clf5.predict(AB_test)
#print('Accuracy of svm %.3f ' % (100*np.mean(y_svm == y_test)))
###Cross validation score of SVM
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(estimator = clf5, X = AB_train, y = y_train, cv = 10, n_jobs =1)
#print('CV accuracy scores of SVM: %s' %scores)
#print('CV accuracy of SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



### Tuning hyperparameters via grid search (WS_SVM) 
from sklearn.model_selection import GridSearchCV
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid2 = [{'c1': param_range, 'c2': param_range, 'c3': param_range}]
#param_grid2 = [{'gamma': param_range}]
gs2 = GridSearchCV(estimator = clf2, param_grid = param_grid2, scoring = 'accuracy', cv = 10)
gs2 = gs2.fit(AB_train_mms, y_train)
#print('best score of WS_SVM: ', gs2.best_score_) 
print('best params of WS_SVM', gs2.best_params_)
clf2 = gs2.best_estimator_
clf2.fit(AB_train_mms, y_train)
print('Test accuracy of WS_SVM: %.3f' % clf2.score(AB_test_mms, y_test))
scores = cross_val_score(estimator = clf2, X = AB_train_mms, y = y_train, cv = 10)
#print('CV accuracy scores of WS_SVM: %s' %scores)
print('CV accuracy of WS_SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

### Tuning hyperparameters via grid search (S_TWSVM) 
#from sklearn.model_selection import GridSearchCV
#param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid3 = [{'c1': param_range, 'c2': param_range, 'c3': param_range}]
#param_grid3 = [{'gamma': param_range}]
gs3 = GridSearchCV(estimator = clf3, param_grid = param_grid3, scoring = 'accuracy', cv = 10)
gs3 = gs3.fit(AB_train_mms, y_train)
#print('best score of S_TWSVM: ', gs3.best_score_) 
print('best params of S_TWSVM', gs3.best_params_) 
clf3 = gs3.best_estimator_
clf3.fit(AB_train_mms, y_train)
print('Test accuracy of S-TWSVM: %.3f' % clf3.score(AB_test_mms, y_test))
scores = cross_val_score(estimator = clf3, X = AB_train_mms, y = y_train, cv = 10)
print('CV accuracy of S-TWSVM: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

### Tuning hyperparameters via grid search (TWSVM) 
#from sklearn.model_selection import GridSearchCV
#param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid4 = [{'c': param_range, 'c_': param_range}]
#param_grid4 = [{'gamma': param_range}]
gs4 = GridSearchCV(estimator = clf4, param_grid = param_grid4, scoring = 'accuracy', cv = 10)
gs4 = gs4.fit(AB_train_mms, y_train)
#print('best score of TWSVM: ', gs4.best_score_)
print('best params of TWSVM', gs4.best_params_)
clf4 = gs4.best_estimator_
clf4.fit(AB_train_mms, y_train)
print('Test accuracy of TWSVM: %.3f' % clf4.score(AB_test_mms, y_test))
scores = cross_val_score(estimator = clf4, X = AB_train_mms, y = y_train, cv = 10)
print('CV accuracy of S-TWSVM: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))