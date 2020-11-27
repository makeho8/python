import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

df1_heart = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data', header = None)
df1_heart.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
               'oldpeak','slope','ca','thal','predicted']

df2_heart = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data', header = None)
df2_heart.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
               'oldpeak','slope','ca','thal','predicted'] 

df3_heart = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header = None)
df3_heart.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
               'oldpeak','slope','ca','thal','predicted'] 

df4_heart = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data', header = None)
df4_heart.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
               'oldpeak','slope','ca','thal','predicted']

from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values = '?', strategy = 'most_frequent')

#imr1 = imr.fit(df1_heart)
#A1 = imr1.transform(df1_heart.values)
#imr2 = imr.fit(df2_heart)
#A2 = imr2.transform(df2_heart.values)
imr3 = imr.fit(df3_heart)
A3 = imr3.transform(df3_heart.values)
#imr4 = imr.fit(df4_heart)
#A4 = imr4.transform(df4_heart.values)

#X0 = np.concatenate((A1,A2,A3,A4))
X = A3[:,:13]
y = A3[:,13]

#A1 = X[np.where(y == 0)]
#A2 = X[np.where(y == 4)]
B1 = X[np.where(y == 1)]
B2 = X[np.where(y == 2)]
B3 = X[np.where(y == 3)]
B4 = X[np.where(y == 4)]
A = X[np.where(y == 0)]
B = np.vstack((B1, B2, B3, B4))
y_A = np.ones(len(A))
y_B = -np.ones(len(B))
AB = np.vstack((A,B))
y_AB = np.hstack((y_A, y_B))
AB_train, AB_test, y_train, y_test = train_test_split(AB, y_AB, test_size = 0.1, random_state = 42)

#scale the dataset
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
AB_train_std = stdsc.fit_transform(AB_train)
AB_test_std = stdsc.transform(AB_test)

from sklearn.preprocessing import MinMaxScaler 
mms = MinMaxScaler()
AB_train_mms = mms.fit_transform(AB_train)
AB_test_mms = mms.transform(AB_test)

###WS_SVM best params of WS_SVM {'c1': 1.0, 'c2': 10.0, 'c3': 10.0}
from WS_SVM_class import WS_SVM
start_time = time.time()
clf2 = WS_SVM(c1 = 1, c2 = 10, c3 = 10)
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

# TWSVM best params of TWSVM {'c': 0.1, 'c_': 0.1}
from TWSVM_class import TWSVM
start_time = time.time()
clf4 = TWSVM(c = 0.1, c_ = 0.1)
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
print('CV accuracy of TWSVM: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))