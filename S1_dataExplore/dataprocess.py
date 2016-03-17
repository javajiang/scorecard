#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import random
import numpy as np
import pandas as pd
import scipy as sci
from datetime import datetime
from sklearn import *
import matplotlib.pyplot as plt
from time import time

#from sklearn.ensemble import RandomForestRegressor
#from sklearn.pipeline import Pipeline

FilePath = '../data/'
Dataset = 'Training Set/'
##MasterFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
##fhadle = open(MasterFile,'r')
##data = fhadle.readlines()
##fhadle.close()
##data = map(lambda x:x.decode('gbk').strip().split(',')+['train_master'],data)
##MasterData = pd.DataFrame(data[1:],columns=data[0][:-1]+['flag'])

'''Summury = pd.read_csv(FilePath + 'PPD_summary.csv')
print len(Summury)
se = Summury.loc[:,'Columns':'type']
for row in se.itertuples():
    print row[1],row[2]
    print MasterData[row[1]]
    if row[1] == 'ListingInfo':
        pd.to_datetime(MasterData[row[1]])
    elif row[1] == 'UserInfo_1' or row[1] == 'UserInfo_10':
        pass
    else:
        MasterData[row[1]] = MasterData[row[1]].astype(row[2])
print MasterData.dtypes
#Describe.to_csv('../Output/foo.csv', encoding='gbk')
#raw_input()'''


test = pd.read_csv(FilePath + Dataset  + 'PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gb18030')
test.dropna(axis=1,how='all')
ab = test.columns
index = 0
for temp in test.dtypes.values:
    if temp == 'object':
        #print ab[index]
        test = test.drop(ab[index],axis=1)
    index += 1
#print test.dtypes
Y_Label=test['target']
X_Data = test.drop('target',axis=1)
X_Data_columns = X_Data.columns
###### fix 'NaN' type with mean value
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_Data)
New_X_Data = imp.transform(X_Data)
######


# Build a forest and compute the feature importances
##forest = ensemble.ExtraTreesClassifier(n_estimators=250,random_state=0,criterion='entropy')#max_features,criterion=gini/entropy
##forest.fit(New_X_Data, Y_Label)
##importances = forest.feature_importances_
##indices = np.argsort(importances)[::-1]


##print("Feature ranking:")
##std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
##for f in range(New_X_Data.shape[1]):
##    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
##plt.figure()
##plt.title("Feature importances")
##plt.bar(range(New_X_Data.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
##plt.xticks(range(New_X_Data.shape[1]), indices)
##plt.xlim([-1, New_X_Data.shape[1]])
##plt.show()

##forest = ensemble.GradientBoostingClassifier(learning_rate=0.005,n_estimators=200,max_depth=5)
##forest.fit(New_X_Data, Y_Label)
##importances = forest.feature_importances_
##indices = np.argsort(importances)[::-1]


temp_index= []
New_X_Data_ori = New_X_Data

##for f in range(1,New_X_Data.shape[1]/5):
##    index = indices[New_X_Data.shape[1]-f]
##    temp_index.append(index)
##    New_X_Data = sci.delete(New_X_Data_ori,temp_index,1)
##    print  New_X_Data.shape[1],temp_index

for f in range(0,1):
    
    #raw_input()

##    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
##                     'C': [1, 10, 100, 1000]},
##                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
##
##    scores = ['auc', 'auto']
##    for score in scores:
##        clf = grid_search.GridSearchCV(svm.SVC(), tuned_parameters, cv=5,scoring='%s_weighted' % score)
##        clf.fit(New_X_Data, Y_Label)
##        print(clf.best_params_)  
##        for params, mean_score, scores in clf.grid_scores_:
##            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
##    raw_input()

    skf = cross_validation.StratifiedKFold(Y_Label.values,5)
    res_l1_LR_list_5 = []
    res_l1_LR_list_1 = []
    res_l1_LR_list_05 = []
    res_l2_LR_list_5 = []
    res_l2_LR_list_1 = []
    res_l2_LR_list_05 = []
    for train_index, test_index in skf:
        #print len(train_index),len(test_index)
        #print("TRAIN:", train_index, "TEST:", test_index)
    
        X_Data_Train = New_X_Data[train_index]
        X_Data_Test = New_X_Data[test_index]
        Y_Label_Train = Y_Label[train_index]
        Y_Label_Test = Y_Label[test_index]
    
    
##        for i,C in enumerate((5,1,0.5)):
##    
##            clf_l1_LR = linear_model.LogisticRegression(C=C,penalty='l1',tol=0.01,solver='liblinear',max_iter=200)
##            clf_l1_LR.fit(X_Data_Train,Y_Label_Train)
##            test_l1_LR_y = clf_l1_LR.predict_proba(X_Data_Test)
##            test_l1_LR_y_1 = [pr[1] for pr in test_l1_LR_y]
##            #class_weight='balanced'
##            clf_l2_LR = linear_model.LogisticRegression(C=C,penalty='l2',tol=0.01,solver='lbfgs',max_iter=1000)
##            clf_l2_LR.fit(X_Data_Train,Y_Label_Train)
##            test_l2_LR_y = clf_l2_LR.predict_proba(X_Data_Test)
##            test_l2_LR_y_1 = [pr[1] for pr in test_l2_LR_y]
##       
##            res_l1_LR = metrics.roc_auc_score(Y_Label_Test, test_l1_LR_y_1)
##            res_l2_LR = metrics.roc_auc_score(Y_Label_Test, test_l2_LR_y_1)
##    ##        res_l1_SVM = metrics.roc_auc_score(Y_Label_Test, test_l1_SVM_y_1)
##    ##        res_l2_SVM = metrics.roc_auc_score(Y_Label_Test, test_l2_SVM_y_1)
##
##            if i==0:
##                res_l1_LR_list_5.append(res_l1_LR)
##                res_l2_LR_list_5.append(res_l2_LR)
##            if i==1:
##                res_l1_LR_list_1.append(res_l1_LR)
##                res_l2_LR_list_1.append(res_l2_LR)
##            if i==2:
##                res_l1_LR_list_05.append(res_l1_LR)
##                res_l2_LR_list_05.append(res_l2_LR)
##
##    print New_X_Data.shape[1],np.average(res_l1_LR_list_5),np.average(res_l1_LR_list_1),np.average(res_l1_LR_list_05)
##    print New_X_Data.shape[1],np.average(res_l2_LR_list_5),np.average(res_l2_LR_list_1),np.average(res_l2_LR_list_05)
############################################################  0.566944712364
 
##
##        clf_l2_SGD = linear_model.SGDClassifier(penalty='l1',loss='log',alpha=0.00001,n_iter=5000) #loss='log',modified_huber
##        clf_l2_SGD.fit(X_Data_Train,Y_Label_Train)
##        ##clf_l2_SGD.partial_fit(X_Data_Train,Y_Label_Train)
##        test_l2_SGD_y = clf_l2_SGD.predict_proba(X_Data_Test)
##        test_l2_SGD_y_1 = [pr[1] for pr in test_l2_SGD_y]
##        res_l2_SGD = metrics.roc_auc_score(Y_Label_Test, test_l2_SGD_y_1)
##        print res_l2_SGD
           

########################################################### 
        for i,C in enumerate((5,1,0.5)):
            kernel_svm_time = time()
            ###########0.633937037158 
##            clf_l2_SVM = svm.LinearSVC(C=C,penalty='l2',tol=0.001,max_iter=200,dual=False)
##            clf_l2_SVM.fit(X_Data_Train,Y_Label_Train)
##            test_l2_SVM_y = clf_l2_SVM.decision_function(X_Data_Test)
##            test_l2_SVM_y_1 = (test_l2_SVM_y - test_l2_SVM_y.min())/(test_l2_SVM_y.max() - test_l2_SVM_y.min())
            ###########
            #clf_l2_SVM = svm.SVC(C=C,kernel='rbf',gamma=0.1,probability=True)
            clf_l2_SVM = svm.SVC(C=C,kernel='poly',degree=3,probability=True)
            clf_l2_SVM.fit(X_Data_Train,Y_Label_Train)
            test_l2_SVM_y = clf_l2_SVM.predict_proba(X_Data_Test)
            test_l2_SVM_y_1 = [pr[1] for pr in test_l2_SVM_y]
            kernel_svm_time = time() - kernel_svm_time

            res_l2_SVM = metrics.roc_auc_score(Y_Label_Test, test_l2_SVM_y_1)
            print kernel_svm_time,i,res_l2_SVM
#############################################  0.596526744699

##        clf_l2_GBD = ensemble.GradientBoostingClassifier(learning_rate=0.0001) #loss='log',modified_huber ,n_estimators=1000,max_depth=5
##        clf_l2_GBD.fit(X_Data_Train,Y_Label_Train)
##        ##clf_l2_SGD.partial_fit(X_Data_Train,Y_Label_Train)
##        test_l2_GBD_y = clf_l2_GBD.predict_proba(X_Data_Test)
##        test_l2_GBD_y_1 = [pr[1] for pr in test_l2_GBD_y]
##        res_l2_GBD = metrics.roc_auc_score(Y_Label_Test, test_l2_GBD_y_1)
##        print res_l2_GBD

##############################################
