#coding = 'utf-8'
"""training model from data set and model decision"""
__author__ = "jiangweiwu"
__date__   = "2016.3.16"

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
TrainDataset = 'Training Set/'
TestDataset = 'Test Set/'
train_set = pd.read_csv(FilePath + TrainDataset  + 'PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gb18030')
test_set = pd.read_csv(FilePath + TestDataset  + 'PPD_Master_GBK_2_Test_Set.csv',encoding='gb18030')

train_set.dropna(axis=1,how='all')  # if all values are NA, drop that label
train_set_col = train_set.columns
index = 0
for temp in train_set.dtypes.values:
    if temp == 'object':
        train_set = train_set.drop(train_set_col[index],axis=1)
        test_set = test_set.drop(train_set_col[index],axis=1)
    index += 1

    
#print test.dtypes
Y_Label=train_set['target']
X_Data = train_set.drop('target',axis=1)
X_Data_columns = X_Data.columns
###### fix 'NaN' type with mean value
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_Data)
New_X_Data = imp.transform(X_Data)
New_Test_Data = imp.transform(test_set)
print New_X_Data.shape,New_Test_Data.shape
######
##cost_time = time()
###forest = ensemble.GradientBoostingClassifier(learning_rate=0.001,n_estimators=200,max_depth=6)
##forest = ensemble.GradientBoostingClassifier(learning_rate=0.05,n_estimators=200,max_depth=3) #test case
##forest.fit(New_X_Data, Y_Label)
##importances = forest.feature_importances_
##indices = np.argsort(importances)[::-1]
##cost_time = time() - cost_time
##print cost_time


#####################################test and submit 2

skf = cross_validation.StratifiedKFold(Y_Label.values,50)

for train_index, test_index in skf:
    X_Data_Train = New_X_Data[train_index]
    X_Data_Test = New_X_Data[test_index]
    Y_Label_Train = Y_Label[train_index]
    Y_Label_Test = Y_Label[test_index]
    #print test_index,len(test_index)
    cost_time = time()
    grd = ensemble.GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.005) #test case learning_rate=0.01
    #grd = ensemble.RandomForestClassifier(max_depth=8, n_estimators=200)
    grd.fit(X_Data_Train, Y_Label_Train)
    cost_time = time() - cost_time
    print cost_time
    cost_time = time()
    grd_enc = preprocessing.OneHotEncoder()
    #print  grd.apply(X_Data_Train)[1, :, 0]
    temp_X_Train = grd.apply(X_Data_Train)[:, :, 0]
    temp_X_Test = grd.apply(X_Data_Test)[:, :, 0]

    grd_enc.fit(temp_X_Train)
    #grd_enc.fit(grd.apply(X_Data_Train))
    cost_time = time() - cost_time
    print cost_time
    New_X_Data_Train = grd_enc.transform(temp_X_Train)
    New_Test_Data = grd_enc.transform(temp_X_Test)
    #New_X_Data_Train = grd_enc.transform(grd.apply(X_Data_Train))
    #New_Test_Data = grd_enc.transform(grd.apply(X_Data_Test))

    print temp_X_Train.shape,New_X_Data_Train.shape
    for i,C in enumerate((0.06,0.05)):
        grd_lm = linear_model.LogisticRegression(C=C,penalty='l1',tol=0.005,solver='liblinear',max_iter=200)
        grd_lm.fit(New_X_Data_Train, Y_Label_Train)

        y_pred_grd_lm = grd_lm.predict_proba(New_Test_Data)[:, 1]
        res_l2_LR = metrics.roc_auc_score(Y_Label_Test, y_pred_grd_lm)
        print i,res_l2_LR


    y_pred_grd = grd.predict_proba(X_Data_Test)[:, 1]
    res_l2_LR = metrics.roc_auc_score(Y_Label_Test, y_pred_grd)
    print res_l2_LR
    

##grd = ensemble.GradientBoostingClassifier(n_estimators=100,max_depth=6) 
##grd.fit(New_X_Data, Y_Label)
##temp_X_Train = grd.apply(New_X_Data)[:, :, 0]
##temp_X_Test = grd.apply(New_Test_Data)[:, :, 0]
##grd_enc = preprocessing.OneHotEncoder()
##grd_enc.fit(temp_X_Train)
##New_X_Data_Train = grd_enc.transform(temp_X_Train)
##New_Test_Data = grd_enc.transform(temp_X_Test)
##grd_lm = linear_model.LogisticRegression(C=0.06,penalty='l1',tol=0.005,solver='liblinear',max_iter=400)
##grd_lm.fit(New_X_Data_Train, Y_Label)
##New_Test_Data_res = grd_lm.predict_proba(New_Test_Data)
##Total_res = np.column_stack((New_Test_Data[:,0].astype(int),New_Test_Data_res[:,1]))
##np.savetxt('2.csv',Total_res,fmt='%.6f')


######################################################









##temp_index= []
##New_X_Data_ori = New_X_Data.copy()
##New_Test_Data_ori = New_Test_Data.copy()


###############################################################cross validation
##for f in range(1,5):
##    index = indices[New_X_Data.shape[1]-f]
##    temp_index.append(index)
##    
##for f in range(5,18):
##    index = indices[New_X_Data.shape[1]-f]
##    temp_index.append(index)
##    New_X_Data = sci.delete(New_X_Data_ori,temp_index,1)
##    New_Test_Data = sci.delete(New_Test_Data_ori,temp_index,1)
##    print  New_X_Data.shape,New_Test_Data.shape,temp_index   
##
##
##
##    skf = cross_validation.StratifiedKFold(Y_Label.values,10)
##    res_l1_LR_list_5 = []
##    res_l1_LR_list_1 = []
##    res_l1_LR_list_05 = []
##    res_l2_LR_list_5 = []
##    res_l2_LR_list_1 = []
##    res_l2_LR_list_05 = []
##    for train_index, test_index in skf:
##        #print len(train_index),len(test_index)
##        #print("TRAIN:", train_index, "TEST:", test_index)
##    
##        X_Data_Train = New_X_Data[train_index]
##        X_Data_Test = New_X_Data[test_index]
##        Y_Label_Train = Y_Label[train_index]
##        Y_Label_Test = Y_Label[test_index]
##
##        cost_time = time()
##        for i,C in enumerate((0.1,0.05,0.01)):
##            
##            clf_l1_LR = linear_model.LogisticRegression(C=C,penalty='l1',tol=0.001,solver='liblinear',max_iter=200)
##            clf_l1_LR.fit(X_Data_Train,Y_Label_Train)
##
##            test_l1_LR_y = clf_l1_LR.predict_proba(X_Data_Test)
##
##            
##            test_l1_LR_y_1 = [pr[1] for pr in test_l1_LR_y]
##            #class_weight='balanced'
####            clf_l2_LR = linear_model.LogisticRegression(C=C,penalty='l2',tol=0.01,solver='lbfgs',max_iter=100)
####            clf_l2_LR.fit(X_Data_Train,Y_Label_Train)
####            test_l2_LR_y = clf_l2_LR.predict_proba(X_Data_Test)
####            test_l2_LR_y_1 = [pr[1] for pr in test_l2_LR_y]
##       
##            res_l1_LR = metrics.roc_auc_score(Y_Label_Test, test_l1_LR_y_1)
####            res_l2_LR = metrics.roc_auc_score(Y_Label_Test, test_l2_LR_y_1)
##
##
##            if i==0:
##                res_l1_LR_list_5.append(res_l1_LR)
####                res_l2_LR_list_5.append(res_l2_LR)
##            if i==1:
##                res_l1_LR_list_1.append(res_l1_LR)
####                res_l2_LR_list_1.append(res_l2_LR)
##            if i==2:
##                res_l1_LR_list_05.append(res_l1_LR)
####                res_l2_LR_list_05.append(res_l2_LR)
##
##
##    cost_time = time() - cost_time
##    print cost_time
##    print New_X_Data.shape[1],np.average(res_l1_LR_list_5),np.average(res_l1_LR_list_1),np.average(res_l1_LR_list_05)
####    print New_X_Data.shape[1],np.average(res_l2_LR_list_5),np.average(res_l2_LR_list_1),np.average(res_l2_LR_list_05)

##############################evaluation
##for f in range(1,13):
##    index = indices[New_X_Data.shape[1]-f]
##    temp_index.append(index)
##
##New_X_Data = sci.delete(New_X_Data_ori,temp_index,1)
##New_Test_Data = sci.delete(New_Test_Data_ori,temp_index,1)
##print  New_X_Data.shape,New_Test_Data.shape,temp_index
##    
##clf_l1_LR = linear_model.LogisticRegression(C=0.1,penalty='l1',tol=0.001,solver='liblinear',max_iter=500)
##clf_l1_LR.fit(New_X_Data,Y_Label)
##New_Test_Data_res = clf_l1_LR.predict_proba(New_Test_Data)
##Total_res = np.column_stack((New_Test_Data_ori[:,0].astype(int),New_Test_Data_res[:,1]))
##np.savetxt('1.csv',Total_res,fmt='%.4f')






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






##for f in range(1,New_X_Data.shape[1]/5):
##    index = indices[New_X_Data.shape[1]-f]
##    temp_index.append(index)
##    New_X_Data = sci.delete(New_X_Data_ori,temp_index,1)
##    print  New_X_Data.shape[1],temp_index


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
##        for i,C in enumerate((5,1,0.5)):
##            kernel_svm_time = time()
##            ###########0.633937037158 
####            clf_l2_SVM = svm.LinearSVC(C=C,penalty='l2',tol=0.001,max_iter=200,dual=False)
####            clf_l2_SVM.fit(X_Data_Train,Y_Label_Train)
####            test_l2_SVM_y = clf_l2_SVM.decision_function(X_Data_Test)
####            test_l2_SVM_y_1 = (test_l2_SVM_y - test_l2_SVM_y.min())/(test_l2_SVM_y.max() - test_l2_SVM_y.min())
##            ###########
##            #clf_l2_SVM = svm.SVC(C=C,kernel='rbf',gamma=0.1,probability=True)
##            clf_l2_SVM = svm.SVC(C=C,kernel='poly',degree=3,probability=True)
##            clf_l2_SVM.fit(X_Data_Train,Y_Label_Train)
##            test_l2_SVM_y = clf_l2_SVM.predict_proba(X_Data_Test)
##            test_l2_SVM_y_1 = [pr[1] for pr in test_l2_SVM_y]
##            kernel_svm_time = time() - kernel_svm_time
##
##            res_l2_SVM = metrics.roc_auc_score(Y_Label_Test, test_l2_SVM_y_1)
##            print kernel_svm_time,i,res_l2_SVM
#############################################  0.596526744699

##        clf_l2_GBD = ensemble.GradientBoostingClassifier(learning_rate=0.0001) #loss='log',modified_huber ,n_estimators=1000,max_depth=5
##        clf_l2_GBD.fit(X_Data_Train,Y_Label_Train)
##        ##clf_l2_SGD.partial_fit(X_Data_Train,Y_Label_Train)
##        test_l2_GBD_y = clf_l2_GBD.predict_proba(X_Data_Test)
##        test_l2_GBD_y_1 = [pr[1] for pr in test_l2_GBD_y]
##        res_l2_GBD = metrics.roc_auc_score(Y_Label_Test, test_l2_GBD_y_1)
##        print res_l2_GBD

##############################################
