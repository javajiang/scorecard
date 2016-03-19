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
cost_time = time()
#forest = ensemble.GradientBoostingClassifier(learning_rate=0.001,n_estimators=200,max_depth=6)
forest = ensemble.GradientBoostingClassifier(learning_rate=0.05,n_estimators=200,max_depth=3) #test case
forest.fit(New_X_Data, Y_Label)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
cost_time = time() - cost_time
print cost_time

temp_index= []
New_X_Data_ori = New_X_Data.copy()
New_Test_Data_ori = New_Test_Data.copy()
##############################evaluation
for f in range(1,13):
    index = indices[New_X_Data.shape[1]-f]
    temp_index.append(index)

New_X_Data = sci.delete(New_X_Data_ori,temp_index,1)
New_Test_Data = sci.delete(New_Test_Data_ori,temp_index,1)
print  New_X_Data.shape,New_Test_Data.shape,temp_index
    
clf_l1_LR = linear_model.LogisticRegression(C=0.1,penalty='l1',tol=0.001,solver='liblinear',max_iter=500)
clf_l1_LR.fit(New_X_Data,Y_Label)
New_Test_Data_res = clf_l1_LR.predict_proba(New_Test_Data)
Total_res = np.column_stack((New_Test_Data_ori[:,0].astype(int),New_Test_Data_res[:,1]))
np.savetxt('1.csv',Total_res,fmt='%.4f')

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



