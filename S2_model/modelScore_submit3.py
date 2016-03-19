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

#####################################test and submit 3

skf = cross_validation.StratifiedKFold(Y_Label.values,25)

for train_index, test_index in skf:
    X_Data_Train = New_X_Data[train_index]
    X_Data_Test = New_X_Data[test_index]
    Y_Label_Train = Y_Label[train_index]
    Y_Label_Test = Y_Label[test_index]
    #print test_index,len(test_index)
    cost_time = time()
    grd = ensemble.RandomForestClassifier(max_depth=8, n_estimators=400)
    grd.fit(X_Data_Train, Y_Label_Train)
    cost_time = time() - cost_time
    print cost_time
    cost_time = time()
    grd_enc = preprocessing.OneHotEncoder()

    temp_X_Train = grd.apply(X_Data_Train)
    temp_X_Test = grd.apply(X_Data_Test)

    grd_enc.fit(temp_X_Train)
    cost_time = time() - cost_time
    print cost_time
    New_X_Data_Train = grd_enc.transform(temp_X_Train)
    New_Test_Data = grd_enc.transform(temp_X_Test)
   

    print temp_X_Train.shape,New_X_Data_Train.shape
    for i,C in enumerate((0.06,0.05)):
        grd_lm = linear_model.LogisticRegression(C=C,penalty='l1',tol=0.002,solver='liblinear',max_iter=400)
        grd_lm.fit(New_X_Data_Train, Y_Label_Train)

        y_pred_grd_lm = grd_lm.predict_proba(New_Test_Data)[:, 1]
        res_l2_LR = metrics.roc_auc_score(Y_Label_Test, y_pred_grd_lm)
        print i,res_l2_LR


    y_pred_grd = grd.predict_proba(X_Data_Test)[:, 1]
    res_l2_LR = metrics.roc_auc_score(Y_Label_Test, y_pred_grd)
    print res_l2_LR
    

##grd = ensemble.GradientBoostingClassifier(n_estimators=300,max_depth=4) 
##grd.fit(New_X_Data, Y_Label)
##temp_X_Train = grd.apply(New_X_Data)[:, :, 0]
##temp_X_Test = grd.apply(New_Test_Data)[:, :, 0]
##grd_enc = preprocessing.OneHotEncoder()
##grd_enc.fit(temp_X_Train)
##New_temp_X_Train = grd_enc.transform(temp_X_Train)
##New_temp_X_Test = grd_enc.transform(temp_X_Test)
##grd_lm = linear_model.LogisticRegression(C=0.06,penalty='l1',tol=0.005,solver='liblinear',max_iter=400)
##grd_lm.fit(New_temp_X_Train, Y_Label)
##New_Test_Data_res = grd_lm.predict_proba(New_temp_X_Test)
##Total_res = np.column_stack((New_Test_Data[:,0].astype(int),New_Test_Data_res[:,1]))
##np.savetxt('2.csv',Total_res,fmt='%.6f')


######################################################






