#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import random
import numpy as np
import pandas as pd
from datetime import datetime


FilePath = '../data/'
Dataset = 'Training Set/'
MasterFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
fhadle = open(MasterFile,'r')
data = fhadle.readlines()
fhadle.close()
data = map(lambda x:x.decode('gbk').strip().split(',')+['train_master'],data)
MasterData = pd.DataFrame(data[1:],columns=data[0][:-1]+['flag'])

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

Summury = pd.read_csv('PPD_summary.csv')
Summary = Summury.ix[:,'Columns':'type']
se = Summary.set_index('Columns')
se1 = pd.Series(se['type'],index = se.index)

for i in se1.index:
    try:
        Master[i] = Master[i].astype(se1.ix[i])
    except:pass
    
print Master.dtypes
Types = Master.dtypes
#Types.to_csv("sel.csv",index=False,encoding="gb2312")
Types.to_csv('types.csv')
