#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

FilePath = '../data/'
Dataset = 'Training Set/'
MasterFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
Master = pd.read_csv(MasterFile,encoding="gb18030")
TypeStatement = pd.read_csv(FilePath + 'TypeState.csv')
Master.dtypes.to_csv('../Output/S1/statistics/oritypes.csv')

#print Master.info()

###### handeling the dtype ######
TypeStatement = TypeStatement.dropna(axis=1)
se = TypeStatement.set_index('Idx')
ser = pd.Series(se['Index'],index = TypeStatement['Idx'])
for i in ser.index:
    if ser[i]=='Categorical':
        ser[i] = 'object'
    #else:pass
    elif ser[i]=='Numerical':
        if Master[i].dtypes == 'object':
            print i, Master.columns.get_loc(i)
            ser[i] = 'float64'
        else:pass
    else:pass
    try:
        Master[i] = Master[i].astype(ser[i])
    except:pass
Master.dtypes.to_csv('../Output/S1/statistics/newtypes.csv')
Master_describe = Master.describe().transpose().to_csv('../Output/S1/statistics/summary/describe_summary.csv',encoding='utf-8',na_rep='NaN')



Mdct = {}
for clm in Master.columns:
    if clm=='Idx':
        continue
    num = Master[clm].value_counts()
    num.to_csv('../Output/S1/statistics/summary/'+clm+'_count.csv',encoding='utf-8',na_rep='NaN')
    Mdct[clm] = num

print len(Mdct)



for key in Mdct.keys():
    if key=='Idx':
        continue
    #print Mdct[key].index
    try:
        if len(Mdct[key])>100:
               print key,'too many items',str(len(Mdct[key]))
               continue
        plt.bar(left=Mdct[key].index,height=Mdct[key])
        plt.savefig('../Output/S1/statistics/fig/'+str(key)+'.png', format='png')
        plt.close('all')
    except:
        print key,'fail'


##
##
##Summary = Summury.ix[:,'Columns':'type']
##se = Summary.set_index('Columns')
##se1 = pd.Series(se['type'],index = se.index)
##
##for i in se1.index:
##    try:
##        MasterData[i] = MasterData[i].astype(se1.ix[i])
##    except:pass
##MasterData1 = MasterData.replace(-1,np.nan)
##print MasterData1['WeblogInfo_3'].dtypes
##
###print MasterData1
###print MasterData1.isnull()
##
##print MasterData.dtypes
##Types = MasterData.dtypes
##Types.to_csv("../Output/S1/sel.csv",index=False,encoding="gb18030")
##Types.to_csv('../Output/S1/types.csv')


