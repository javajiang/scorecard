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
            ser[i] = 'float64'
        else:pass
    else:pass
    try:
        Master[i] = Master[i].astype(ser[i])
    except:pass
Master.dtypes.to_csv('../Output/S1/newtypes.csv')

Mdct = {}
count = 0
for clm in Master.columns:
    num = Master[clm].value_counts()
    #key = (clm, count)
    Mdct[count] = num
    count += 1
'''
flag = 0
for i,j in Mdct.items():
    try:
        plt.bar(Mdct[i].index, j)
        #plt.savefig('../Output/Images/'+str(i)+'.png', format='png')
        plt.show()
        flag += 1
        if flag >3:
            break
    except:pass'''
#plt.show()
flag = 0
for i in range(2,len(Mdct.keys())):
    try:
        plt.bar(Mdct[i].index, Mdct[i])
        plt.show()
        flag+=1
        raw_input('enen')
        if flag>5:
            break
        #plt.savefig('../Output/Images/'+str(i)+'.png', format='png')
    except:
        print 'haha'
#print num.level()
    #empty.append(num)
#empty.to_csv('../Output/S1/num.csv',encoding = 'gb18030')

'''
###### dataAnalyse #######
Mfeature = Master.describe()
def generatedct(serie,dct):
    for s in serie:
        dct[s] = dct.get(s,0)+1
    return dct

Mlist = []
for clm in Master.columns:
    Mdct = {}
    Mdct = generatedct(Master[clm],Mdct)
    Mlist.append(Mdct)

#Mfile = open('../Output/S1/tem.txt')
for i in range(1,len(Master.columns)-2):
    Xxis=Mlist[i].keys()
    Yxis=Mlist[i].values()

plt.bar(Xxis, Yxis)
plt.show()
'''


'''Summary = Summury.ix[:,'Columns':'type']
se = Summary.set_index('Columns')
se1 = pd.Series(se['type'],index = se.index)

for i in se1.index:
    try:
        MasterData[i] = MasterData[i].astype(se1.ix[i])
    except:pass
MasterData1 = MasterData.replace(-1,np.nan)
print MasterData1['WeblogInfo_3'].dtypes

#print MasterData1
#print MasterData1.isnull()'''
'''
print MasterData.dtypes
Types = MasterData.dtypes
Types.to_csv("../Output/S1/sel.csv",index=False,encoding="gb18030")
Types.to_csv('../Output/S1/types.csv')
<<<<<<< Updated upstream
'''

