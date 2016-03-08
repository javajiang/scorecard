#coding = 'utf-8'
author = "changandao"

import random
import numpy as np
import pandas as pd


FilePath = '../data/Training Set/'
MasterFile = FilePath + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
f = open(MasterFile,'r')
data = f.readlines()
f.close()
data = map(lambda x:x.decode('gbk').strip().split(',')+['train_master'],data)
MasterData = pd.DataFrame(data[1:],columns=data[0][:-1]+['flag'])
Describe = MasterData.describe(include='all')
print MasterData.dtypes
print MasterData.get_dtype_counts()
MasterData['Idx'] = MasterData['Idx'].astype('int32')
print MasterData.dtypes
Describe.to_csv('../Output/foo.csv', encoding='gbk')
raw_input()

