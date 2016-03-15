#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

FilePath = '../data/'
Dataset = 'Training Set/'
MasterFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
Master = pd.read_csv(MasterFile,encoding="gb18030")

#def WOEcalculate(good, bad, total):


def IVFunc(Aframe, k):
    lst = []
    newFrame = Aframe.set_index('Idx')
    if -1 in newFrame:
        print 'aaaaaa!'
    goodN = newFrame.groupby('target').get_group(0)
    badN = newFrame.groupby('target').get_group(1)
    dropgroup = []
    for clm in newFrame.columns:
        #tmpFrame = newFrame[clm].dropna()
        if len(newFrame[clm].dropna())<30000*0.8:
            del newFrame[clm]
        else:
            ###### Handling the Categories ######
            oricount = newFrame[clm].count()
            if newFrame[clm].dtypes =='object':
                goodOValue = goodN[clm].value_counts()
                #print goodOValue
                badOValue = badN[clm].value_counts()
                totalOvalue = newFrame[clm].value_counts()
                #print (totalOvalue - goodOValue).isnull()
                #addindex =  totalOvalue[(totalOvalue - badOValue).isnull()].index
                #badOValue[addindex] =1
                #print totalOvalue[(totalOvalue - badOValue).isnull()]
                #print badOValue[(totalOvalue - badOValue).isnull()]
                #goodOValue = goodOValue
                #badOValue = badOValue
                #totalOvalue
                pct_good = goodOValue/totalOvalue
                pct_bad = badOValue/totalOvalue

                #print goodOValue
                #print badOValue
                #print totalOvalue
                #print pct_good
                #print pct_bad
            ######## numerical#######
            else:
                ####### Discretization
                Maxmum = float(newFrame[clm].max())
                Minmum = float(newFrame[clm].min())
                dist = float((Maxmum - Minmum)/k)
                if dist == 0.:
                    dropgroup.append(newFrame[clm])
                    del newFrame[clm]
                    continue
                bins =[]
                for i in range(0,k+1):
                    start = Minmum + i*dist
                    bins.append(start)
                totalcuts = pd.cut(newFrame[clm], bins, right = True, include_lowest = True)
                goodcuts = pd.cut(goodN[clm], bins, right = True, include_lowest = True)
                badcuts = pd.cut(badN[clm], bins, right = True, include_lowest = True)

                ####### do statistics
                totalsum = totalcuts.value_counts()
                goodsum = goodcuts.value_counts()
                badsum = badcuts.value_counts()

                totalsum = totalsum.sort_index() # sort the index
                goodsum = goodsum.sort_index()# sort the index
                badsum = badsum.sort_index()# sort the index
                badsum[(totalsum==0) != (badsum == 0)] = 1
                badall = badsum.sum()
                goodsum[(totalsum==0) != (goodsum == 0)] = 1
                goodall = goodsum.sum()
                #print (totalsum==0) == (badsum == 0)

                ###### WOE CACULATINF ######
                pct_good = goodsum/goodall
                pct_bad = badsum/badall
                pct_diff = pct_good-pct_bad
                pct_ratio = pct_good/pct_bad
                pct_diff =  pct_diff.fillna(0.)
                pct_ratio =  pct_ratio.fillna(0.)
                #print pct_diff
                #print pct_ratio
                npdiff = np.array(pct_diff)
                npratio = np.array(pct_ratio)
                #print npdiff
                #print npratio

                WOE = np.log(npratio)
                iv = npdiff*WOE
                #print iv
                ivser = pd.Series(iv)
                IV = ivser.sum()
                lst.append(IV)
    return lst,newFrame

IV,DF = IVFunc(Master,20)
print IV
for element in IV:
    if element < 0.5:
        print element
#print IV
#print len(DF.columns)
#DF.to_csv('../Output/S1/staticstics/firstclean.csv',encoding = 'gb18030')