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

def WOEFunc(Aframe, k):
    lst = []
    newFrame = Aframe.set_index('Idx')
    goodN = newFrame.groupby('target').get_group(0)
    badN = newFrame.groupby('target').get_group(1)
    #print goodN
    for clm in newFrame.columns:
        #totalsum = newFrame[clm].value_counts()
        Maxmum = newFrame[clm].max()
        Minmum = newFrame[clm].min()
        dist = (Maxmum - Minmum)/k
        start = Minmum
        bins =[]
        for i in range(0,k+1):
            start += i*dist
            bins.append(start)
        totalcuts = pd.cut(newFrame[clm], bins, right = False, include_lowest = True)
        goodcuts = pd.cut(goodN[clm], bins, right = False, include_lowest = True)
        badcuts = pd.cut(badN[clm], bins, right = False, include_lowest = True)
        totalsum = totalcuts.value_counts()
        goodsum = goodcuts.value_counts()
        badsum = badcuts.value_counts()
        pct_good = goodsum/totalsum
        pct_bad = badsum/totalsum
        IV=0
        for i in totalsum:
            try:
                IV += (pct_good[i]-pct_bad[i])*math.log(pct_good[i]/pct_bad[i])
            except:pass
        lst.append[IV]
        #woe = math.log(pct_good/pct_bad)
        #lst.append(woe)
    return lst

IVlist = []
IV = WOEFunc(Master,20)
print IVlist