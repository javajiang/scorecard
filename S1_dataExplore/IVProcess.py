
#coding = 'utf-8'
"""data explore for raw datasheet"""
__author__ = "changandao&jiangweiwu"
__date__   = "2016.3.8"

import numpy as np
import pandas as pd
from sklearn import *
#from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from pandas import datetime


class IVProcess:
    corresWOE = pd.DataFrame()
    replacedclm = pd.DataFrame()
    dropdf = pd.DataFrame()
    ori_train_file = ''
    ori_type_file = ''
    ori_test_file = ''
    k = 0
    case = 0
    timeoption = 0
    NaNRate = 0
    def __init__(self, ori_train_file_name, ori_test_file_name, ori_type_file_name, k, case, timeoption,NaNRate):
        self.ori_train_file = pd.read_csv(ori_train_file_name,encoding="gb18030")
        self.ori_type_file = pd.read_csv(ori_type_file_name,encoding="gb18030")
        self.ori_test_file = pd.read_csv(ori_test_file_name,encoding="gb18030")
        self.k = k
        self.case = case
        self.timeoption = timeoption
        self.NaNRate = NaNRate
        self.ori_train_file = self.__replacenull(self.ori_train_file)
        self.ori_test_file = self.__replacenull(self.ori_test_file)
        print 'all -1 has been replaced'
        self.__resetType(self.ori_train_file,self.ori_type_file)
        self.__resetType(self.ori_test_file,self.ori_type_file)
        print 'resetType finished'
   

    def __replacenull(self,ori_file):
        ori_file = ori_file.replace(-1, np.nan)
        return ori_file

    def __WOEFUNC(self, goodpct, badpct):
        pct_ratio = goodpct/badpct
        pct_ratio =  pct_ratio.fillna(0.)
        npratio = np.array(pct_ratio)
        WOE = np.log(npratio)
        WOE = pd.Series(WOE)
        return WOE


    def __IVCAL(self, good,bad):
        goodall = good.sum()
        badall = bad.sum()
        pct_good = good/goodall
        pct_bad = bad/badall
        WOEvalue = self.__WOEFUNC(pct_good,pct_bad)
        pct_diff = pct_good-pct_bad
        pct_diff =  pct_diff.fillna(0.)
        npdiff = np.array(pct_diff)
        iv = npdiff*WOEvalue
        ivser = pd.Series(iv)
        IV = ivser.sum()
        return IV


    def __Objectprocess(self, goodser, badser,totalser):##case =1 replace unicode ,case = 2 replace all
        goodOValue = goodser.value_counts()
        badOValue = badser.value_counts()
        totalOvalue = totalser.value_counts()
        badtmp = totalOvalue.copy()
        goodtmp = totalOvalue.copy()
        badtmp[(totalOvalue - badOValue).notnull()] = badOValue+1
        badtmp[(totalOvalue - badOValue).isnull()] = 1
        goodtmp[(totalOvalue - goodOValue).notnull()] = goodOValue+1
        goodtmp[(totalOvalue - goodOValue).isnull()] = 1
        goodtmp = goodtmp.sort_index()
        badtmp = badtmp.sort_index()
        good_pct = goodtmp/goodtmp.sum()
        bad_pct = badtmp/badtmp.sum()
        totalOvalue = totalOvalue.sort_index()
        self.corresWOE =totalOvalue.copy()
        Woe_of_O = self.__WOEFUNC(good_pct,bad_pct)
        self.corresWOE[:] = Woe_of_O###coresponding woe
        tmptotal = totalser.copy()###original series
        tmptotal,shouldreplace = self.__repalcewoe(tmptotal)
        IV_Objekt = self.__IVCAL(goodtmp,badtmp)
        return IV_Objekt,tmptotal,shouldreplace


    def __repalcewoe(self,oriseries):##replace unicode with woe
        shouldreplace = 0
        def _case1(tmp,ser):
            totalidx = ser.index
            for i in tmp.index:
                if type(tmp[i]) == unicode:
                    idx = totalidx.get_loc(tmp[i])
                    tmp[i] = ser[idx]
                    shouldreplace = 1
            return tmp
        def _case2(tmp,ser):
            totalidx = ser.index
            for i in tmp.index:
                #if type(tmptotal[i]) == unicode:
                idx = totalidx.get_loc(tmp[i])
                tmp[i] = ser[idx]
            return tmp
        if self.case == 1:
            oriseries = _case1(oriseries,self.corresWOE)
        elif self.case == 2:
            oriseries = _case2(oriseries,self.corresWOE)
        return oriseries,shouldreplace###return the replaced series and flag of relace all or only unicode

    def __NUMprocess(self, goodser, badser, totalser,bin):
        totalcuts = pd.cut(totalser, bin, right = True, include_lowest = True)
        goodcuts = pd.cut(goodser, bin, right = True, include_lowest = True)
        badcuts = pd.cut(badser, bin, right = True, include_lowest = True)

        ####### do statistics
        totalsum = totalcuts.value_counts()
        goodsum = goodcuts.value_counts()
        badsum = badcuts.value_counts()

        totalsum = totalsum.sort_index() # sort the index
        goodsum = goodsum.sort_index()# sort the index
        badsum = badsum.sort_index()# sort the index
        badsum[(totalsum==0) != (badsum == 0)] = 1
        goodsum[(totalsum==0) != (goodsum == 0)] = 1
        IV_NUM = self.__IVCAL(goodsum, badsum)
        return IV_NUM


    def __dateprocess(self, dateseries):
        dateser = pd.to_datetime(dateseries)
        dateser = dateser-dateser.min()
        return dateser


    def __replaceunicode(self,repalceser):
        flag = 0
        for i in repalceser.index:
            if type(repalceser.ix[i]) == unicode:
                flag=1
            else:flag = 0
        if flag == 1:
            repalceser = repalceser.fillna(u'no') #option with D E
        else:pass
        return repalceser


    def handel_test(self):
        testdrop = pd.DataFrame()
        testframe = self.ori_train_file.set_index('Idx')
        for clm in testframe.columns:
            if clm == 'ListingInfo':
                if self.timeoption == 0:
                    testframe[clm] = self.__dateprocess(testframe[clm])
                else:continue
            elif clm in self.dropdf.columns:
                testdrop[clm] = testframe[clm]
                del testframe[clm]
            elif clm in self.replacedclm.columns:
                testframe[clm] = self.__replaceunicode(testframe[clm])
                testframe[clm],shouldreplace = self.__repalcewoe(testframe[clm])
            else:pass
        return testframe,testdrop



    def __resetType(self,ori_file,ori_type_file):
        typetmp = ori_type_file.dropna(axis=1)
        se = typetmp.set_index('Idx')
        ser = pd.Series(se['Index'],index = typetmp['Idx'])
        for i in ser.index:
            if ser[i]=='Categorical':
                ser[i] = 'object'
            elif ser[i]=='Numerical':
                try:
                    if ori_file[i].dtypes == 'object':
                        ser[i] = 'float64'
                    else:pass
                except KeyError:pass
            else:pass
            try:
                ori_file[i] = ori_file[i].astype(ser[i])
            except:pass

        return ori_file


    def IVFunc(self):#case defined in Objectprocess ,1 only replace unicoid ,2 replace all
        lst = []
        final_IV = 0
        newFrame = self.ori_train_file.set_index('Idx')
        goodN = newFrame.groupby('target').get_group(1)
        badN = newFrame.groupby('target').get_group(0)
        for clm in newFrame.columns:
            if clm == 'target':
                continue
            elif clm == 'ListingInfo':
                if self.timeoption == 0:
                    newFrame[clm] = self.__dateprocess(newFrame[clm])
                else:continue
            else:
                rate = float(len(newFrame[clm].dropna()))/30000
                if rate<self.NaNRate:
                    print clm,' drop ',str(rate)
                    self.dropdf[clm] = newFrame[clm]
                    del newFrame[clm]
                    continue

                else:
                    if newFrame[clm].dtypes =='object':
                        newFrame[clm] = self.__replaceunicode(newFrame[clm])
                        final_IV,newFrame[clm],reflag = self.__Objectprocess(goodN[clm],badN[clm],newFrame[clm])
                        if reflag == 1:
                             self.replacedclm[clm] = newFrame[clm]
                        else:pass
                        if final_IV >1:
                            continue

                    ######## numerical#######
                    else:
                        continue
        print newFrame.dtypes
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(newFrame)
        test = imp.transform(newFrame)
        print newFrame.shape,test.shape
        newFrame[clm] = test
                        
                        ####### Discretization
##                        Maxmum = float(newFrame[clm].max())
##                        Minmum = float(newFrame[clm].min())
##                        dist = float((Maxmum - Minmum)/self.k)
##                        if (Maxmum - Minmum) < self.k:
##                            final_IV,newFrame[clm],reflag = self.__Objectprocess(goodN[clm],badN[clm],newFrame[clm])
##                        else:
##                            if dist == 0:
##                                self.dropdf[clm] = newFrame[clm]
##                                print clm,' drop  with 0 dist '
##                                del newFrame[clm]
##                                continue
##                            bins =[]
##                            for i in range(0,self.k+1):
##                                start = Minmum + i*dist
##                                bins.append(start)
##                            final_IV = self.__NUMprocess(goodN[clm],badN[clm],newFrame[clm],bins)
##                            if final_IV >1:
##                                continue
##                            elif final_IV >0.1:
##                                pass

##                    lst.append(final_IV)
        return lst,newFrame,self.dropdf

if __name__=='__main__':
    FilePath = '../data/'
    Dataset = 'Training Set/'
    Testset = 'Test Set/'
    MasterTrainFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
    MasterTestile = FilePath + Testset + 'PPD_Master_GBK_2_Test_Set.csv'
    MasterFileType = FilePath + 'TypeState.csv'

    ivpro = IVProcess(MasterTrainFile,MasterTestile,MasterFileType,20,1,0,0.75)
    IV, DF, droped = ivpro.IVFunc()
    
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # 
    #TESTDATA = Dateprocess(Master, TEST, TypeStatement,20,1,0,0.75)
    DF_test,droped_Test = Dateprocess.handel_test(TESTDATA)
    #print DF
    DF.to_csv('../Output/S1/statistics/seleted_Master_Train.csv', encoding="gb18030")
    droped.to_csv('../Output/S1/statistics/droped_Master_Train.csv', encoding="gb18030")
    DF_test.to_csv('../Output/S1/statistics/seleted_Master_Train.csv', encoding="gb18030")
    droped_Test.to_csv('../Output/S1/statistics/droped_Master_Test.csv', encoding="gb18030")
    #droped_test.to_csv('../Output/S1/statistics/droped_Master.csv', encoding="gb18030")
