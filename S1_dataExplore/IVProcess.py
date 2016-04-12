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
    corresWOE = pd.Series()##the correspoding WOE for each element in a column
    #replacedclm = pd.DataFrame()
    dropdf = pd.DataFrame()
    tmptotal = pd.DataFrame()
    Iv_object = pd.DataFrame()
    WOE = pd.Series()##pure WOE 
    #dateser = pd.Series()
    ori_train_file = ''
    ori_type_file = ''
    ori_test_file = ''
    segnum = 0
    case = 0
    timeoption = 0
    NaNRate = 0
    IV = 0
    reflag = 0
    IVthreshhold = 0
    indexname = ''
    labelname = ''
    def __init__(self, ori_train_file_name, ori_test_file_name, ori_type_file_name, indexname, labelname, segnum, case, timeoption, NaNRate, IVthreshhold):
        self.ori_train_file = pd.read_csv(ori_train_file_name,encoding="gb18030")
        self.ori_type_file = pd.read_csv(ori_type_file_name,encoding="gb18030")
        self.ori_test_file = pd.read_csv(ori_test_file_name,encoding="gb18030")
        #self.ori_test_file[19999,:] =
        self.labelname = labelname
        self.indexname = indexname
        self.segnum = segnum
        self.case = case
        self.timeoption = timeoption
        self.NaNRate = NaNRate
        self.IVthreshhold = IVthreshhold
        self.__replacenull()
        self.__resetType()

   

    def __replacenull(self):
        self.ori_train_file = self.ori_train_file.replace(-1, np.nan)
        self.ori_test_file = self.ori_test_file.replace(-1, np.nan)
        print 'all -1 has been replaced'

    def __resetType(self):       
        typetmp = self.ori_type_file.dropna(axis=1)
        se = typetmp.set_index(self.indexname)
        ser = pd.Series(se['Index'],index = typetmp[self.indexname])
        for i in ser.index:
            if ser[i]=='Categorical':
                ser[i] = 'object'
            elif ser[i]=='Numerical':
                ser[i] = 'float64'
            else:
                print str(i),ser[i]
            try:
                 self.ori_train_file[i] =  self.ori_train_file[i].astype(ser[i])
            except:
                 print 'cannot change type in train',str(i),ser[i]
            try:
                 self.ori_test_file[i] =  self.ori_test_file[i].astype(ser[i])
            except:
                 print 'cannot change type in test',str(i),ser[i]
        print 'resetType finished'

    def __WOEcal(self, goodpct, badpct):
        pct_ratio = goodpct/badpct
        pct_ratio =  pct_ratio.fillna(0.)
        npratio = np.array(pct_ratio)
        WOE = np.log(npratio)
        self.WOE = pd.Series(WOE)


    def __IVcal(self, good,bad):
        goodall = good.sum()
        badall = bad.sum()
        pct_good = good/goodall
        pct_bad = bad/badall
        self.__WOEcal(pct_good,pct_bad)
        pct_diff = pct_good-pct_bad
        pct_diff =  pct_diff.fillna(0.)
        npdiff = np.array(pct_diff)
        iv = npdiff*self.WOE
        ivser = pd.Series(iv)
        self.IV = ivser.sum()


    def __Objectprocess(self, goodser, badser, totalser):##case =1 replace unicode ,case = 2 replace all
        #tmpWOE = pd.Series()
        goodOValue = goodser.value_counts().sort_index()
        badOValue = badser.value_counts().sort_index()
        totalOvalue = totalser.value_counts().sort_index()
        badtmp = totalOvalue.copy() ##length normalized
        goodtmp = totalOvalue.copy()
        badtmp[(totalOvalue - badOValue).notnull()] = badOValue+1 #revise by weiwu
        badtmp[(totalOvalue - badOValue).isnull()] = 1
        goodtmp[(totalOvalue - goodOValue).notnull()] = goodOValue+1
        goodtmp[(totalOvalue - goodOValue).isnull()] = 1

        good_pct = goodtmp/goodtmp.sum()
        bad_pct = badtmp/badtmp.sum()
        totalOvalue = totalOvalue.sort_index()
        self.corresWOE =totalOvalue.copy()
        self.__WOEcal(good_pct,bad_pct)
        self.corresWOE[:] = self.WOE###coresponding woe
        self.tmptotal = totalser.copy()###original series
        self.tmptotal = self.__repalceWOE(self.tmptotal)
        self.__IVcal(goodtmp,badtmp)
        #print self.IV
        return self.tmptotal


    def __repalceWOE(self,oriseries):##replace unicode with woe
        def _case1():
            totalidx = self.corresWOE.index
            for i in oriseries.index:
                if type(oriseries[i]) == unicode:
                    try:
                        idx = totalidx.get_loc(oriseries[i])
                    except:
                        pass
                    oriseries[i] = self.corresWOE[idx]
                    self.reflag = 1
                else:pass
        def _case2():
            totalidx = self.corresWOE.index
            for i in self.tmptotal.index:
                #if type(tmptotal[i]) == unicode:
                idx = totalidx.get_loc(self.tmptotal[i])
                self.tmptotal[i] = self.corresWOE[idx]
        if self.case == 1:
            _case1()
        elif self.case == 2:
            _case2()
        return oriseries###return the replaced series

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
        self.__IVcal(goodsum, badsum)


    def __dateprocess(self, dateseries):
        dateser = pd.to_datetime(dateseries)
        dateser = dateser-dateser.min()
        return dateser
         

    def __replaceunicode(self,repalceser):
        flag = 0
        for i in repalceser.index:
            if type(repalceser.ix[i]) == unicode:
                flag=1
                break
        if flag == 1:
            repalceser = repalceser.fillna(u'no') #option with D E
        return repalceser

    
    
    '''def __del_low_IV(self,train,test):
        if self.IV >1:
            continue
            print clm, 'bug'
        elif self.IV < self.IVthreshhold:
            self.dropdf[clm] = train
            print clm,' drop  with low IV'
            del train
            del test'''



    def IVFunc(self):#case defined in Objectprocess ,1 only replace unicoid ,2 replace all
        lst = []
        #count = 0
        newFrame = self.ori_train_file.set_index(self.indexname)
        testFrame = self.ori_test_file.set_index(self.indexname)
        goodN = newFrame.groupby(self.labelname).get_group(1)
        badN = newFrame.groupby(self.labelname).get_group(0)
        totalNum = len(newFrame[self.labelname])
        for clm in newFrame.columns:
##            if clm == 'ThirdParty_Info_Period7_16' or clm == 'ThirdParty_Info_Period7_10':
##                print 'debug ..'
##                raw_input()
            if clm == self.labelname:
                continue
            elif clm == 'ListingInfo': #datatype
                if self.timeoption == 1:
                    newFrame[clm] = self.__dateprocess(newFrame[clm])
                    newFrame[clm] = newFrame[clm].astype('timedelta64[D]')
                    testFrame[clm] = self.__dateprocess(testFrame[clm])
                    testFrame[clm] = testFrame[clm].astype('timedelta64[D]')
            else:
                rate = float(len(newFrame[clm].dropna()))/totalNum
                if rate < self.NaNRate:
                    print clm,' drop ',str(rate)
                    self.dropdf[clm] = newFrame[clm]
                    del newFrame[clm]
                    del testFrame[clm]
                    continue
                else:
                    if newFrame[clm].dtypes =='object':
                        newFrame[clm] = self.__replaceunicode(newFrame[clm])
                        newFrame[clm] = self.__Objectprocess(goodN[clm],badN[clm],newFrame[clm])
                        if self.IV < self.IVthreshhold:
                            self.dropdf[clm] = newFrame[clm]
                            print clm,' drop  with low IV ',self.IV
                            del newFrame[clm]
                            del testFrame[clm]
                            continue
                        testFrame[clm] = self.__replaceunicode(testFrame[clm])
                        tmpser = testFrame[clm].copy()
                        testFrame[clm] = self.__repalceWOE(tmpser)
                        for i in testFrame[clm].index:
                            if type(testFrame.ix[i,clm]) == unicode:
                                testFrame[clm] = testFrame[clm].replace(testFrame[i,clm],self.corresWOE[u'no'])
                        #if self.reflag == 1:
                        newFrame[clm] = newFrame[clm].astype('float64')

                        #self.replacedclm[clm] = newFrame[clm]
                    ######## numerical#######
                    else:
                        ##                        
                        ####### Discretization
                        Maxmum = float(newFrame[clm].max())
                        Minmum = float(newFrame[clm].min())
                        dist = float((Maxmum - Minmum)/self.segnum)
                        if dist == 0:
                            self.dropdf[clm] = newFrame[clm]
                            print clm,' drop  with 0 dist '
                            del newFrame[clm]
                            del testFrame[clm]
                            continue 
                        elif (Maxmum - Minmum) < self.segnum:
                            newFrame[clm] = self.__Objectprocess(goodN[clm],badN[clm],newFrame[clm])
                            if self.IV < self.IVthreshhold:
                                self.dropdf[clm] = newFrame[clm]
                                print clm,' drop  with low IV',self.IV
                                del newFrame[clm]
                                del testFrame[clm]
                        else:
                            bins =[]
                            for i in range(0,self.segnum+1):
                                start = Minmum + i*dist
                                bins.append(start)
                            self.__NUMprocess(goodN[clm],badN[clm],newFrame[clm],bins)
                            if self.IV >1:
                                print clm, 'bug'
                                continue
                            elif self.IV < self.IVthreshhold:
                                self.dropdf[clm] = newFrame[clm]
                                print clm,' drop  with low IV',self.IV
                                del newFrame[clm]
                                del testFrame[clm]
                    try:
                        train_mean = newFrame.mean() # NaAN is not include in object
                        newFrame[clm]=newFrame[clm].fillna(train_mean)
                        testFrame[clm]=testFrame[clm].fillna(train_mean)
                    except KeyError:
                        pass
                    lst.append(self.IV)
        return lst,newFrame,testFrame,self.dropdf

if __name__=='__main__':
    FilePath = '../data/'
    Dataset = 'Training Set/'
    Testset = 'Test Set/'
    MasterTrainFile = FilePath + Dataset + 'PPD_Training_Master_GBK_3_1_Training_Set.csv'
    MasterTestile = FilePath + Testset + 'PPD_Master_GBK_2_Test_Set.csv'
    MasterFileType = FilePath + 'TypeState.csv'
    ###ori_train_file_name, ori_test_file_name, ori_type_file_name, indexname, labelname, segnum, case, timeoption, NaNRate, IVthreshhold
    ivpro = IVProcess(MasterTrainFile,MasterTestile,MasterFileType,'Idx','target',5000000,1,1,0.025,0.0005)
    IV, DF, DF_test, droped = ivpro.IVFunc()
    
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # 
    #TESTDATA = Dateprocess(Master, TEST, TypeStatement,20,1,0,0.75)
    #DF_test,droped_Test = Dateprocess.handel_test(TESTDATA)
    #print DF
    DF.to_csv('../Output/S1/seleted_Master_Train.csv', encoding="gb18030")
    droped.to_csv('../Output/S1/droped_Master_Train.csv', encoding="gb18030")
    DF_test.to_csv('../Output/S1/seleted_Master_Test.csv', encoding="gb18030")
    print 'the end'
    #droped_Test.to_csv('../Output/S1/statistics/droped_Master_Test.csv', encoding="gb18030")
    #droped_test.to_csv('../Output/S1/statistics/droped_Master.csv', encoding="gb18030")

        
