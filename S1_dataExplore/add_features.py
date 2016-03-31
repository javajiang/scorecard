import numpy as np
import pandas as pd

FilePath = '../data/'
Dataset = 'Training Set/'
Testset = 'Test Set/'
Trainadd1file = FilePath + Dataset + 'PPD_LogInfo_3_1_Training_Set.csv'
Trainadd2file = FilePath + Dataset + 'PPD_Userupdate_Info_3_1_Training_Set.csv'
Testadd1file = FilePath + Testset + 'PPD_LogInfo_2_Test_Set.csv'
Testadd2file = FilePath + Testset + 'PPD_Userupdate_Info_2_Test_Set.csv'


Trainadd1 = pd.read_csv(Trainadd1file,encoding="gb18030")
Trainadd2 = pd.read_csv(Trainadd2file,encoding="gb18030")
Testadd1 = pd.read_csv(Testadd1file,encoding="gb18030")
Testadd2 = pd.read_csv(Testadd2file,encoding="gb18030")
Trainadd1.fillna(0)
Trainadd2.fillna(0)
Testadd1.fillna(0)
Testadd2.fillna(0)

#print Trainadd1.shape


def generate_new_feature(newframe,testframe,column):
    for clm in newframe.columns:
        if newframe[clm].dtypes == object:
            try:
                newframe[clm] = newframe[clm].astype('int32')
            except:pass
    for clm in testframe.columns:
        if testframe[clm].dtypes == object:
            try:
                testframe[clm] = testframe[clm].astype('int32')
            except:pass
    df1 = pd.DataFrame()
    df1['Idx'] = newframe['Idx']
    df1[column] = newframe[column]

    df2 = pd.DataFrame()
    df2['Idx'] = newframe['Idx']
    df2[column] = newframe[column]


    typecounts = newframe[column].value_counts()
    type = typecounts.index
    type = sorted(type)

    lst = []
    type_name_lst = []
    for i in range(0,len(type)):
        lst.append(0)
        name = column + str(i)
        type_name_lst.append(name)

    idxcounts = df1['Idx'].value_counts()
    idxtype = idxcounts.index
    idxtype = sorted(idxtype)
    testidxcounts = testframe[column].value_counts()
    testidxtype = testidxcounts.index
    testidxtype = sorted(testidxtype)
    print idxtype

    def _generate_new(df,typelst,idx,type_name,inilst):
        finalframe = pd.DataFrame(index = idx, columns = type_name)
        finalarray = np.zeros(finalframe.shape)
        preid = -1
        currentid = 0
        typearray = np.array(inilst)
        for i in df.index:
            currentid = df.ix[i,0]
            rowplace = idx.index(df.ix[i,0])
            clmplace = typelst.index(df.ix[i,1])
            if currentid != preid:
                if preid ==-1:
                    pass
                else:
                    finalarray[rowplace] = typearray
                typearray = np.array(inilst)
            typearray[clmplace] += 1
            preid = df.ix[i,0]
        return finalarray
    #for i in df2.index:

    newarray = _generate_new(df1,type,idxtype,type_name_lst,lst)
    testarray = _generate_new(df2,type,testidxtype,type_name_lst,lst)
    finalframe = pd.DataFrame(newarray)
    testframe = pd.DataFrame(testarray)
    finalframe.index = idxtype
    finalframe.columns = type_name_lst
    testframe.index = testidxtype
    testframe.columns = type_name_lst
    return finalframe,testframe


def _str2id(str_list):
    str_list = map(lambda x: x.lower(), str_list)
    uni_list = list(set(str_list))
    id_list = []
    for tag in str_list:
        tag_id = uni_list.index(tag)
        id_list.append(tag_id)
    return id_list


def generate_new_str2id(newframe, column):
    idx = list(newframe['Idx'].values)
    idd = list(newframe[column].values)
    idlst = _str2id(idd)
    newser = pd.Series(idlst,index = idx)
    return newser


def _date_summary(index_list, date_list):
    date_info_dict = dict()
    for i in xrange(len(date_list)):
        date_index = index_list[i]
        if date_index not in date_info_dict:
            date_info_dict[date_index] = [date_list[i]]
        else:
            date_info_dict.get(date_index).append(date_list[i])
    for index in date_info_dict:
        date_info_dict[index] = len(list(set(date_info_dict.get(index))))
    return date_info_dict


def generate_new_date(newframe, column):
    idxlst = list(newframe['Idx'].values)
    datelst = list(newframe[column].values)
    date_info_dct = _date_summary(idxlst,datelst)
    date_info_ser = pd.Series(date_info_dct,name=column)
    return date_info_ser


def generate_newframe1(oldframe,oldtestframe):
    logframe1,testlogframe1 = generate_new_feature(oldframe,oldtestframe, 'LogInfo1')
    logframe2,testlogframe2 = generate_new_feature(oldframe,oldtestframe, 'LogInfo2')
    logser3, = generate_new_date(oldframe,'LogInfo3')
    testlogser3 = generate_new_date(oldtestframe)
    newframe = logframe1.join(logframe2).join(logser3)
    testnewframe = testlogframe1.join(testlogframe2).join(testlogser3)
    return newframe,testnewframe


def generate_newframe2(oldframe,oldtestframe):
    Userupser = generate_new_str2id(oldframe,'UserupdateInfo1')
    testUserupser = generate_new_str2id(oldtestframe,'UserupdateInfo1')
    oldframe['UserupdateInfo1'] = Userupser.values
    oldtestframe['UserupdateInfo1'] = testUserupser.values
    userframe1,testuserframe1 = generate_new_feature(oldframe,oldtestframe,'UserupdateInfo1')
    userser2 = generate_new_date(oldframe,'UserupdateInfo2')
    testuserser2 = generate_new_date(oldtestframe,'UserupdateInfo2')
    newframe = userframe1.join(userser2)
    testnewframe = testuserframe1.join(testuserser2)
    return newframe, testnewframe


trainframe1,testframe1 = generate_newframe1(Trainadd1,Testadd1)
#testframe1 = generate_newframe1(Testadd1)

trainframe2,testframe2 = generate_newframe2(Trainadd2,Testadd2)
#testframe2 = generate_newframe2(Testadd2)

trainframe1.to_csv('../Output/S1/trainadd1.csv',encoding='gb18030')
trainframe2.to_csv('../Output/S1/trainadd2.csv',encoding='gb18030')

testframe1.to_csv('../Output/S1/testadd1.csv',encoding='gb18030')
testframe2.to_csv('../Output/S1/testadd2.csv',encoding='gb18030')

print 'that is the end'
