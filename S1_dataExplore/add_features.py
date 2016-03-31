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

#print Trainadd1.shape


def generate_new_feature(newframe,column):
    for clm in newframe.columns:
        if newframe[clm].dtypes == object:
            try:
                newframe[clm] = newframe[clm].astype('int32')
            except:pass
    df1 = pd.DataFrame()
    df1['Idx'] = newframe['Idx']
    df1[column] = newframe[column]

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
    print idxtype

    finalframe = pd.DataFrame(index = idxtype, columns = type_name_lst)
    finalarray = np.zeros(finalframe.shape)
    preid = -1
    currentid = 0
    typearray = np.array(lst)
    for i in df1.index:
        currentid = i
        rowplace = idxtype.index(df1.ix[i,0])
        clmplace = type.index(df1.ix[i,1])
        if currentid != preid:
            if preid ==-1:
                pass
            else:
                finalarray[rowplace] = typearray
            typearray = np.array(lst)
        #print i
        typearray[clmplace] += 1
        preid = i
    finalframe = pd.DataFrame(finalarray)
    finalframe.index = idxtype
    finalframe.columns = type_name_lst
    return finalframe


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


#train1_logframe1 = generate_new_feature(Trainadd1, 'LogInfo1')
#train1_logframe2 = generate_new_feature(Trainadd1, 'LogInfo2')
#train1_logser3 = generate_new_date(Trainadd1,'LogInfo3')
'''
train2_Userupser = generate_new_str2id(Trainadd2,'UserupdateInfo1')
Trainadd2['UserupdateInfo1'] = train2_Userupser.values
train2_userframe1 = generate_new_feature(Trainadd2,'UserupdateInfo1')
train2_userser2 = generate_new_date(Trainadd2,'UserupdateInfo2')

#train1frame = train1_logframe1.join(train1_logframe2).join(train1_logser3)
train2frame = train2_userframe1.join(train2_userser2)
#train1frame.to_csv('../Output/S1/trainadd1',encoding='gb18030')
train2frame.to_csv('../Output/S1/trainadd2',encoding='gb18030')'''

def generate_newframe1(oldframe):
    logframe1 = generate_new_feature(oldframe, 'LogInfo1')
    logframe2 = generate_new_feature(oldframe, 'LogInfo2')
    logser3 = generate_new_date(oldframe,'LogInfo3')
    newframe = logframe1.join(logframe2).join(logser3)
    return newframe


def generate_newframe2(oldframe):
    Userupser = generate_new_str2id(oldframe,'UserupdateInfo1')
    oldframe['UserupdateInfo1'] = Userupser.values
    userframe1 = generate_new_feature(oldframe,'UserupdateInfo1')
    userser2 = generate_new_date(oldframe,'UserupdateInfo2')
    newframe = userframe1.join(userser2)
    return newframe

trainframe1 = generate_newframe1(Trainadd1)
testframe1 = generate_newframe1(Testadd1)

trainframe2 = generate_newframe2(Trainadd2)
testframe2 = generate_newframe2(Testadd2)

trainframe1.to_csv('../Output/S1/trainadd1.csv',encoding='gb18030')
trainframe2.to_csv('../Output/S1/trainadd2.csv',encoding='gb18030')

testframe1.to_csv('../Output/S1/testadd1.csv',encoding='gb18030')
testframe2.to_csv('../Output/S1/testadd2.csv',encoding='gb18030')

