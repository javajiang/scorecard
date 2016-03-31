import numpy as np
import pandas as pd

FilePath = '../Output/S1/'

MasterTrainFile = FilePath + 'seleted_Master_Train.csv'
MasterTestile = FilePath + 'seleted_Master_Test.csv'

Trainadd1file = FilePath + 'trainadd1.csv'
Trainadd2file = FilePath + 'trainadd2.csv'
Testadd1file = FilePath + 'testadd1.csv'
Testadd2file = FilePath + 'testadd2.csv'

MasterTrain = pd.read_csv(MasterTrainFile,encoding="gb18030")
MasterTest = pd.read_csv(MasterTestile,encoding="gb18030")
#print MasterTrain
Trainadd1 = pd.read_csv(Trainadd1file,encoding="gb18030")
Trainadd2 = pd.read_csv(Trainadd2file,encoding="gb18030")
Testadd1 = pd.read_csv(Testadd1file,encoding="gb18030")
Testadd2 = pd.read_csv(Testadd2file,encoding="gb18030")

Trainadd = Trainadd1.join(Trainadd2.ix[:,1:])
Testadd = Testadd1.join(Testadd2.ix[:,1:])
#print Trainadd
#print MasterTrain
#Trainadd = Trainadd.set_index('Unnamed: 0')
#Testadd = Testadd.set_index('Unnamed: 0')
print MasterTrain.shape,Trainadd.shape
newMastertrain = pd.concat([MasterTrain, Trainadd], axis=1)
#print newMastertrain

print MasterTrain.ix[:,0]
print Trainadd1.ix[:,0]
lacked = []
has = []
#newframe = pd.DataFrame()
for idx in MasterTrain.ix[:,0]:
    if idx in Trainadd1.ix[:,0]:
        #print idx
        has.append(idx)
    else:
        #print idx
        lacked.append(idx)

#testhas,testlack = generate_new(MasterTest,Testadd)
print len(has),len(lacked)
