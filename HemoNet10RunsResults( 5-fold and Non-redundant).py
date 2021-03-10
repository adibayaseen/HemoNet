# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:55:01 2021

@author: 92340
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:11:33 2019

@author: AdibaYaseen
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.svm import SVC
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import matthews_corrcoef
###
from roc import roc_VA
from new_AAC_Features_Extract import *
from  Clusterify import *
from  Results import *
###
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn import metrics
from Bio import SeqIO
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from sklearn.model_selection import KFold,StratifiedKFold
from platt import *
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        self.fc4 = nn.Linear(3072,2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x=torch.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x) 
        return x
"""
Created on Thu Oct  3 13:11:33 2019
https://allennlp.org/elmo
@author: AdibaYaseen
"""
from sklearn.svm import SVC
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import matthews_corrcoef
###
from roc import roc_VA
from new_AAC_Features_Extract import *
from  Clusterify import *
from  Results import *
###
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn import metrics
from Bio import SeqIO
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
path="D:\PhD\Hemo_All_SeQ/"
####ELMO Features
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
"""
NC Smiles Features
"""
N_terminous_names=pickle.load(open(path+'nTerminus_All_Dict.npy', "rb"))
C_terminous_names=pickle.load(open(path+'cTerminus_All_Dict.npy', "rb"))
dimension=1024
print("Feature Dimension",dimension)
N_terminous_Smiles_features=pickle.load(open(path+'Nmod_Dict_'+str(dimension)+'.npy', "rb"))
C_terminous_Smiles_features=pickle.load(open(path+'Cmod_Dict_'+str(dimension)+'.npy', "rb"))
###
N_terminous_Smiles_features['Free']=' ',np.zeros(len(list(N_terminous_Smiles_features.values())[0][1]))
C_terminous_Smiles_features['Free']=' ',np.zeros(len(list(C_terminous_Smiles_features.values())[0][1]))
N_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_hemo]
C_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1]for n in Names_hemo]
NC_hemo=np.hstack((N_hemo,C_hemo))
###Non hemo
N_non_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_non_hemo]
C_non_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
#records_non_hemo=np.hstack((records_non_hemo,NC_non_hemo))
NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
####
ELMO_hemo_dict=dict(zip(Names_hemo,records_hemo))
ELMO_non_hemo_dict=dict(zip(Names_non_hemo,records_non_hemo))
####
Hemo_Dict=dict(zip(Names_hemo,np.hstack((list(ELMO_hemo_dict.values()),list(NC_hemo_dict.values())))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((list(ELMO_non_hemo_dict.values()),list(NC_non_hemo_dict.values())))))
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
Featurename='ELMO_Smile'
Classifier='NN'
runs=10
print("5-fold")
ResultMeanStd_5fold(runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost

UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
percent='90'
print(Classifier,"\n",percent)
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)
print("Total feature dimension",len(Features[0]),Featurename)
percent='70'
print(Classifier,"\n",percent)
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)

