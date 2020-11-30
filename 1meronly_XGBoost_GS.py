#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
#records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
#1HotEncoding
N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
N_hemo=[N_terminous[int(n)] for n in Names_hemo]
C_hemo=[C_terminous[int(n)] for n in Names_hemo]
NC_hemo=np.hstack((N_hemo,C_hemo))
#nonhemo
N_nonhemo=[N_terminous[int(n)] for n in Names_non_hemo]
C_nonhemo=[C_terminous[int(n)] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_nonhemo,C_nonhemo))
NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
oneHot_features=np.vstack((list(NC_hemo_dict.values()),list(NC_non_hemo_dict.values())))
#NC Smiles Features
#"""
#N_terminous_names=pickle.load(open(path+'nTerminus_All_Dict.npy', "rb"))
#C_terminous_names=pickle.load(open(path+'cTerminus_All_Dict.npy', "rb"))
######
#dimension=1024
#print("Feature Dimension",dimension)
#N_terminous_Smiles_features=pickle.load(open(path+'Nmod_Dict_'+str(dimension)+'.npy', "rb"))
#C_terminous_Smiles_features=pickle.load(open(path+'Cmod_Dict_'+str(dimension)+'.npy', "rb"))
####
#N_terminous_Smiles_features['Free']=' ',np.zeros(len(list(N_terminous_Smiles_features.values())[0][1]))
#C_terminous_Smiles_features['Free']=' ',np.zeros(len(list(C_terminous_Smiles_features.values())[0][1]))
#N_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_hemo]
#C_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1]for n in Names_hemo]
#NC_hemo=np.hstack((N_hemo,C_hemo))
#NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
##NC_hemo=(NC_hemo-np.mean(NC_hemo, axis = 0))/(np.std(NC_hemo, axis = 0)+0.000001)
##records_hemo=(records_hemo-np.mean(records_hemo, axis = 0))/(np.std(records_hemo, axis = 0)+0.000001)
##records_hemo=np.hstack((records_hemo,NC_hemo))
###Non hemo
#N_non_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_non_hemo]
#C_non_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1] for n in Names_non_hemo]
##N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
##C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
#NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
#NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
##NC_features=np.vstack((NC_hemo,NC_non_hemo))
#NC_features=np.vstack((list(NC_hemo_dict.values()),list(NC_non_hemo_dict.values())))
#NC_features=(NC_features-np.mean(NC_features, axis = 0))/(np.std(NC_features, axis = 0)+0.000001)
####
###AAC Features
mer=1
print("AAC features",mer,"mer")
Hemo_Dict=All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',mer)
Non_hemo_Dict=  All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',mer)
AAC_Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
AAC_Features=(AAC_Features-np.mean(AAC_Features, axis = 0))/(np.std(AAC_Features, axis = 0)+0.000001)
#Features=np.hstack((AAC_Features,NC_features))

###############
Features=np.hstack((AAC_Features,oneHot_features))
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
Classifier=RandomForestClassifier
runs=10
Featurename='AAC'
ResultMeanStd_5fold(runs,Classifier,Features,Label,90,20,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
print("Total feature dimension",len(Features[0]),Featurename)
1/0
##path='D:/Downloads/'
#Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[170]#[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
Depth=[170]#[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
cv = StratifiedKFold(n_splits=5, shuffle=True)
pre_roc,cc,gg,ee,dd=0,0,0,0,0

"""
XGboost
"""
cv = StratifiedKFold(n_splits=5, shuffle=True)
for d in Depth:
    for e in Estimator:
        for train_index, test_index in cv.split(Features,Label):
           X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
           svr_lin = XGBClassifier(max_depth=d, n_estimators=e)
           svr_lin.fit(X_train, y_train)
           test_score=svr_lin.predict_proba(X_test)[:,1]
           Y_score.extend(test_score)
           Y_t.extend(y_test)
           Roc_VA.append((test_score,list(y_test)))
        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
        print("auc_roc",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
        if pre_roc<avgmean:
           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
           pre_roc=avgmean
           ee=e
           dd=d
#################
Roc_VA,Y_score,Y_t=[],[],[]
for train_index, test_index in cv.split(Features,Label):
   X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
   svr_lin = XGBClassifier(max_depth=dd, n_estimators=ee)
   svr_lin.fit(X_train, y_train)
   test_score=svr_lin.predict_proba(X_test)[:,1]
   Y_score.extend(test_score)
   Y_t.extend(y_test)
   Roc_VA.append((test_score,list(y_test)))
   avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
   auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("avgmean",avgmean)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUCXGboost= {:.2f}'.format(auc_roc))
plt.grid()
plt.figure()
plt.plot(avgfpr, avgtpr, color='b',marker='.',label='AUC_avgmeanXGboost= {:.2f}'.format(avgmean))
plt.legend(loc='lower right')
plt.grid()
plt.show()   
threshold=0.386
specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemo_Dict),len(Non_hemo_Dict))
print("senstivity specificity MCC,AUCROC and PR of OUR model",Senstivity,specificity, MCC,avgmean,average_precision_score(np.array(Y_t), np.array(Y_score)))
#####
mer=1
print("External AAC features",mer,"mer")
External_AAC=list( All_FeaturesWithoutNC(path,path+'External.txt',mer).values())
External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
External_features=External_AAC
External_features=External_AAC
#External_features=np.vstack((External_f[:2],External_f[4:6]))
#External_features=np.vstack((External_features,External_f[8:12]))
#External_features=np.vstack((External_features,External_f[15]))
#External_features=np.vstack((External_features,External_f[17]))
#External_features=np.vstack((External_features,External_f[18:20]))
#External_features=np.vstack((External_features,External_f[21:]))
#External_score =svr_lin.predict_proba(External_features)[:,1]
print("External_score",External_score)
External_label=np.append(np.zeros(8),np.ones(7))
#External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_90r)
External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation 5-fold:{: .2f}'.format(External_auc_roc_90r))
#plt.grid()
plt.legend(loc='lower right')
plt.grid()
######DRAMP Clinical data

mer=1
print("External AAC features",mer,"mer")
DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
DRAMP_Clinical_data_AAC=(DRAMP_Clinical_data_AAC-np.mean(DRAMP_Clinical_data_AAC, axis = 0))/(np.std(DRAMP_Clinical_data_AAC, axis = 0)+0.000001)
DRAMP_Clinical_data_AAC_score=svr_lin.predict_proba(DRAMP_Clinical_data_AAC)[:,1]
plt.figure()
plt.hist(np.sort(DRAMP_Clinical_data_AAC_score),bins=len(DRAMP_Clinical_data_AAC_score))
plt.grid()
print("Average score From our model",np.mean(DRAMP_Clinical_data_AAC_score))
#####DRAMP Predicted from Hemopred
HemoPred_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPred_predicted_DRAMP_Clinical_data.csv')
Score_HemoPred_predicted_DRAMP_Clinical_data=HemoPred_predicted_DRAMP_Clinical_data[['Prediction']].values
plt.figure()
plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data)
print("Average score HemoPred",np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data))
plt.grid()
####DRAMP HEmoPI predicted results
HemoPI_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPI_predicted_DRAMP_Clinical_data.csv')
Score_HemoPI_predicted_DRAMP_Clinical_data=HemoPI_predicted_DRAMP_Clinical_data[['PROB Score']].values
plt.figure()
plt.hist(Score_HemoPI_predicted_DRAMP_Clinical_data)
print("Average score HemoPI",np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data))
plt.grid()
####HaPPeNN
####DRAMP HEmoPI predicted results
HaPPeNN_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HaPPeNN_predicted_DRAMP_Clinical_data.csv')
Score_HaPPeNN_predicted_DRAMP_Clinical_data=HaPPeNN_predicted_DRAMP_Clinical_data[['PROB']].values
plt.figure()
plt.hist(Score_HaPPeNN_predicted_DRAMP_Clinical_data)
print("Average score HaPPeNN total 21/28",np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data))

plt.grid()
plt.figure()
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
plt.hist(DRAMP_Clinical_data_AAC_score,density=True, color='lightsteelblue',label='Our method:{: .2f}'.format(np.mean(DRAMP_Clinical_data_AAC_score)))
plt.grid()
plt.legend(loc='upper center')
plt.savefig('Ourmodel_ClinicalData.png', dpi=300)
plt.figure()
plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data, density=True,
 linestyle='--',color='darkseagreen', linewidth=3,label='HemoPred:{: .2f}'.format(np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data)))
plt.grid()
plt.legend(loc='upper center')
plt.savefig('HemoPrd_ClinicalData.png', dpi=300)
plt.figure()
plt.hist(Score_HemoPI_predicted_DRAMP_Clinical_data, density=True,
linestyle='solid',linewidth=3,label='HemoPI:{: .2f}'.format(np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data)))
plt.grid()
plt.legend(loc='upper right')
plt.savefig('HemoPI_ClinicalData.png', dpi=300)
plt.figure()
plt.hist(Score_HaPPeNN_predicted_DRAMP_Clinical_data, density=True,
linestyle='-', color='gray',linewidth=3,label='HaPPeNN :{: .2f}'.format(np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data)))
plt.legend(loc='upper center')
#plt.legend(frameon=False)
plt.show()
plt.grid()
plt.savefig('HaPPeNN_ClinicalData.png', dpi=300)
#plt.savefig(allplotfile)
plt.close()
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')