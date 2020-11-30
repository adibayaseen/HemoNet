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
#from  Clusterify import *
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

#N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
#C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
#N_hemo=[N_terminous[int(n)] for n in Names_hemo]
#C_hemo=[C_terminous[int(n)] for n in Names_hemo]
#NC_hemo=np.hstack((N_hemo,C_hemo))
"""
NC Smiles Features
"""
N_terminous_names=pickle.load(open(path+'nTerminus_All_Dict.npy', "rb"))
C_terminous_names=pickle.load(open(path+'cTerminus_All_Dict.npy', "rb"))
#####
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
NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
#NC_hemo=(NC_hemo-np.mean(NC_hemo, axis = 0))/(np.std(NC_hemo, axis = 0)+0.000001)
#records_hemo=(records_hemo-np.mean(records_hemo, axis = 0))/(np.std(records_hemo, axis = 0)+0.000001)
#records_hemo=np.hstack((records_hemo,NC_hemo))
##Non hemo
N_non_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_non_hemo]
C_non_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1] for n in Names_non_hemo]
#N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
#C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
#NC_features=np.vstack((NC_hemo,NC_non_hemo))
NC_features=np.vstack((list(NC_hemo_dict.values()),list(NC_non_hemo_dict.values())))
NC_features=(NC_features-np.mean(NC_features, axis = 0))/(np.std(NC_features, axis = 0)+0.000001)
####
###AAC Features
mer=1
print("AAC features",mer,"mer")
Hemo_Dict=dict(zip(Names_hemo,np.hstack((list(All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',mer).values()),list(NC_hemo_dict.values())))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((list(All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',mer).values()),list(NC_non_hemo_dict.values())))))
#1/0
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
AAC_Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
AAC_Features=(AAC_Features-np.mean(AAC_Features, axis = 0))/(np.std(AAC_Features, axis = 0)+0.000001)
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
#Hemo_Dict=dict(zip(Names_hemo,Hemo_Dict))
#Non_hemo_Dict=dict(zip(Names_non_hemo,Non_hemo_Dict))
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
Features=torch.FloatTensor(Features)
##
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Without Normalization of features Execution Completed")
Featurename='AAC_Smile'
Classifier=RandomForestClassifier
runs=10
percent='90'
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,200,150,Featurename,Hemo_Dict,Non_hemo_Dict)
ResultMeanStd_5fold(runs,Classifier,Features,Label,150,30,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
print("Total feature dimension",len(Features[0]),Featurename)
1/0
###Total 10 times mean std  
##ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
##ResultMeanStd(Features,Label,135,5)#ELMO
#auc_roc 0.8670257770063259 e= 70 d= 120 Average Mean= 0.8797813034346526
#ResultMeanStd(Features,Label,135,135)#ELMO
###auc_roc 0.8644185446662404 e= 135 d= 135 Average Mean= 0.880763134676796
#1/0
print("Without 1 norm")
#Roc= 0.8830212585195861 e= 100 d= 5 Average Mean= 0.8980306401051673
cv = StratifiedKFold(n_splits=5, shuffle=True)
#C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192,16384,32768,65535]
#Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
Depth=[135,140,145,150,155,160,165,170,175,180]
#Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[135,140,145,150,155,160,165,170,175,180]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
###
"""
10 runs results
L=[0.866,0.866,0.867,0.871,0.867,0.872,0.87,0.867,0.868,0.87]
L=np.array(L)
print(np.mean(L),np.std(L))
"""
###Best parameters
#Depth=[6]#XGBoost
#Estimator=[120]
pre_roc,cc,gg,ee,dd=0,0,0,0,0
"""
XGboost
"""
#for d in Depth:
#   for e in Estimator:
#        Y_score,Y_t=[],[]
#        for train_index, test_index in cv.split(Features,Label):
#            Roc_VA=[]
#            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#            svr_lin = XGBClassifier(learning_rate=0.1,max_depth=d, n_estimators=e)
##            svr_lin = RandomForestClassifier(n_estimators=e, max_depth=d,random_state=0)
#            svr_lin.fit(X_train, y_train)
##            Y_p=svr_lin.predict(X_test)
#            test_score=svr_lin.predict_proba(X_test)[:,1]
#            Y_score.extend(test_score)
#            Y_t.extend(y_test)
##            Y_pred.extend(Y_p) 
#            Roc_VA.append((test_score,list(y_test)))
#        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#        print("auc_roc",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
#        if pre_roc<auc_roc:
#           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
#           pre_roc=auc_roc
#           ee=e
#           dd=d
##################
#ee=130
#dd=60
#Roc= 0.8725546391103853 e= 145 d= 10 Average Mean= 0.8754423640558743
ee=90
dd=10
Roc_VA,Y_score,Y_t=[],[],[]
cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(Features,Label):
    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#    svr_lin = RandomForestClassifier(n_estimators=135, max_depth=30,random_state=0)
    svr_lin = XGBClassifier(max_depth=dd, n_estimators=ee)
#    svr_lin = XGBClassifier(learning_rate=0.1,max_depth=dd, n_estimators=ee)
    svr_lin.fit(X_train, y_train)
    test_score=svr_lin.predict_proba(X_test)[:,1]
    Y_score.extend(test_score)
    Y_t.extend(y_test)
    Roc_VA.append((test_score,list(y_test)))
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
MCC=MCC_fromAUCROC(Senstivity, threshold, len(records_hemo),len(records_non_hemo))
print("senstivity specificity MCC,AUCROC and PR of OUR model",Senstivity,specificity, MCC,avgmean,average_precision_score(np.array(Y_t), np.array(Y_score)))
#####
###External testing
path='D:/PhD/Hemo_All_SeQ/'
External=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
mer=1
print("External AAC features",mer,"mer")
External_AAC=list( All_FeaturesWithoutNC(path,path+'Hemolytic_External_validation.txt',mer).values())
External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
ELmo_External_features=External[:,0:1024]
#####
External_NC=np.zeros((24,2048))
External_Features=np.hstack((ELmo_External_features,External_NC))
External_f=np.hstack((External_Features,External_AAC))
###
External_features=np.vstack((External_f[:2],External_f[4:6]))
External_features=np.vstack((External_features,External_f[8:12]))
External_features=np.vstack((External_features,External_f[15]))
External_features=np.vstack((External_features,External_f[17]))
External_features=np.vstack((External_features,External_f[18:20]))
External_features=np.vstack((External_features,External_f[21:]))
####1mer+ELMO
#External_Features=np.hstack((ELmo_External_features,External_AAC))
External_score =svr_lin.predict_proba(External_features)[:,1]
print("External_score",External_score)
External_label=np.append(np.zeros(8),np.ones(7))
#External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_90r)
External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation 5-fold:{: .2f}'.format(External_auc_roc_90r))
plt.legend(loc='lower right')
plt.grid()
####Clinical using ELMO
ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
DRAMP_Clinical_NC=np.zeros((28,2048))
ALL_DRAMP_Clinical_data_Features=np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),DRAMP_Clinical_data_AAC))
ALL_DRAMP_Clinical_data_score=svr_lin.predict_proba(ALL_DRAMP_Clinical_data_Features)[:,1]
plt.figure()
plt.hist(np.sort(ALL_DRAMP_Clinical_data_score),bins=len(ALL_DRAMP_Clinical_data_score))
plt.grid()
print("Average score ALL features",np.mean(ALL_DRAMP_Clinical_data_score))
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