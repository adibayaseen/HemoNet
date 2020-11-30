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
from Results import *
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
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
Hemo_Dict=dict(zip(Names_hemo,records_hemo))
Non_hemo_Dict=dict(zip(Names_non_hemo,records_non_hemo))
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Label=np.append(np.ones(len(Hemo_Dict)),np.zeros(len(Non_hemo_Dict)))
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
####
Featurename='ELMO'
Classifier= XGBClassifier#RandomForestClassifier
runs=10
ResultMeanStd_5fold(runs,Classifier,Features,Label,170,170,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
print("Total feature dimension",len(Features[0]),Featurename)
percent='90'
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
print(Classifier,"\n",percent)
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,200,40,Featurename,Hemo_Dict,Non_hemo_Dict)
#percent='70'
#print(Classifier,"\n",percent)
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,5,Featurename,Hemo_Dict,Non_hemo_Dict)
print("Total feature dimension",len(Features[0]),Featurename)
1/0
#Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[170]#[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
Depth=[170]#[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
#ELMO best results Roc= 0.685580142973737 e= 135 d= 6 Average Mean= 0.6893453668904338
#1mer Roc= 0.6722432260758173 e= 135 d= 135 Average Mean= 0.6764716431715094#1mer Roc= 0.6814864124507055 e= 170 d= 155 Average Mean= 0.6822134531724315

####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
#cv = StratifiedKFold(n_splits=5, shuffle=True)
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
External_AAC=list( All_FeaturesWithoutNC(path,path+'Hemolytic_External_validation.txt',mer).values())
External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
External_features=External_AAC
External_f=External_AAC
External_features=np.vstack((External_f[:2],External_f[4:6]))
External_features=np.vstack((External_features,External_f[8:12]))
External_features=np.vstack((External_features,External_f[15]))
External_features=np.vstack((External_features,External_f[17]))
External_features=np.vstack((External_features,External_f[18:20]))
External_features=np.vstack((External_features,External_f[21:]))
External_score =svr_lin.predict_proba(External_features)[:,1]
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
ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
ELMO_DRAMP_Clinical_data_Features=(ELMO_DRAMP_Clinical_data_Features-np.mean(ELMO_DRAMP_Clinical_data_Features, axis = 0))/(np.std(ELMO_DRAMP_Clinical_data_Features, axis = 0)+0.000001)
#ELMO_DRAMP_Clinical_data_Names=np.load(path+'DRAMP_Clinical_data_Names.npy')
DRAMP_Clinical_data_AAC_score=svr_lin.predict_proba(ELMO_DRAMP_Clinical_data_Features)[:,1]
plt.figure()
plt.hist(np.sort(DRAMP_Clinical_data_AAC_score),bins=len(DRAMP_Clinical_data_AAC_score))
plt.grid()
print("Average score",np.mean(DRAMP_Clinical_data_AAC_score))
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