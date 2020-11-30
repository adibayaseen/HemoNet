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
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
def Results(Features,Label,e,d):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in cv.split(Features,Label):
        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
    #    svr_lin = RandomForestClassifier(n_estimators=120, max_depth=20,random_state=0)
        svr_lin = XGBClassifier(max_depth=d, n_estimators=e)
    #    svr_lin = XGBClassifier(learning_rate=0.1,max_depth=dd, n_estimators=ee)
        svr_lin.fit(X_train, y_train)
        test_score=svr_lin.predict_proba(X_test)[:,1]
        Y_p.extend(test_score)
        Y_test.extend(y_test)
        Roc_VA.append((test_score,list(y_test)))
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
#    average_precision_score=average_precision_score(np.array(Y_test), np.array(Y_p))
#    print("auc_roc",auc_roc)
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#    print("Average_mean",avgmean)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
    plt.plot(fpr, tpr, color='c',marker=',',label='AUC_avgmeanXGboost= {:.2f}'.format(auc_roc))
#    plt.grid()
    plt.legend(loc='lower right')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(records_hemo),len(records_non_hemo))
#    import pdb;pdb.set_trace()
    
#    print("MCC of OUR model",MCC_HELMO)
#    print("Senstivity of our model",Senstivity_HELMO)
    #Senstivity,specifity, MCC,AUCROC
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p))
def ResultMeanStd(Features,Label,e,d):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(Features,Label,e,d)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n")
"""
resulrs of 10 runs
AUC-ROC
L=[0.869,0.868,0.874,0.866,0.865,0.870,0.87,0.866,0.871,0.866]
L=np.array(L)
np.mean(L) 0.868
np.std(L) 0.0026
"""
path="D:\PhD\Hemo_All_SeQ/"
#####ELMO Features
#records_hemo=np.load(path+'new_Hemo_Features.npy')
#records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
#Names_hemo=np.load(path+'new_Hemo_Names.npy')
#Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
#Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
#Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
#N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
#C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
#N_hemo=[N_terminous[int(n)] for n in Names_hemo]
#C_hemo=[C_terminous[int(n)] for n in Names_hemo]
#NC_hemo=np.hstack((N_hemo,C_hemo))
#records_hemo=np.hstack((records_hemo,NC_hemo))
####Non hemo
#N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
#C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
#NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
#records_non_hemo=np.hstack((records_non_hemo,NC_non_hemo))
#Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
##Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
#Features=np.vstack((records_hemo,records_non_hemo))
#Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
##1mer best parameters Roc= 0.86 e= 120 d= 11 Average Mean= 0.86
##2mer best parameters Roc= 0.8698190622187595 e= 130 d= 15 Average Mean= 0.8741186942230065
##ELMO best parameters  Roc= 0.8817441496967066 e= 135 d= 5 Average Mean= 0.8908726145934155
#####AAC Features
mer=1
print("AAC features",mer,"mer")
records_hemo=list(All_Features(path,path+'hemo_All_seq.txt',mer).values())
records_non_hemo= list(All_Features(path,path+'Nonhemo_All_seq.txt',mer).values())
Label=np.append(np.ones(len(records_hemo)),np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
##
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
##Total 10 times mean std  
ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
#ResultMeanStd(Features,Label,135,5)#ELMO
1/0
print("Without 1 norm")
cv = StratifiedKFold(n_splits=5, shuffle=True)
#C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192,16384,32768,65535]
#Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
#Depth=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
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
for d in Depth:
   for e in Estimator:
        Y_score,Y_t=[],[]
        for train_index, test_index in cv.split(Features,Label):
            Roc_VA=[]
            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
            svr_lin = XGBClassifier(learning_rate=0.1,max_depth=d, n_estimators=e)
#            svr_lin = RandomForestClassifier(n_estimators=e, max_depth=d,random_state=0)
            svr_lin.fit(X_train, y_train)
#            Y_p=svr_lin.predict(X_test)
            test_score=svr_lin.predict_proba(X_test)[:,1]
            Y_score.extend(test_score)
            Y_t.extend(y_test)
#            Y_pred.extend(Y_p) 
            Roc_VA.append((test_score,list(y_test)))
        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
        print("auc_roc",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
        if pre_roc<auc_roc:
           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
           pre_roc=auc_roc
           ee=e
           dd=d
#################
Roc_VA,Y_score,Y_t=[],[],[]
cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(Features,Label):
    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#    svr_lin = RandomForestClassifier(n_estimators=120, max_depth=20,random_state=0)
    svr_lin = XGBClassifier(max_depth=15, n_estimators=130)
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
#fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
#plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(auc_roc))
#plt.plot(avgfpr, avgtpr, color='b',marker='.',label='AUC_avgmean= {:.2f}'.format(avgmean))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show() 
fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
#ELMO_accuracy=Best_accuracy(Y_t, Y_score)
#print("ELMO_Accuracy=",ELMO_accuracy)
#plt.plot(fpr, tpr, color='c',marker=',',label='SVM:{: .2f}'.format(auc_roc))
#plt.scatter(fpr, tpr, color='c',marker=',',label='Hemo_with_ELMO= {:.2f}'.format(auc_roc))
#plt.legend(loc='lower right')
Senstivity_HELMO_XGboost=np.max(tpr[np.where(fpr-0.29<0.01)])
#plt.grid()
#plt.show()
MCC_HELMO_XGboost=MCC_fromAUCROC(Senstivity_HELMO_XGboost,np.max(fpr[np.where(fpr-0.29<0.01)]), len(records_hemo),len(records_non_hemo))
print("MCC of OUR model",MCC_HELMO_XGboost)
print("Senstivity of our model",Senstivity_HELMO_XGboost)