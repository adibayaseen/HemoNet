#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:11:33 2019

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
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
def Results(Features,Label,kk,cc,gg):
    Roc_VA,Y_score,Y_t=[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in cv.split(Features,Label):
        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
        svr_lin=SVC(kernel=kk, C=cc,gamma=gg)
        svr_lin.fit(X_train, y_train)
        test_score=svr_lin.decision_function(X_test)
        Y_score.extend(test_score)
        Y_t.extend(y_test)
        Roc_VA.append((test_score,list(y_test)))
        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
    auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#    print("auc_roc",auc_roc)
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#    print("avgmean",avgmean)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
    plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(auc_roc))
    plt.plot(avgfpr, avgtpr, color='b',marker='.',label='AUC_avgmean= {:.2f}'.format(avgmean))
    plt.legend(loc='lower right')
    plt.grid()
    plt.show() 
    fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr-threshold<0.01)])
    Senstivity_HELMO=np.max(tpr[np.where(fpr-threshold<0.01)])
    MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO,threshold, len(records_hemo),len(records_non_hemo))
    #Senstivity,specifity, MCC,AUCROC
    return Senstivity_HELMO,specificity,MCC_HELMO ,avgmean,average_precision_score(np.array(Y_t), np.array(Y_score))
def ResultMeanStd(Features,Label,kk,cc,gg):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(Features,Label,kk,cc,gg)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n")
################
path="D:\PhD\Hemo_All_SeQ/"
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
ELMO_Hemo_Dict=dict(zip(Names_hemo,records_hemo))
ELMO_Non_hemo_Dict=dict(zip(Names_non_hemo,records_non_hemo))
#####
N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
N_hemo=[N_terminous[int(n)] for n in Names_hemo]
C_hemo=[C_terminous[int(n)] for n in Names_hemo]
NC_hemo=np.hstack((N_hemo,C_hemo))
###Non hemo
N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
###
NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
Hemo_Dict=dict(zip(Names_hemo,np.hstack((list(ELMO_Hemo_Dict.values()),list(NC_hemo_dict.values())))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((list(ELMO_Non_hemo_Dict.values()),list(NC_non_hemo_dict.values())))))
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
#####
Featurename='ELMO_NC'
#1/0
runs=10
#Classifier=RandomForestClassifier
#ResultMeanStd_5fold(runs,Classifier,Features,Label,90,20,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
#percent='90'
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#print(Classifier,"\n",percent)
####XGB parameters
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,10,170,Featurename,Hemo_Dict,Non_hemo_Dict)
#percent='70'
#print(Classifier,"\n",percent)
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,1,170,Featurename,Hemo_Dict,Non_hemo_Dict)
#print("Total feature dimension",len(Features[0]),Featurename)
#Classifier=XGBClassifier
#ResultMeanStd_5fold(runs,Classifier,Features,Label,135,5,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
#percent='90'
#print(Classifier,"\n",percent)
####XGB parameters
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,5,Featurename,Hemo_Dict,Non_hemo_Dict)
#percent='70'
#print(Classifier,"\n",percent)
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,1,Featurename,Hemo_Dict,Non_hemo_Dict)
#print("Total feature dimension",len(Features[0]),Featurename)
#Features=F.normalize(Features, p=1, dim=1)
#Classifier=SVC
#print(Classifier)
#ResultMeanStd_SVM_5fold(runs,Classifier,Features,Label,'rbf',4,512,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
#print("Total feature dimension",len(Features[0]),Featurename)
#percent='90'
#print(Classifier,"\n",percent)
#ResultMeanStd_SVM_NR_fold(percent,UNames,runs,Classifier,Features,Label,'linear',1,0 ,Featurename,Hemo_Dict,Non_hemo_Dict)
#percent='70'
#print(Classifier,"\n",percent)
#ResultMeanStd_SVM_NR_fold(percent,UNames,runs,Classifier,Features,Label,'rbf',1,1e-05 ,Featurename,Hemo_Dict,Non_hemo_Dict)
####Neural network
Classifier='NN'
#runs=1
print("5-fold")
ResultMeanStd_5fold(runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
percent='90'
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
print(Classifier,"\n",percent)
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)
print("Total feature dimension",len(Features[0]),Featurename)
percent='70'
print(Classifier,"\n",percent)
ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)
1/0

1/0
cv = StratifiedKFold(n_splits=5, shuffle=True)
#1/0
C=[4]#[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192]
Gamma=[512]#[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
#Gamma=[1,2,4,8,16,32,64,128,256,512,1024,2056]
pre_roc,cc,gg=0,0,0
###Roc= 0.8768712520886215 c= 4 g= 512 Average Mean= 0.8773268609081484 kernel= rbf
#"""
#For Linear SVC
#"""
#for c in C:
#    Y_score,Y_t,Roc_VA=[],[],[]
#    print("c",c)
#    for train_index, test_index in cv.split(Features,Label):
#        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#        svr_lin=SVC(kernel='linear', C=c)
#        svr_lin.fit(X_train, y_train)
#        test_score=svr_lin.decision_function(X_test)
#        Y_score.extend(test_score)
#        Y_t.extend(y_test)
#        Roc_VA.append((test_score,list(y_test)))
#    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#    auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#    print("auc_roc",auc_roc,"c=",c)
#    if pre_roc<auc_roc:
#       print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",c,"Average Mean=",avgmean)
#       pre_roc=auc_roc
#       cc=c
#"""
#For RBF kernel SVC
#"""
#for i in C:
#    for j in Gamma:
#        Y_score,Y_t=[],[]
#        for train_index, test_index in cv.split(Features,Label):
#            Roc_VA=[]
#            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#    #        svr_lin=XGBClassifier(max_depth=3, learning_rate=0.1)
#            svr_lin=SVC(kernel='rbf', C=i,gamma=j)
##                svr_lin=SVC(kernel='linear', C=i)
#            svr_lin.fit(X_train, y_train)
#            test_score=svr_lin.decision_function(X_test)
##            Y_p=svr_lin.predict(X_test)
#            Y_score.extend(test_score)
#            Y_t.extend(y_test)
##            Y_pred.extend(Y_p) 
#            Roc_VA.append((test_score,list(y_test)))
#        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#        print("auc_roc",auc_roc,"c=",i,"g=",j)
#        if pre_roc<auc_roc:
#           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",i,"g=",j,"Average Mean=",avgmean)
#           pre_roc=auc_roc
#           cc=i
#           gg=j
"""
For Both Linear and RBF kernel 
"""
K=['linear','rbf']
for k in K:
    if k=='rbf':
        for i in C:
            for j in Gamma:
                Y_score,Y_t,Roc_VA=[],[],[]
                for train_index, test_index in cv.split(Features,Label):
                    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
                    svr_lin=SVC(kernel=k, C=i,gamma=j)
                    svr_lin.fit(X_train, y_train)
                    test_score=svr_lin.decision_function(X_test)
                    Y_score.extend(test_score)
                    Y_t.extend(y_test)
                    Roc_VA.append((test_score,list(y_test)))
                avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
                auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
                print("auc_roc",auc_roc,"c=",i,"g=",j)
                if pre_roc<auc_roc:
                   print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",i,"g=",j,"Average Mean=",avgmean,"kernel=",k)
                   pre_roc=auc_roc
                   cc=i
                   gg=j
                   kk=k
    elif k=='linear':
        for i in C:
            Y_score,Y_t,Roc_VA=[],[],[]
            for train_index, test_index in cv.split(Features,Label):
                X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
                svr_lin=SVC(kernel='linear', C=i)
                svr_lin.fit(X_train, y_train)
                test_score=svr_lin.decision_function(X_test)
                Y_score.extend(test_score)
                Y_t.extend(y_test)
                Roc_VA.append((test_score,list(y_test)))
            avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
            auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
            print("auc_roc",auc_roc,"c=",i,"g=",j)
            if pre_roc<auc_roc:
               print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",i,"g=",j,"Average Mean=",avgmean)
               pre_roc=auc_roc
               cc=i    