# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:58:36 2020

@author: 92340
"""

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
def Results(kk,cc,gg,path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Y_test, Y_p,Roc_VA=[],[],[]
    if kk=='linear':
        for i in range(5):
            X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
            svr_lin=SVC(kernel='linear', C=cc)
            svr_lin.fit(X_train, y_train)
            test_score=svr_lin.decision_function(X_test)
            Y_p.extend(test_score)
            Y_test.extend(y_test)
            Roc_VA.append((test_score,list(y_test)))
    elif kk=='rbf':
        for i in range(5):
            X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
            svr_lin=SVC(kernel=kk, C=cc,gamma=gg)
            svr_lin.fit(X_train, y_train)
            test_score=svr_lin.decision_function(X_test)
            Y_p.extend(test_score)
            Y_test.extend(y_test)
            Roc_VA.append((test_score,list(y_test)))
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
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
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p))
def ResultMeanStd(kk,cc,gg,path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(kk,cc,gg,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n")
def RedendencyRemoval(fold,path,UNames,hemo_file,nonhemo_file,Hemo_Dict,Non_hemo_Dict):
    #################90 new %
    hemo_CL=Make_Cluster(UNames,path,hemo_file)
    Non_hemo_CL=Make_Cluster(UNames,path,nonhemo_file)
    hemo_Folds=chunkify(hemo_CL)
    Non_hemo_Folds=chunkify(Non_hemo_CL)
    ####
    for i in range(5):
           X_train= np.array([], dtype=np.int64).reshape(0,len(Features[0]))
           X_test, y_train, y_test=[],[],[]
           train_len=0
           for idx, pbag in enumerate(hemo_Folds):
               if fold==idx:
                   nbag=Non_hemo_Folds[idx]
                #####test data##########
                   nfeatures=name2feature(nbag,Non_hemo_Dict)
                   pfeatures=name2feature(pbag,Hemo_Dict)
                   X_test=np.vstack((pfeatures,nfeatures))
                   y_test=np.append(np.ones(len(pfeatures)),-1*np.ones(len(nfeatures)))
               else:
                    nbag=Non_hemo_Folds[idx]
                #####test data##########
                    nfeatures=name2feature(nbag,Non_hemo_Dict)
                    pfeatures=name2feature(pbag,Hemo_Dict)
                    train_features=np.vstack((pfeatures,nfeatures))
                    train_label=np.append(np.ones(len(pfeatures)),-1*np.ones(len(nfeatures)))
                    X_train=np.vstack((X_train,train_features))
                    y_train=np.append(y_train,train_label)
    return X_train,X_test, y_train, y_test
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
#Hemo_Dict,Non_hemo_Dict={},{}
#for n in range(len( Names_hemo)):
#    Hemo_Dict[Names_hemo[n]]=records_hemo[n]
#for n in range(len( Names_non_hemo)):
#    Non_hemo_Dict[Names_non_hemo[n]]=records_non_hemo[n]
#Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
##Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
#Features=np.vstack((records_hemo,records_non_hemo))
#Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
###1mer best parameters Roc= 0.86 e= 120 d= 11 Average Mean= 0.86
###2mer best parameters Roc= 0.8698190622187595 e= 130 d= 15 Average Mean= 0.8741186942230065
###ELMO best parameters  Roc= 0.8817441496967066 e= 135 d= 5 Average Mean= 0.8908726145934155
#####AAC Features
mer=2
print("AAC features",mer,"mer")
Hemo_Dict=All_Features(path,path+'hemo_All_seq.txt',mer)
Non_hemo_Dict=All_Features(path,path+'Nonhemo_All_seq.txt',mer)
records_hemo=list(Hemo_Dict.values())
records_non_hemo= list(Non_hemo_Dict.values())
Label=np.append(np.ones(len(records_hemo)),np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
Features=F.normalize(Features, p=1, dim=1)
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
percent='70'#'90'
##ResultMeanStd('linear',1,00,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#90%ELMO
##ResultMeanStd('rbf',1,1e-05 ,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#70%ELMO
#Roc= 0.6669183256685723 c= 128 Average Mean= 0.6656726997905217#90_1mer
#ResultMeanStd('linear',128,00 ,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
#Roc= 0.6196755257681593 c= 100 Average Mean= 0.6203563732721998#70_1mer
#ResultMeanStd('linear',100,00 ,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
#Roc= 0.6417168027377055 c= 64 Average Mean= 0.6408218483437891#2mer_90
#ResultMeanStd('linear',64,00 ,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
#Roc= 0.5564058690617675 c= 100 Average Mean= 0.5528408120212803
ResultMeanStd('linear',100,00 ,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#2mer_70
#ResultMeanStd(Features,Label,120,11)#1mer
1/0
#Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
#print ("Execution Completed")
###Total 10 times mean std  
##ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
##ResultMeanStd(Features,Label,135,5)#ELMO
#1/0
###
#UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#################90 new %
C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192]
Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
#cv = StratifiedKFold(n_splits=5, shuffle=True)
pre_roc,cc,gg,ee,dd=0,0,0,0,0
#Roc= 0.7142067794380845 c= 1 Average Mean= 0.7135302772565445
"""
For Both Linear and RBF kernel 
"""

K=['linear','rbf']
for k in K:
    if k=='rbf':
        for c in C:
            for g in Gamma:
                 for i in range(5):
                    X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
                    svr_lin=SVC(kernel=k, C=c,gamma=g)
                    svr_lin.fit(X_train, y_train)
                    test_score=svr_lin.decision_function(X_test)
                    Y_score.extend(test_score)
                    Y_t.extend(y_test)
                    Roc_VA.append((test_score,list(y_test)))
                 avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
                 auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
                 print("auc_roc",auc_roc,"c=",c,"g=",g)
                 if pre_roc<auc_roc:
                   print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",c,"g=",g,"Average Mean=",avgmean,"kernel=",k)
                   pre_roc=auc_roc
                   cc=c
                   gg=g
                   kk=k
    elif k=='linear':
        for c in C:
            Y_score,Y_t,Roc_VA=[],[],[]
            for i in range(5):
                X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+pecent+'.txt','new_Nonhemo_'+pecent+'.txt',Hemo_Dict,Non_hemo_Dict)
                svr_lin=SVC(kernel='linear', C=c)
                svr_lin.fit(X_train, y_train)
                test_score=svr_lin.decision_function(X_test)
                Y_score.extend(test_score)
                Y_t.extend(y_test)
                Roc_VA.append((test_score,list(y_test)))
            avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
            auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
            print("auc_roc",auc_roc,"c=",c)
            if pre_roc<auc_roc:
               print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",c,"Average Mean=",avgmean)
               pre_roc=auc_roc
               cc=c