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
####ELMO Features
records_hemo=np.load(path+'new_Hemo_Features.npy')
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
#records_hemo=np.hstack((records_hemo,NC_hemo))
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
###Total 10 times mean std  
##ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
##ResultMeanStd(Features,Label,135,5)#ELMO
Featurename='ELMO_Smile'
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#runs=10
#Classifier=SVC
#print(Classifier)
#ResultMeanStd_SVM_5fold(runs,Classifier,Features,Label,'rbf',16,16,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
#print("Total feature dimension",len(Features[0]),Featurename)
#percent='90'
#print(Classifier,"\n",percent)
#ResultMeanStd_SVM_NR_fold(percent,UNames,runs,Classifier,Features,Label,'rbf',1,1e-05  ,Featurename,Hemo_Dict,Non_hemo_Dict)
percent='70'
print(Classifier,"\n",percent)
ResultMeanStd_SVM_NR_fold(percent,UNames,runs,Classifier,Features,Label,'linear',1,1e-05 ,Featurename,Hemo_Dict,Non_hemo_Dict)
1/0
##Classifier=SVC
##print(Classifier)
##ResultMeanStd_SVM_5fold(runs,Classifier,Features,Label,'rbf',8,256,Featurename,Hemo_Dict,Non_hemo_Dict)#1mer_Smile based XGboost
##print("Total feature dimension",len(Features[0]),Featurename)
##1/0
#Classifier='NN'
#print("5-fold")
#ResultMeanStd_5fold(runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
#
#UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#percent='90'
#print(Classifier,"\n",percent)
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)
#print("Total feature dimension",len(Features[0]),Featurename)
#percent='70'
#ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,0,0,Featurename,Hemo_Dict,Non_hemo_Dict)
#print("Total feature dimension",len(Features[0]),Featurename)
#print(Classifier,"\n",percent)
#1/0
###Classifier=XGBClassifier
###
####ResultMeanStd_5fold(runs,Classifier,Features,Label,85,5,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
####percent='90'
###UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
####print(Classifier,"\n",percent)
####ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,5,Featurename,Hemo_Dict,Non_hemo_Dict)
###percent='70'
###print(Classifier,"\n",percent)
###ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,5,Featurename,Hemo_Dict,Non_hemo_Dict)
###print("Total feature dimension",len(Features[0]),Featurename)
###1/0
###Classifier= RandomForestClassifier
###runs=10
###ResultMeanStd_5fold(runs,Classifier,Features,Label,80,13,Featurename,Hemo_Dict,Non_hemo_Dict)#ELMO_Smile based XGboost
###percent='90'
###UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
###print(Classifier,"\n",percent)
##ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,135,Featurename,Hemo_Dict,Non_hemo_Dict)
##percent='70'
##print(Classifier,"\n",percent)
##ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,170,5,Featurename,Hemo_Dict,Non_hemo_Dict)
##print("Total feature dimension",len(Features[0]),Featurename)
##1/0
Features=torch.FloatTensor(Features)
Features=F.normalize(Features, p=1, dim=1)
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
##
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Without Normalization of features Execution Completed")
C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192]
Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
cv = StratifiedKFold(n_splits=5, shuffle=True)
pre_roc,cc,gg,ee,dd=0,0,0,0,0
#Roc= 0.7142067794380845 c= 1 Average Mean= 0.7135302772565445
"""
For Both Linear and RBF kernel 
"""
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
percent='70'
K=['linear']
for k in K:
    if k=='rbf':
        for c in C:
            for g in Gamma:
                for i in range(5):
                  X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
                  Roc_VA=[]
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
        for i in range(5):
            X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
            Roc_VA=[]
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