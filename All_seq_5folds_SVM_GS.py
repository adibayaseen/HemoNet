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
###
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score as auc_roc
from sklearn import metrics
from Bio import SeqIO
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from sklearn.model_selection import KFold,StratifiedKFold

Names,cluster_ids,Seqs, Labels=[],[],[],[]
names,hemo_CL,Non_hemo_CL=[],[],[]
path="D:\PhD\Hemo_All_SeQ/"
records_hemo=list(All_Features(path,path+'hemo_All_seq.txt',1).values())
records_non_hemo= list(All_Features(path,path+'Nonhemo_All_seq.txt',1).values())
Label=np.append(np.ones(len(records_hemo)),np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
#1/0
Features=F.normalize(Features, p=1, dim=1)
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
cv = StratifiedKFold(n_splits=5, shuffle=True)
#1/0
C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192]
Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
#Gamma=[1,2,4,8,16,32,64,128,256,512,1024,2056]
pre_roc,cc,gg=0,0,0
"""
For Linear SVC
"""
for c in C:
    Y_score,Y_t,Roc_VA=[],[],[]
    print("c",c)
    for train_index, test_index in cv.split(Features,Label):
        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
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
"""
For RBF kernel SVC
"""
for i in C:
    for j in Gamma:
        Y_score,Y_t=[],[]
        for train_index, test_index in cv.split(Features,Label):
            Roc_VA=[]
            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
    #        svr_lin=XGBClassifier(max_depth=3, learning_rate=0.1)
            svr_lin=SVC(kernel='rbf', C=i,gamma=j)
#                svr_lin=SVC(kernel='linear', C=i)
            svr_lin.fit(X_train, y_train)
            test_score=svr_lin.decision_function(X_test)
#            Y_p=svr_lin.predict(X_test)
            Y_score.extend(test_score)
            Y_t.extend(y_test)
#            Y_pred.extend(Y_p) 
            Roc_VA.append((test_score,list(y_test)))
        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
        print("auc_roc",auc_roc,"c=",i,"g=",j)
        if pre_roc<auc_roc:
           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"c=",i,"g=",j,"Average Mean=",avgmean)
           pre_roc=auc_roc
           cc=i
           gg=j
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
    elif k=='rbf':
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