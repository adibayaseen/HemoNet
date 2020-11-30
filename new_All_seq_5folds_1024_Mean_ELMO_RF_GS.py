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
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
def Results(Features,Label,ee,dd):
    Roc_VA,Y_score,Y_t=[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in cv.split(Features,Label):
        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
        svr_lin = RandomForestClassifier(n_estimators=ee, max_depth=dd,random_state=0)
        svr_lin.fit(X_train, y_train)
        test_score=svr_lin.predict_proba(X_test)[:,1]
        Y_score.extend(test_score)
        Y_t.extend(y_test)
        Roc_VA.append((test_score,list(y_test)))
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
def ResultMeanStd(Features,Label,ee,dd):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(Features,Label,ee,dd)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
    print("Total Runs",i+1,"\n",np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n")
path="D:\PhD\Hemo_All_SeQ/"
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
##Features=F.normalize(Features, p=1, dim=1)
#Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
#print ("Execution Completed")
#####AAC
mer=1
records_hemo=list(All_Features(path,path+'hemo_All_seq.txt',mer).values())
records_non_hemo= list(All_Features(path,path+'Nonhemo_All_seq.txt',mer).values())
Label=np.append(np.ones(len(records_hemo)),np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
#####
#ResultMeanStd(Features,Label,140,145)#ELMO
ResultMeanStd(Features,Label,90,20)#1mer
#ResultMeanStd(Features,Label,130,50)#2mer
1/0
print("Without 1 norm")
cv = StratifiedKFold(n_splits=5, shuffle=True)
C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192,16384,32768,65535]
Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
Depth=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
Estimator=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
pre_roc,cc,gg,ee,dd=0,0,0,0,0
"""
RF
"""
for d in Depth:
   for e in Estimator:
        Y_score,Y_t=[],[]
        for train_index, test_index in cv.split(Features,Label):
            Roc_VA=[]
            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
            svr_lin = RandomForestClassifier(n_estimators=e, max_depth=d,random_state=0)
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
    svr_lin = RandomForestClassifier(n_estimators=ee, max_depth=dd,random_state=0)
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
plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(auc_roc))
plt.plot(avgfpr, avgtpr, color='b',marker='.',label='AUC_avgmean= {:.2f}'.format(avgmean))
plt.legend(loc='lower right')
plt.grid()
plt.show() 
fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
#ELMO_accuracy=Best_accuracy(Y_t, Y_score)
#print("ELMO_Accuracy=",ELMO_accuracy)
#plt.plot(fpr, tpr, color='c',marker=',',label='SVM:{: .2f}'.format(auc_roc))
#plt.scatter(fpr, tpr, color='c',marker=',',label='Hemo_with_ELMO= {:.2f}'.format(auc_roc))
#plt.legend(loc='lower right')
Senstivity_HELMO_RF=np.max(tpr[np.where(fpr-0.386<0.01)])
#plt.grid()
#plt.show()
MCC_HELMO_svm=MCC_fromAUCROC(Senstivity_HELMO_RF,np.max(fpr[np.where(fpr-0.386<0.01)]), len(records_hemo),len(records_non_hemo))
print("MCC of OUR model",MCC_HELMO_svm)
print("Senstivity of our model",Senstivity_HELMO_RF)
1/0
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