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
def Results(e,d,path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Y_test,Y_p,Roc_VA=[],[],[]
    for i in range(5):
        X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
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
def ResultMeanStd(e,d,path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(e,d,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
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
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]

####
###AAC Features
mer=1
print("AAC features",mer,"mer")
AAC_hemo=list( All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',mer).values())
AAC_non_hemo= list( All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',mer).values())
AAC_Features=np.vstack((AAC_hemo,AAC_non_hemo))
AAC_Features=(AAC_Features-np.mean(AAC_Features, axis = 0))/(np.std(AAC_Features, axis = 0)+0.000001)
#only 1mer
Features=AAC_Features
Hemo_Dict=dict(zip(Names_hemo,AAC_hemo))
Non_hemo_Dict=dict(zip(Names_non_hemo,AAC_non_hemo))
###############
Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
Features=torch.FloatTensor(Features)
######AAC Features
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")
###Total 10 times mean std  
##ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
##ResultMeanStd(Features,Label,135,5)#ELMO
#1/0
###
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
percent='90';#'90'
#Roc= 0.680922729762652 e= 170 d= 5 Average Mean= 0.6848083327955702
#ResultMeanStd(170,5,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#ELMO_90
#ResultMeanStd(170,1,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#ELMO_70
#Roc= 0.6711839831124 e= 135 d= 30 Average Mean= 0.6728603346565925
#ResultMeanStd(135,30,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#1mer_90
#ResultMeanStd(170,1,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#1mer_70
#ResultMeanStd(170,15,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#2mer_90
#ResultMeanStd(170,1,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#2mer_70
#1/0
#################90 new %
##path='D:/Downloads/'
#Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[200]#[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
Depth=[20]#[135,140,145,150,155,160,165,170]#1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,
#ELMO best results Roc= 0.685580142973737 e= 135 d= 6 Average Mean= 0.6893453668904338
#1mer Roc= 0.6722432260758173 e= 135 d= 135 Average Mean= 0.6764716431715094#1mer Roc= 0.6814864124507055 e= 170 d= 155 Average Mean= 0.6822134531724315

####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
#cv = StratifiedKFold(n_splits=5, shuffle=True)
pre_roc,cc,gg,ee,dd=0,0,0,0,0

"""
XGboost
"""
for d in Depth:
    for e in Estimator:
        for i in range(5):
               X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
#               print("Total Train:",len(X_train),"Total test",len(X_test))
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
for i in range(5):
   X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
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

mer=1
print("External AAC features",mer,"mer")
DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
DRAMP_Clinical_data_AAC=(DRAMP_Clinical_data_AAC-np.mean(DRAMP_Clinical_data_AAC, axis = 0))/(np.std(DRAMP_Clinical_data_AAC, axis = 0)+0.000001)
DRAMP_Clinical_data_AAC_score=svr_lin.predict_proba(DRAMP_Clinical_data_AAC)[:,1]
plt.figure()
plt.hist(np.sort(DRAMP_Clinical_data_AAC_score),bins=len(DRAMP_Clinical_data_AAC_score))
plt.grid()
#####DRAMP Predicted from Hemopred
HemoPred_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPred_predicted_DRAMP_Clinical_data.csv')
Score_HemoPred_predicted_DRAMP_Clinical_data=HemoPred_predicted_DRAMP_Clinical_data[['Prediction']].values
plt.figure()
plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data)
plt.grid()
####DRAMP HEmoPI predicted results
#####DRAMP Predicted from Hemopred
HemoPI_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPI_predicted_DRAMP_Clinical_data.csv')
Score_HemoPI_predicted_DRAMP_Clinical_data=HemoPI_predicted_DRAMP_Clinical_data[['PROB Score']].values
plt.figure()
plt.hist(Score_HemoPI_predicted_DRAMP_Clinical_data)
plt.grid()