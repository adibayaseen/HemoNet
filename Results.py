# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:13:14 2020

@author: 92340
"""

###
from roc import roc_VA
from new_AAC_Features_Extract import *
from  Clusterify import *
###'
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
from sklearn import metrics
from Bio import SeqIO
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from platt import *
import torch.optim as optim
import torch
from sklearn.svm import SVC
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        
#        self.fc4 = nn.Linear(1024, 2150)
        self.fc4 = nn.Linear(3072,2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x=torch.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x) 
        return x
class Results_HemoNet(nn.Module):
    def __init__(self):
        super(Results_HemoNet, self).__init__()
        self.fc4 = nn.Linear(3112,2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x=torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x) 
        return x
class AAC_Smiles_Net(nn.Module):
    def __init__(self):
        super(AAC_Smiles_Net, self).__init__()
        self.fc4 = nn.Linear(2088, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x=torch.relu(self.fc4(x))
#        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x) 
        return x
class ELMO_NC_Net(nn.Module):
    def __init__(self):
        super(ELMO_NC_Net, self).__init__()
        self.fc1 = nn.Linear(1073, 512)
        self.fc2= nn.Linear(512, 100)
        self.fc3= nn.Linear(100, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
class ELMO_AAC2_Smile_Net(nn.Module):
    def __init__(self):
        super(ELMO_AAC2_Smile_Net, self).__init__()
        self.fc4 = nn.Linear(4672,2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x=torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x) 
        return x    
class AAC_NC_Net(nn.Module):
    def __init__(self):
        super(AAC_NC_Net, self).__init__()
        self.fc6 = nn.Linear(89, 200)
        self.fc7 = nn.Linear(200, 1)
    def forward(self, x):
        x = torch.relu(self.fc6(x))
        x = self.fc7(x) 
        return x
class AAC2_NC_Net(nn.Module):
    def __init__(self):
        super(AAC2_NC_Net, self).__init__()
        self.fc6 = nn.Linear(1649, 100)
        self.fc7 = nn.Linear(100, 1)
    def forward(self, x):
        x = torch.relu(self.fc6(x))
        x = self.fc7(x) 
        return x    
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
def RedendencyRemoval(fold,path,UNames,hemo_file,nonhemo_file,Hemo_Dict,Non_hemo_Dict):
    #################90 new %
    hemo_CL=Make_Cluster(UNames,path,hemo_file)
    Non_hemo_CL=Make_Cluster(UNames,path,nonhemo_file)
    hemo_Folds=chunkify(hemo_CL)
    Non_hemo_Folds=chunkify(Non_hemo_CL)
    ####
    for i in range(5):
           X_train= np.array([], dtype=np.int64).reshape(0, len(list(Hemo_Dict.values())[0]))
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
#                   y_test=np.append(np.ones(len(pfeatures)),np.zeros(len(nfeatures)))
               else:
                    nbag=Non_hemo_Folds[idx]
                #####test data##########
                    nfeatures=name2feature(nbag,Non_hemo_Dict)
                    pfeatures=name2feature(pbag,Hemo_Dict)
                    train_features=np.vstack((pfeatures,nfeatures))
                    train_label=np.append(np.ones(len(pfeatures)),-1*np.ones(len(nfeatures)))
#                    train_label=np.append(np.ones(len(pfeatures)),np.zeros(len(nfeatures)))
                    X_train=np.vstack((X_train,train_features))
                    y_train=np.append(y_train,train_label)
    return X_train,X_test, y_train, y_test
def Results_5fold(Classifier,Features,Label,e,d,Featurename,Hemodict,Nonhemodict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in cv.split(Features,Label):
        X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
        svr_lin = Classifier(max_depth=d, n_estimators=e)
        svr_lin.fit(X_train, y_train)
        test_score=svr_lin.predict_proba(X_test)[:,1]
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
    plt.legend(loc='lower right')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemodict),len(Nonhemodict))
    V=svr_lin.predict_proba(X_train)[:,1]
    L=y_train
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External

def Results_5fold_SVM(Classifier,Features,Label,kk,cc,gg,Featurename,Hemodict,Nonhemodict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    Features=torch.FloatTensor(Features)
    Features=F.normalize(Features, p=1, dim=1)
    if kk=='linear':
        for train_index, test_index in cv.split(Features,Label):
            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
            svr_lin=SVC(kernel='linear', C=cc)
            svr_lin.fit(X_train, y_train)
            test_score=svr_lin.decision_function(X_test)
            Y_p.extend(test_score)
            Y_test.extend(y_test)
            Roc_VA.append((test_score,list(y_test)))
    elif kk=='rbf':
        for train_index, test_index in cv.split(Features,Label):
            X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
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
    plt.plot(fpr, tpr, color='c',marker=',',label='AUC_SVVM= {:.2f}'.format(auc_roc))
    plt.legend(loc='lower right')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemodict),len(Nonhemodict))
    V=svr_lin.decision_function(X_train)
    L=y_train
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External
def Results_NR_fold_SVM(percent,UNames,Classifier,Features,Label,kk,cc,gg,Featurename,Hemo_Dict,Non_hemo_Dict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    Features=torch.FloatTensor(Features)
    Features=F.normalize(Features, p=1, dim=1)
    path="D:\PhD\Hemo_All_SeQ/"
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
    fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemo_Dict),len(Non_hemo_Dict))
    V=svr_lin.decision_function(X_train)
    L=y_train
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External
def ResultMeanStd_SVM_NR_fold(percent,UNames,runs,Classifier,Features,Label,kk,cc,gg,Featurename,Hemodict,Nonhemodict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list,External_list,Clinical_list=[],[],[],[],[],[],[]
    for i in range(runs):
        Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_NR_fold_SVM(percent,UNames,Classifier,Features,Label,kk,cc,gg,Featurename,Hemodict,Nonhemodict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
        Clinical_list.append( Clinical)
        External_list.append( External)
#        pdb.set_trace()
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n",
          np.mean( Clinical_list).round(4),'±',np.std( Clinical_list).round(4),"\n",
          np.mean( External_list).round(4),'±',np.std( External_list).round(4),"\n")
def ResultMeanStd_SVM_5fold(runs,Classifier,Features,Label,kk,cc,gg,Featurename,Hemodict,Nonhemodict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list,External_list,Clinical_list=[],[],[],[],[],[],[]
    for i in range(runs):
        Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_5fold_SVM(Classifier,Features,Label,kk,cc,gg,Featurename,Hemodict,Nonhemodict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
        Clinical_list.append( Clinical)
        External_list.append( External)
#        pdb.set_trace()
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n",
          np.mean( Clinical_list).round(4),'±',np.std( Clinical_list).round(4),"\n",
          np.mean( External_list).round(4),'±',np.std( External_list).round(4),"\n")
def Results_5fold_NN(Classifier,Features,Label,Featurename,Hemodict,Nonhemodict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in cv.split(Features,Label):
       X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
       if Featurename=='ELMO_Smile':
            Hemonet = HemoNet().cuda()
       elif Featurename=='AAC_Smile':
           Hemonet = AAC_Smiles_Net().cuda()
       elif Featurename=='AAC2_Smile':
           Hemonet = HemoNet().cuda()
       elif Featurename=='ELMO_NC':
           Hemonet =ELMO_NC_Net().cuda()
       elif Featurename=='ELMO_AAC2_Smile':
           Hemonet = ELMO_AAC2_Smile_Net().cuda()
       elif Featurename=='AAC_NC':
           Hemonet = AAC_NC_Net().cuda()
       elif Featurename=='AAC2_NC':
           Hemonet = AAC2_NC_Net().cuda()
       else:
           Hemonet = Results_HemoNet().cuda()
       Loss=[]
       criterion = nn.MSELoss()
       optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.00001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
       X_train=torch.FloatTensor(X_train).cuda()
       y_train=torch.FloatTensor( y_train).cuda()
       X_test=torch.FloatTensor(X_test).cuda()
       y_test=torch.FloatTensor( y_test).cuda()
       for epoch in range(1500):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss<0.2:#0.45#new#0.55:#90%#0.74 #0.78 average for external
                break;
       Loss=[]
       test_score = Hemonet(X_test)
       test_score=torch.squeeze(test_score, 1)
       Y_p.extend(test_score.cpu().data.numpy())
       Y_test.extend(y_test.cpu().data.numpy())
       test_score.tolist()
       y_test.tolist()
       auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
       Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
    print("auc_roc",auc_roc)
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
    print("Average_mean",avgmean)
    fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemodict),len(Nonhemodict))
    svr_lin=Hemonet
    V=Hemonet(X_train).cpu().data.numpy()
    L=y_train.cpu().data.numpy()
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External
def Results_NR_fold_NN(percent,UNames,Classifier,Features,Label,Featurename,Hemodict,Nonhemodict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    path="D:\PhD\Hemo_All_SeQ/"
    for i in range(5):
       X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemodict,Nonhemodict)
#       print("Total Train:",len(X_train),"Total test",len(X_test))
       if Featurename=='ELMO_Smile':
            Hemonet = HemoNet().cuda()
       elif Featurename=='AAC_Smile':
           Hemonet = AAC_Smiles_Net().cuda()
       elif Featurename=='AAC2_Smile':
           Hemonet = HemoNet().cuda()
       elif Featurename=='ELMO_NC':
           Hemonet =ELMO_NC_Net().cuda()
       elif Featurename=='ELMO_AAC2_Smile':
           Hemonet = ELMO_AAC2_Smile_Net().cuda()
       elif Featurename=='AAC_NC':
           Hemonet = AAC_NC_Net().cuda()
       elif Featurename=='AAC2_NC':
           Hemonet = AAC2_NC_Net().cuda()
       else:
           Hemonet = Results_HemoNet().cuda()
       Loss=[]
       criterion = nn.MSELoss()
       optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.00001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
       X_train=torch.FloatTensor(X_train).cuda()
       y_train=torch.FloatTensor( y_train).cuda()
       X_test=torch.FloatTensor(X_test).cuda()
       y_test=torch.FloatTensor( y_test).cuda()
       Y_t,Y_score=[],[]
       for epoch in range(1500):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if percent=='70' and loss<0.35:#0.45#new#0.55:#90%#0.74 #0.78 average for external
                break;
            elif percent=='90' and loss<0.15:#0.45#new#0.55:#90%#0.74 #0.78 average for external
                break;
#            if ( epoch) % 10== 0:
#                params = list(Hemonet.parameters())
#                test_score = Hemonet(X_test)
#                test_score=torch.squeeze(test_score, 1)
#                Y_score.extend(test_score.cpu().data.numpy())
#                Y_t.extend(y_test.cpu().data.numpy())
#                test_score.tolist()
#                y_test.tolist()
#                auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#                AUC_list.append(auc_roc)
#                print("Epoch",epoch,"auc_roc",auc_roc, "Loss=",np.average(Loss))
#                Y_t,Y_score,Loss=[],[],[]
#                LP.append(loss.cpu().data.numpy())
#                Loss.append(loss.cpu().data.numpy())
       Loss=[]
       test_score = Hemonet(X_test)
       test_score=torch.squeeze(test_score, 1)
       Y_p.extend(test_score.cpu().data.numpy())
       Y_test.extend(y_test.cpu().data.numpy())
       test_score.tolist()
       y_test.tolist()
       auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
       Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
    print("auc_roc",auc_roc)
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
    print("Average_mean",avgmean)
    fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemodict),len(Nonhemodict))
    svr_lin=Hemonet
    V=Hemonet(X_train).cpu().data.numpy()
    L=y_train.cpu().data.numpy()
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External
def Results_NR_fold(percent,UNames,Classifier,Features,Label,e,d,Featurename,Hemo_Dict,Non_hemo_Dict):
    Y_test,Y_score, Y_p,names,avg_roc,Roc_VA,LP,all_losses,AUC_list,Loss=[],[],[],[],[],[],[],[],[],[]
    path="D:\PhD\Hemo_All_SeQ/"
    for i in range(5):
        X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
        svr_lin = Classifier(max_depth=d, n_estimators=e)
        svr_lin.fit(X_train, y_train)
        test_score=svr_lin.predict_proba(X_test)[:,1]
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
    plt.legend(loc='lower right')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC=MCC_fromAUCROC(Senstivity, threshold, len(Hemo_Dict),len(Non_hemo_Dict))
    V=svr_lin.predict_proba(X_train)[:,1]
    L=y_train
    A,B = plattFit(V,L)
    Clinical=Clinical_data_result(svr_lin,A,B,Featurename)
    External=External_data_result(svr_lin,Featurename)
    return Senstivity,specificity,MCC ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)),Clinical,External
def Clinical_data_result(Classifier,A,B,FeatureName):
    path="D:\PhD\Hemo_All_SeQ/"
    ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
    mer=1
    AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
    AAC2=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',2).values())
    DRAMP_Clinical_NC=np.zeros((28,49))
    Smiles_NC=np.zeros((28,2048))
    ALL_DRAMP_Clinical_data_Features=np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,AAC)),Smiles_NC))
    FeatureDict={'ELMO':ELMO_DRAMP_Clinical_data_Features,
                 'NC':DRAMP_Clinical_NC,
                 'AAC':AAC,
                 'AAC2':AAC2,
                 'ELMO_NC':np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),
                 'AAC_NC':np.hstack((AAC,DRAMP_Clinical_NC)),
                 'AAC2_NC':np.hstack((AAC2,DRAMP_Clinical_NC)),
                 'ELMO_Smile':np.hstack((ELMO_DRAMP_Clinical_data_Features,Smiles_NC)),
                 'AAC_Smile':np.hstack((AAC,Smiles_NC)),
                 'AAC2_Smile':np.hstack(( AAC2,Smiles_NC)),
                 'ELMO_AAC_Smile': ALL_DRAMP_Clinical_data_Features,
                 'ELMO_AAC2_Smile':np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,AAC2)),Smiles_NC))
                 }
    Features= FeatureDict[FeatureName]
    import pdb; pdb.set_trace()
    Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
    if len(str(Classifier).split('SVC'))>1:
        Score=Classifier.decision_function(Features)
    elif len(str(Classifier).split('XGBClassifier'))>1 or len(str(Classifier).split('RandomForestClassifier'))>1 :
        Score=Classifier.predict_proba(Features)[:,1]
    else:
        Features=torch.FloatTensor(Features).cuda()
        Score =Classifier(Features).cpu().data.numpy()
    meanScore=np.mean(Score)
    print("Average score ALL features",meanScore)
    V=Score
    rasacaled_clinical = sigmoid(V,A,B)
    rasacaled_clinical = np.mean(rasacaled_clinical )
    print("rasacaled_clinical ",rasacaled_clinical )
    return rasacaled_clinical 
def External_data_result(Classifier,FeatureName):
    path='D:\PhD\Hemo_All_SeQ/'
    ELMO_F=np.load(path+'Hemolytic_External_validation_Features.npy')
    ELMO_Features=ELMO_F
    mer=1
    AAC=list( All_FeaturesWithoutNC(path,path+'External.txt',mer).values())
    AAC2=list( All_FeaturesWithoutNC(path,path+'External.txt',2).values())
    NC=np.zeros((len(ELMO_Features),49))
    Smiles_NC=np.zeros((len(ELMO_Features),2048))
    FeatureDict={
            'ELMO':ELMO_Features,
                 'NC':NC,
                 'AAC':AAC,
                 'AAC2':AAC2,
                 'ELMO_NC':np.hstack((ELMO_Features,NC)),
                 'AAC_NC':np.hstack(( AAC,NC)),
                 'AAC2_NC':np.hstack(( AAC2,NC)),
                 'ELMO_Smile':np.hstack((ELMO_Features,Smiles_NC)),
                 'AAC_Smile':np.hstack(( AAC,Smiles_NC)),
                 'AAC2_Smile':np.hstack(( AAC2,Smiles_NC)),
                 'ELMO_AAC_Smile': np.hstack((np.hstack(( ELMO_Features,AAC)),Smiles_NC)),
                 'ELMO_AAC2_Smile':np.hstack((np.hstack((ELMO_Features,AAC2)),Smiles_NC))
                 }
    Features= FeatureDict[FeatureName]
    Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
    if len(str(Classifier).split('SVC'))>1:
        Score=Classifier.decision_function(Features)
    elif len(str(Classifier).split('XGBClassifier'))>1 or len(str(Classifier).split('RandomForestClassifier'))>1 :
        Score=Classifier.predict_proba(Features)[:,1]
    else:
        Features=torch.FloatTensor(Features).cuda()
        Score =Classifier(Features).cpu().data.numpy()
    External_label=np.append(np.zeros(8),np.ones(5))
#    External_label=np.append(np.zeros(7),np.ones(7))
    AUCROC=roc_auc_score(np.array(External_label),np.array(Score))
    print("External_auc_roc",AUCROC)#,All_FeaturesWithoutNC(path,path+'External.txt',mer).keys(),Score)
#    External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(Score))
#    plt.figure()
#    plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation 5-fold:{: .2f}'.format(AUCROC))
#    plt.legend(loc='lower right')
#    plt.grid()
#    import pdb;pdb.set_trace()
    return AUCROC
def ResultMeanStd_NR_fold(percent,UNames,runs,Classifier,Features,Label,e,d,Featurename,Hemodict,Nonhemodict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list,External_list,Clinical_list=[],[],[],[],[],[],[]
    for i in range(runs):
        if Classifier!='NN':
            Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_NR_fold(percent,UNames,Classifier,Features,Label,e,d,Featurename,Hemodict,Nonhemodict)
        else:
            Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_NR_fold_NN(percent,UNames,Classifier,Features,Label,Featurename,Hemodict,Nonhemodict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
        Clinical_list.append( Clinical)
        External_list.append( External)
#        pdb.set_trace()
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n",
          np.mean( Clinical_list).round(4),'±',np.std( Clinical_list).round(4),"\n",
          np.mean( External_list).round(4),'±',np.std( External_list).round(4),"\n")
def ResultMeanStd_5fold(runs,Classifier,Features,Label,e,d,Featurename,Hemodict,Nonhemodict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list,External_list,Clinical_list=[],[],[],[],[],[],[]
    for i in range(runs):
        if Classifier!='NN':
            Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_5fold(Classifier,Features,Label,e,d,Featurename,Hemodict,Nonhemodict)
        elif  Classifier=='NN':
             Senstivity,specificity,MCC ,AUCROC,PR,Clinical,External=Results_5fold_NN(Classifier,Features,Label,Featurename,Hemodict,Nonhemodict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
        Clinical_list.append( Clinical)
        External_list.append( External)
#        pdb.set_trace()
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n",
          np.mean( Clinical_list).round(4),'±',np.std( Clinical_list).round(4),"\n",
          np.mean( External_list).round(4),'±',np.std( External_list).round(4),"\n")