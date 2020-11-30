#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:11:33 2019

@author: AdibaYaseen
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
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
from AAC_Features_Extract import *
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
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        
#        self.fc4 = nn.Linear(1024, 2150)
        self.fc4 = nn.Linear(1073, 512)#2048)
#        self.fc5 = nn.Linear(2048, 512)
        self.fc6 = nn.Linear(512, 1)
#        self.fc7 = nn.Linear(100, 1)
        """
        self.fc4 = nn.Linear(1076, 2150)
        self.fc5 = nn.Linear(2150, 512)
        self.fc6 = nn.Linear(512, 1)
        """


    def forward(self, x):
        x = torch.tanh(self.fc4(x))
#        x = torch.tanh(self.fc5(x))
#        x = torch.tanh(self.fc6(x))
        x = self.fc6(x) 
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
def Results(path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Y_test,Y_p,Roc_VA=[],[],[]
    for i in range(5):
        X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
        Hemonet = HemoNet().cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
        X_train=torch.FloatTensor(X_train).cuda()
        y_train=torch.FloatTensor( y_train).cuda()
        X_test=torch.FloatTensor(X_test).cuda()
        y_test=torch.FloatTensor( y_test).cuda()
        for epoch in range(500):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            if loss<0.45:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            LP.append(loss.cpu().data.numpy())
    #Testing
        test_score = Hemonet(X_test)
        test_score=torch.squeeze(test_score, 1)
    #    print("auc_roc",auc_roc)
        Y_p.extend(test_score.cpu().data.numpy())
        Y_test.extend(y_test.cpu().data.numpy())
        test_score.tolist()
        y_test.tolist()
        auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
        Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
#    average_precision_score=average_precision_score(np.array(Y_test), np.array(Y_p))
#    print("auc_roc",auc_roc)
    avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#    print("Average_mean",avgmean)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
    plt.plot(fpr, tpr, color='c',marker=',',label='Our Proposed Model:{: .2f}'.format(auc_roc))
#    plt.grid()
    plt.legend(loc='lower right')
    plt.grid()
    threshold=0.386
    specificity=np.min(1-fpr[np.where(fpr- threshold<0.01)])
    Senstivity_HELMO=np.max(tpr[np.where(fpr- threshold<0.01)])
    MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO, threshold, len(records_hemo),len(records_non_hemo))
#    import pdb;pdb.set_trace()
    
#    print("MCC of OUR model",MCC_HELMO)
#    print("Senstivity of our model",Senstivity_HELMO)
    #Senstivity,specifity, MCC,AUCROC
    return Senstivity_HELMO,specificity,MCC_HELMO ,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p))
def ResultMeanStd(path,UNames,percent,Hemo_Dict,Non_hemo_Dict):
    Senstivity_list,specificity_list,MCC_list ,AUCROC_list,PR_list=[],[],[],[],[]
    for i in range(10):
        Senstivity,specificity,MCC ,AUCROC,PR=Results(path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
        Senstivity_list.append( Senstivity)
        specificity_list.append(specificity)
        MCC_list.append(MCC)
        AUCROC_list.append(AUCROC)
        PR_list.append(PR)
    print(np.mean(Senstivity_list).round(4),'±',np.std(Senstivity_list).round(2),"\n",np.mean( specificity_list).round(4),'±',np.std( specificity_list).round(4),"\n",
          np.mean( MCC_list).round(4),'±',np.std( MCC_list).round(4),"\n",np.mean(   AUCROC_list).round(4),'±',np.std(   AUCROC_list).round(4),"\n",
          np.mean( PR_list).round(4),'±',np.std( PR_list).round(4),"\n")
path="D:/PhD/Hemo_All_SeQ/"
#UNames=RemoveDuplicates(path,'HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#####40% threshhold
#hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo.fasta.clstr.sorted')
#Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo.fasta.clstr.sorted')
#######70% threshold
#hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_70.fasta.clstr.sorted')
#Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_70.fasta.clstr.sorted')
################90%
#hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_90.fasta.clstr.sorted')
#Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_90.fasta.clstr.sorted')
#################
#records_hemo=np.load(path+'ELMO_1024_Hemo_Features.npy')
#hemo_names=np.load(path+'ELMO_1024_Hemo_Names.npy')
#records_non_hemo=np.load(path+'ELMO_1024_NonHemo_Features.npy')
#Non_hemo_names=np.load(path+'ELMO_1024_NonHemo_Names.npy')
#####
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
#####
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
N_hemo=[N_terminous[int(n)] for n in Names_hemo]
C_hemo=[C_terminous[int(n)] for n in Names_hemo]
NC_hemo=np.hstack((N_hemo,C_hemo))
records_hemo=np.hstack((records_hemo,NC_hemo))
###Non hemo
N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
records_non_hemo=np.hstack((records_non_hemo,NC_non_hemo))
##
Hemo_Dict,Non_hemo_Dict={},{}
for n in range(len( Names_hemo)):
    Hemo_Dict[Names_hemo[n]]=records_hemo[n]
for n in range(len( Names_non_hemo)):
    Non_hemo_Dict[Names_non_hemo[n]]=records_non_hemo[n]
Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
#Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
##Hemo_Dict1= All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',1)
##Non_hemo_Dict1= All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',1)
##Hemo_Dict,Non_hemo_Dict={},{}
##for i in range(len(records_hemo)):
##    name=str(hemo_names[i]).split('_')[0]
##    Hemo_Dict[name]=records_hemo[i]
##for i in range(len(records_non_hemo)):
##    name=str(Non_hemo_names[i]).split('_')[0]
##    Non_hemo_Dict[name]=records_non_hemo[i] 
##Hemo=np.hstack((list(Hemo_Dict1.values()),list(Hemo_Dict.values())))
##Non_hemo=np.hstack((list(Non_hemo_Dict1.values()),list(Non_hemo_Dict.values())))
##Features=np.vstack((Hemo,Non_hemo))
#Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
#Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
#Features=torch.FloatTensor(Features)
##Features=F.normalize(Features, p=1, dim=1)
#Label=np.append(np.ones(len(list(Hemo_Dict.values()))),-1*np.ones(len(list(Non_hemo_Dict.values()))))
##1/0
#####
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
percent='70'
#ResultMeanStd(path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#90ELMO
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")

#cv = StratifiedKFold(n_splits=5, shuffle=True)
for i in range(5):
       X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
       print("Total Train:",len(X_train),"Total test",len(X_test))
#       X_train= np.array([], dtype=np.int64).reshape(0,len(Features[0]))
#       X_test, y_train, y_test=[],[],[]
#       train_len=0
#       for idx, pbag in enumerate(hemo_Folds):
#           if i==idx:
#               #pdb.set_trace()
#               nbag=Non_hemo_Folds[idx]
#            #####test data##########
#               nfeatures=name2feature(nbag,Non_hemo_Dict)
#               pfeatures=name2feature(pbag,Hemo_Dict)
#               X_test=np.vstack((pfeatures,nfeatures))
#               y_test=np.append(np.ones(len(pfeatures)),-1*np.ones(len(nfeatures)))
#           else:
#                nbag=Non_hemo_Folds[idx]
#            #####test data##########
#                nfeatures=name2feature(nbag,Non_hemo_Dict)
#                pfeatures=name2feature(pbag,Hemo_Dict)
#                train_features=np.vstack((pfeatures,nfeatures))
#                train_label=np.append(np.ones(len(pfeatures)),-1*np.ones(len(nfeatures)))
#                X_train=np.vstack((X_train,train_features))
#                y_train=np.append(y_train,train_label)
       Hemonet = HemoNet().cuda()
       print(Hemonet)
       Loss=[]
       criterion = nn.MSELoss()
       optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
       scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
       X_train=torch.FloatTensor(X_train).cuda()
       y_train=torch.FloatTensor( y_train).cuda()
       X_test=torch.FloatTensor(X_test).cuda()
       y_test=torch.FloatTensor( y_test).cuda()
       for epoch in range(15000):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            if loss<0.61:#40%
#                break;
#            if loss<0.58:#70%
#                break;
            if loss<0.55:#0.45#new#0.55:#90%#0.74 #0.78 average for external
                break;
#            print("Loss=",loss)
            if ( epoch) % 50== 0:
                params = list(Hemonet.parameters())
                test_score = Hemonet(X_test)
                test_score=torch.squeeze(test_score, 1)
                Y_score.extend(test_score.cpu().data.numpy())
                Y_t.extend(y_test.cpu().data.numpy())
                test_score.tolist()
                y_test.tolist()
                auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
                AUC_list.append(auc_roc)
                print("Epoch",epoch,"auc_roc",auc_roc, "Loss=",np.average(Loss))
                Y_t,Y_score,Loss=[],[],[]
                LP.append(loss.cpu().data.numpy())
                Loss.append(loss.cpu().data.numpy())
       Loss=[]
       test_score = Hemonet(X_test)
       test_score=torch.squeeze(test_score, 1)
       print("auc_roc",auc_roc)
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
plt.figure()                
plt.plot(LP)
plt.plot( AUC_list)
plt.grid() 
plt.figure()
Hemonet_pytorch_total_params = sum(p.numel() for p in Hemonet.parameters() if p.requires_grad)
print ("total trainable parameters",Hemonet_pytorch_total_params)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
fpr_90r, tpr_90r, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#plt.plot(fpr_90r, tpr_90r, color='darkorange',marker='_',label='Non Redundand Cluster 90%: {:.2f}'.format(avgmean))
plt.plot(fpr_90r, tpr_90r, color='darkorange',marker='_',label='Non Redundand Cluster 90%: {:.2f}'.format(auc_roc))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()
####Hemopimod
####

path='D:/PhD/Hemo_All_SeQ/HemoPImodPdbfiles/'
hemopimod_hemo_features=np.load(path+"HemoPImod_Hemo_Features.npy")
hemopimod_Nonhemo_features=np.load(path+"HemoPImod_NonHemo_Features.npy")
hemopimod_Label=np.append(np.ones(len(hemopimod_hemo_features)),-1*np.ones(len(hemopimod_Nonhemo_features)))
#Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
hemopimod_Features=np.vstack((hemopimod_hemo_features,hemopimod_Nonhemo_features))
nc_mod=np.zeros((len(hemopimod_Features),49))
hemopimod_Features=np.hstack((hemopimod_Features,nc_mod))
hemopimod_Features=(hemopimod_Features-np.mean(hemopimod_Features, axis = 0))/(np.std(hemopimod_Features, axis = 0)+0.000001)
hemopimod_Features=torch.FloatTensor(hemopimod_Features).cuda()
hemopimod_score =Hemonet(hemopimod_Features).cpu().data
hemopimod_auc_roc=roc_auc_score(np.array(hemopimod_Label),np.array(hemopimod_score ))
print("hemopimod_auc_roc",hemopimod_auc_roc)
fpr, tpr, thresholds = roc_curve(np.array(hemopimod_Label),np.array(hemopimod_score ))
plt.plot(fpr, tpr, color='c',marker=',',label='External:{: .2f}'.format(hemopimod_auc_roc))
plt.legend(loc='lower right')
###External testing
path='D:/PhD/Hemo_All_SeQ/'
External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
External_features=External_f
External_features=np.vstack((External_f[:2],External_f[4:6]))
External_features=np.vstack((External_features,External_f[8:12]))
External_features=np.vstack((External_features,External_f[15]))
External_features=np.vstack((External_features,External_f[17]))
External_features=np.vstack((External_features,External_f[18:20]))
External_features=np.vstack((External_features,External_f[21:]))
External_features=torch.FloatTensor(External_features).cuda()
External_features=External_features[:,0:1073]
External_score =Hemonet(External_features).cpu().data
print("External_score",External_score)
External_label=np.append(np.zeros(8),np.ones(7))
#External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_90r)
External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation at 90%:{: .2f}'.format(External_auc_roc_90r))
#plt.grid()
plt.legend(loc='lower right')
plt.grid()
###
#HemoPred_external_pred=[1,0,0,0,
#1,1,1,0,#5 randaom 5-12
##13,14
#1,#15
#0,0,1,#16-21 examples
#1,0,0]#22-24 random
#HemoPred_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPred_external_pred))
#print("HemoPred External_auc_roc=",HemoPred_External_auc_roc)
#from sklearn.metrics import accuracy_score
#score = accuracy_score(np.array(External_label),np.array(HemoPred_external_pred))
#print("Accuracy_HemoPred",score)
####HemoPI
#HemoPi_external_pred=[0.51,0.47,0.5,0.45,0.44,0.44,0.48,0.44,0.44,0.46,0.47,0.53,0.44,0.44,0.44]
#HemoPi_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPi_external_pred))
#print("HemoPi External validation ",HemoPi_External_auc_roc)
######
