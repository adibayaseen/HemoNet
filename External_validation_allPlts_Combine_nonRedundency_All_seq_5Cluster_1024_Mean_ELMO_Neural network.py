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
from sklearn.metrics import roc_auc_score as auc_roc
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
        self.fc4 = nn.Linear(1076, 512)
        self.fc6 = nn.Linear(512, 1)
    def forward(self, x):
        x = torch.tanh(self.fc4(x))
        x = self.fc6(x) 
        return x
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
def hing(prediction,target):
    return torch.max(torch.from_numpy(np.zeros(1)),torch.from_numpy(np.ones(1))-target*prediction)
"""
HELMO without non reduendency removal
"""
path="D:\PhD\Hemo_All_SeQ/"
records_hemo=np.load(path+'ELMO_1024_Hemo_Features.npy')
records_non_hemo=np.load(path+'ELMO_1024_NonHemo_Features.npy')
Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
#Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
LP=[]
print ("Execution Completed")
all_losses=[]
Roc_VA, Y_pred,Y_t,Y_score=[],[],[],[]
avg_roc=[]
AUC_list=[]
Loss=[]
Y_test,Y_p=[],[]
cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(Features,Label):
    Roc_V=[]
    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
    Hemonet = HemoNet().cuda()
    print(Hemonet)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    X_train=torch.FloatTensor(X_train).cuda()
    y_train=torch.FloatTensor( y_train).cuda()
    X_test=torch.FloatTensor(X_test).cuda()
    y_test=torch.FloatTensor( y_test).cuda()
    for epoch in range(400):
        output = Hemonet(X_train)
        output=torch.squeeze(output, 1)
        target =   y_train
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        params = list(Hemonet.parameters())
        if ( epoch) % 50== 0:
            scheduler.step()
            test_score = Hemonet(X_test)
            test_score=torch.squeeze(test_score, 1)
            Y_score.extend(test_score.cpu().data.numpy())
            Y_t.extend(y_test.cpu().data.numpy())
            test_score.tolist()
            y_test.tolist()
            auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
            print("Epoch",epoch,"auc_roc",auc_roc, "Loss=",np.average(LP))
            Y_t,Y_score=[],[]
            Loss.append(np.average(LP))
        test_score = Hemonet(X_test)
        test_score=torch.squeeze(test_score, 1)
        Y_score.extend(test_score.cpu().data.numpy())
        Y_t.extend(y_test.cpu().data.numpy())
        test_score.tolist()
        y_test.tolist()
        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
        AUC_list.append(auc_roc)
        Y_t,Y_score=[],[]
        LP.append(loss.cpu().data.numpy())
#Testing
    test_score = Hemonet(X_test)
    test_score=torch.squeeze(test_score, 1)
    print("auc_roc",auc_roc)
    Y_p.extend(test_score.cpu().data.numpy())
    Y_test.extend(y_test.cpu().data.numpy())
    test_score.tolist()
    y_test.tolist()
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
    Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
    ###New Testing 
    """
    from allennlp.commands.elmo import ElmoEmbedder
    from pathlib import Path
    path='/content/drive/My Drive/ELMO_Embedding/'#Colab
    model_dir =path# Path('/home/fayyaz/Desktop/Adiba/ELMO_Embedding/seqvec/uniref50_v2')#turing
    #model_dir = Path('/home/adiba/Desktop/PhD/4rth/ELMO/uniref50_v2')#Nangaparbet
    weights = model_dir +'weights.hdf5'
    options = model_dir + 'options.json'
    seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU
    #seqvec=torch.load(path+'weights.hdf5_Epoch_1auc_roc0.6781307888574324')
    #seqvec.load_state_dict(torch.load(path+'weights.hdf5_Epoch_1auc_roc0.6781307888574324'))
    seq1F,name1F=All_Features(seq)
    newtest_score = Hemonet(seq1F)
    """
#plt.figure()                
#plt.plot(LP)
#plt.plot( AUC_list)
#plt.grid() 

#plt.figure()
#plt.plot(Loss)
auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.legend()
#fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#plt.plot(fpr, tpr, color='c',marker=',',label='Without Redendency removal :{: .2f}'.format(auc_roc))
#plt.legend(loc='lower right')
#Senstivity_HELMO=np.max(tpr[np.where(fpr-0.3<0.01)])
##plt.grid()
##plt.show()
#MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO,0.3, len(records_hemo),len(records_non_hemo))
#print("MCC of OUR model",MCC_HELMO)
#print("Senstivity of our model",Senstivity_HELMO)
####
path='D:/PhD/Hemo_All_SeQ/'
External_features=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
External_features=torch.FloatTensor(External_features).cuda()
External_score =Hemonet(External_features).cpu().data
print("External_score",External_score)
External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc)
fpr, tpr, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(fpr, tpr, color='c',marker=',',label='External:{: .2f}'.format(External_auc_roc))
"""
HemoPImod testing features extracted using ELMO With 5 fold model 
"""
path='D:/PhD/Hemo_All_SeQ/'
Hemopimod_hemo=np.load(path+"HemoPImod_Hemo_Features.npy")
Hemopimod_Nonhemo=np.load(path+"HemoPImod_NonHemo_Features.npy")
Hemopimod_features=np.vstack((Hemopimod_hemo,Hemopimod_Nonhemo))
Hemopimod_label=np.append(np.ones(len(Hemopimod_hemo)),np.zeros(len(Hemopimod_Nonhemo)))
Hemopimod_features=torch.FloatTensor(Hemopimod_features).cuda()
Hemopimod_score =Hemonet(Hemopimod_features).cpu().data
Hemopimod_auc_roc=roc_auc_score(np.array(Hemopimod_label),np.array(Hemopimod_score))
print("Hemopimod_auc_roc",Hemopimod_auc_roc)
Hemopimod_fpr, Hemopimod_tpr, thresholds = roc_curve(np.array(Hemopimod_label),np.array(Hemopimod_score))
plt.plot(Hemopimod_fpr, Hemopimod_tpr, color='darkorange',marker='v',label='External at 40%:{: .2f}'.format(Hemopimod_auc_roc))
1/0
#plt.legend(loc='lower right')
####################
path="D:\PhD\Hemo_All_SeQ/"
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
records_hemo=np.load(path+'ELMO_1024_Hemo_Features.npy')
hemo_names=np.load(path+'ELMO_1024_Hemo_Names.npy')
records_non_hemo=np.load(path+'ELMO_1024_NonHemo_Features.npy')
Non_hemo_names=np.load(path+'ELMO_1024_NonHemo_Names.npy')

#Hemo_Dict1= All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',1)
#Non_hemo_Dict1= All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',1)
Hemo_Dict,Non_hemo_Dict={},{}
for i in range(len(records_hemo)):
    Hemo_Dict[hemo_names[i]]=records_hemo[i]
for i in range(len(records_non_hemo)):
    Non_hemo_Dict[Non_hemo_names[i]]=records_non_hemo[i] 
#Hemo=np.hstack((list(Hemo_Dict1.values()),list(Hemo_Dict.values())))
#Non_hemo=np.hstack((list(Non_hemo_Dict1.values()),list(Non_hemo_Dict.values())))
#Features=np.vstack((Hemo,Non_hemo))
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
#Features=F.normalize(Features, p=1, dim=1)
Label=np.append(np.ones(len(list(Hemo_Dict.values()))),-1*np.ones(len(list(Non_hemo_Dict.values()))))
####
UNames=RemoveDuplicates(path,'HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
#####40% threshhold
#hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo.fasta.clstr.sorted')
#Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo.fasta.clstr.sorted')
################90%
hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_90.fasta.clstr.sorted')
Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_90.fasta.clstr.sorted')
######
hemo_Folds=chunkify(hemo_CL)
Non_hemo_Folds=chunkify(Non_hemo_CL)
####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
#cv = StratifiedKFold(n_splits=5, shuffle=True)
for i in range(5):
       X_train= np.array([], dtype=np.int64).reshape(0,len(Features[0]))
       X_test, y_train, y_test=[],[],[]
       train_len=0
       for idx, pbag in enumerate(hemo_Folds):
           if i==idx:
               #pdb.set_trace()
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
       for epoch in range(1500):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            if loss<0.60:#40%
#                break;
#            if loss<0.58:#70%
#                break;
            if loss<0.5:#90%
                break;
#            print("Loss=",loss)
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
#auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
#print("auc_roc",auc_roc)
#avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#print("Average_mean",avgmean)
#plt.figure()                
#plt.plot(LP)
#plt.plot( AUC_list)
#plt.grid() 
#plt.figure()
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
#fpr_40r, tpr_40r, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#plt.plot(fpr_40r, tpr_40r, color='darkorange',marker='_',label='Non Redundand Cluster 40%: {:.2f}'.format(avgmean))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()
"""
HemoPImod testing features extracted using ELMO With redendency removal model
"""
path='D:/PhD/Hemo_All_SeQ/'
Hemopimod_hemo=np.load(path+"HemoPImod_Hemo_Features.npy")
Hemopimod_Nonhemo=np.load(path+"HemoPImod_NonHemo_Features.npy")
Hemopimod_features=np.vstack((Hemopimod_hemo,Hemopimod_Nonhemo))
Hemopimod_label=np.append(np.ones(Hemopimod_features),np.zeros(len(Hemopimod_Nonhemo)))
Hemopimod_features=torch.FloatTensor(Hemopimod_features).cuda()
Hemopimod_score =Hemonet(Hemopimod_features).cpu().data
Hemopimod_auc_roc_40r=roc_auc_score(np.array(Hemopimod_label),np.array(Hemopimod_score))
print("External_auc_roc",Hemopimod_auc_roc_40r)
Hemopimod_fpr_40r, Hemopimod_tpr_40r, thresholds = roc_curve(np.array(Hemopimod_label),np.array(Hemopimod_score))
plt.plot(Hemopimod_fpr_40r, Hemopimod_tpr_40r, color='darkorange',marker='v',label='External at 40%:{: .2f}'.format(Hemopimod_auc_roc_40r))
1/0
       
####External testing
path='D:/PhD/Hemo_All_SeQ/'
External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
External_features=np.vstack((External_f[:2],External_f[4:6]))
External_features=np.vstack((External_features,External_f[8:12]))
External_features=np.vstack((External_features,External_f[15]))
External_features=np.vstack((External_features,External_f[17]))
External_features=np.vstack((External_features,External_f[18:20]))
External_features=np.vstack((External_features,External_f[21:]))
External_features=torch.FloatTensor(External_features).cuda()
External_score =Hemonet(External_features).cpu().data
print("External_score",External_score)
External_label=np.append(np.zeros(8),np.ones(7))
#External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc_40r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_40r)
External_fpr_40r, External_tpr_40r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_40r, External_tpr_40r, color='darkorange',marker='v',label='External at 40%:{: .2f}'.format(External_auc_roc_40r))
#plt.legend(loc='lower right')
#########
#HemoPred_external_pred=[1,0,1,0,1,1,0,1,1,1,0,1,0,0,1]
##
#HemoPi_external_pred=[0.51,0.47,0.44,0.33,0.5,0.45,0.43,0.44,0.44,0.44,0.48,0.44,0.76,0.87,0.44,0.83,0.46,0.8,0.47,0.53,0.82,0.44,0.44,0.44]

#####
HemoPred_external_pred=[1,0,0,0,
1,1,1,0,#5 randaom 5-12
#13,14
1,#15
0,0,1,#16-21 examples
1,0,0]#22-24 random
HemoPred_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPred_external_pred))
print("HemoPred_External_auc_roc=",HemoPred_External_auc_roc)
HemoPred_External_fpr, HemoPred_External_tpr, thresholds = roc_curve(np.array(External_label),np.array(HemoPred_external_pred))
plt.plot(HemoPred_External_fpr, HemoPred_External_tpr, color='r',marker='+',label='HemoPred External Validation:{: .2f}'.format(HemoPred_External_auc_roc))
plt.legend(loc='lower right')
from sklearn.metrics import accuracy_score
score = accuracy_score(np.array(External_label),np.array(HemoPred_external_pred))
print("Accuracy_HemoPred",score)
###HemoPI
HemoPi_external_pred=[0.51,0.47,0.5,0.45,0.44,0.44,0.48,0.44,0.44,0.46,0.47,0.53,0.44,0.44,0.44]
HemoPi_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPi_external_pred))
print("HemoPi_external_pred",HemoPi_External_auc_roc)
HemoPi_External_fpr, HemoPi_External_tpr, thresholds = roc_curve(np.array(External_label),np.array(HemoPi_external_pred))
plt.plot(HemoPi_External_fpr, HemoPi_External_tpr, color='b',marker='x',label='HemoPi External:{: .2f}'.format(HemoPred_External_auc_roc))
######
#######70% threshold
hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_70.fasta.clstr.sorted')
Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_70.fasta.clstr.sorted')
######
hemo_Folds=chunkify(hemo_CL)
Non_hemo_Folds=chunkify(Non_hemo_CL)
####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
#cv = StratifiedKFold(n_splits=5, shuffle=True)
for i in range(5):
       X_train= np.array([], dtype=np.int64).reshape(0,len(Features[0]))
       X_test, y_train, y_test=[],[],[]
       train_len=0
       for idx, pbag in enumerate(hemo_Folds):
           if i==idx:
               #pdb.set_trace()
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
       for epoch in range(1500):
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            if loss<0.61:#40%
#                break;
            if loss<0.58:#70%
                break;
#            if loss<0.5:#90%
#                break;
#            print("Loss=",loss)
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
#auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
#print("auc_roc",auc_roc)
#avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#print("Average_mean",avgmean)
#plt.figure()                
#plt.plot(LP)
#plt.plot( AUC_list)
#plt.grid() 
#plt.figure()
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
#fpr_40r, tpr_40r, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#plt.plot(fpr_40r, tpr_40r, color='darkorange',marker='_',label='Non Redundand Cluster 40%: {:.2f}'.format(avgmean))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()
####External testing
#path='D:/PhD/Hemo_All_SeQ/'
#External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
#External_features=np.vstack((External_f[:2],External_f[4:6]))
#External_features=np.vstack((External_features,External_f[8:12]))
#External_features=np.vstack((External_features,External_f[15]))
#External_features=np.vstack((External_features,External_f[17]))
#External_features=np.vstack((External_features,External_f[18:20]))
#External_features=np.vstack((External_features,External_f[21:]))
#External_features=torch.FloatTensor(External_features).cuda()
External_score =Hemonet(External_features).cpu().data
print("External_score",External_score)
#External_label=np.append(np.zeros(8),np.ones(7))
External_auc_roc_70r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_70r)
External_fpr_70r, External_tpr_70r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_70r, External_tpr_70r, color='k',marker='o',label='External at 70%:{: .2f}'.format(External_auc_roc_70r))
#plt.grid()
#plt.legend(loc='lower right')
#####
################90%
hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_90.fasta.clstr.sorted')
Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_90.fasta.clstr.sorted')
######
hemo_Folds=chunkify(hemo_CL)
Non_hemo_Folds=chunkify(Non_hemo_CL)
####
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed")
#cv = StratifiedKFold(n_splits=5, shuffle=True)
for i in range(5):
       X_train= np.array([], dtype=np.int64).reshape(0,len(Features[0]))
       X_test, y_train, y_test=[],[],[]
       train_len=0
       for idx, pbag in enumerate(hemo_Folds):
           if i==idx:
               #pdb.set_trace()
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
       for epoch in range(1500):
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
            if loss<0.5:#90%
                break;
#            print("Loss=",loss)
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
#auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
#print("auc_roc",auc_roc)
#avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#print("Average_mean",avgmean)
#plt.figure()                
#plt.plot(LP)
#plt.plot( AUC_list)
#plt.grid() 
#plt.figure()
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
#fpr_40r, tpr_40r, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#plt.plot(fpr_40r, tpr_40r, color='darkorange',marker='_',label='Non Redundand Cluster 40%: {:.2f}'.format(avgmean))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()
####External testing
#path='D:/PhD/Hemo_All_SeQ/'
#External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
#External_features=np.vstack((External_f[:2],External_f[4:6]))
#External_features=np.vstack((External_features,External_f[8:12]))
#External_features=np.vstack((External_features,External_f[15]))
#External_features=np.vstack((External_features,External_f[17]))
#External_features=np.vstack((External_features,External_f[18:20]))
#External_features=np.vstack((External_features,External_f[21:]))
#External_features=torch.FloatTensor(External_features).cuda()
External_score =Hemonet(External_features).cpu().data
print("External_score",External_score)
#External_label=np.append(np.zeros(8),np.ones(7))
External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_90r)
External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_70r, External_tpr_70r, color='g',marker='*',label='External at 90%:{: .2f}'.format(External_auc_roc_90r))
plt.grid()
plt.legend(loc='lower right')
####Save Model###
torch.save(Hemonet.state_dict(),path +'HELMO90PercentRedendencyModel'+"auc_roc"+str(round(auc_roc,3)))
#plt.grid()
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
