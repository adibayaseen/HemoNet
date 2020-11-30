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
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        self.fc1 = nn.Linear(89, 200)
#        self.fc2 = nn.Linear(300, 100)
##        self.fc1 = nn.Linear(1076, 512)
##        self.fc2 = nn.Linear(512, 128)
#        self.fc1 = nn.Linear(1076, 3076)
#        self.fc2 = nn.Linear(3076, 128)
##        self.fc1 = nn.Linear(84,10)
##        self.fc2 = nn.Linear(256, 128)
###        self.fc5 = nn.Linear(1644, 1)
##        self.fc5 = nn.Linear(1644, 1)
##        self.fc4 = nn.Linear(128, 65)
##        self.fc5 = nn.Linear(256, 1)
##        self.fc5 = nn.Linear(3116, 1)
#        self.fc5 = nn.Linear(128, 1)
##        self.fc5 = nn.Linear(1076, 1)
        self.fc5 = nn.Linear(200, 1)
    def forward(self, x):
###        x = F.relu(self.fc1(x))
###        x = F.relu(self.fc2(x))
###        x = F.sigmoid(self.fc1(x))
###        x = F.sigmoid(self.fc2(x))
        x = F.tanh(self.fc1(x))
#        x = F.tanh(self.fc2(x))
##        x = F.tanh(self.fc3(x))
###        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x) 
        return x
path="D:\PhD\Hemo_All_SeQ/"
Hemo_Dict= All_Features(path,path+'hemo_All_seq.txt',1)
Non_hemo_Dict= All_Features(path,path+'Nonhemo_All_seq.txt',1)
Features=np.vstack((list(Hemo_Dict.values()),list(Non_hemo_Dict.values())))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
#Features=F.normalize(Features, p=1, dim=1)
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
Label=np.append(np.ones(len(Hemo_Dict)),np.zeros(len(Non_hemo_Dict)))
print ("Execution Completed")
"""
Leave one Cluster out
"""
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
X_train, X_test, y_train, y_test =[],[],[],[]
##########
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
LP=[]
#print ("Execution Completed")
cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(Features,Label):
    Roc_V=[]
    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
    Hemonet = HemoNet().cuda()
    print(Hemonet)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Hemonet.parameters(),lr=0.01,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    X_train=torch.FloatTensor(X_train).cuda()
    y_train=torch.FloatTensor( y_train).cuda()
    X_test=torch.FloatTensor(X_test).cuda()
    y_test=torch.FloatTensor( y_test).cuda()
    Loss=[]
    for epoch in range(400):
#        pdb.set_trace()
        output = Hemonet(X_train)
        output=torch.squeeze(output, 1)
        target =   y_train
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        print("Loss=",loss)
        params = list(Hemonet.parameters())
        if ( epoch) % 100== 0:
            scheduler.step()
            test_score = Hemonet(X_test)
            test_score=torch.squeeze(test_score, 1)
            Y_score.extend(test_score.cpu().data.numpy())
            Y_t.extend(y_test.cpu().data.numpy())
            test_score.tolist()
            y_test.tolist()
            auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
            print("Epoch",epoch,"auc_roc",auc_roc, "Loss=",np.average(Loss))
            Y_t,Y_score,Loss=[],[],[]
        LP.append(loss.cpu().data.numpy())
        Loss.append(loss.cpu().data.numpy())
    test_score = Hemonet(X_test)
    test_score=torch.squeeze(test_score, 1)
    Y_score.extend(test_score.cpu().data.numpy())
    Y_t.extend(y_test.cpu().data.numpy())
    test_score.tolist()
    y_test.tolist()
    Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(avgmean))
plt.legend(loc='lower right')
plt.grid()
plt.show()
plt.figure()
plt.plot(LP)
plt.grid()  
#####
#cv = StratifiedKFold(n_splits=5, shuffle=True)
#### 
UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
################90 new %
#path='D:/Downloads/'
hemo_CL=Make_Cluster(UNames,path,'new_hemo_90.txt')
Non_hemo_CL=Make_Cluster(UNames,path,'new_Nonhemo_90.txt')
######
################90%
#hemo_CL=Make_Cluster(UNames,path,'All_seq_hemo_90.fasta.clstr.sorted')
#Non_hemo_CL=Make_Cluster(UNames,path,'All_seq_Nonhemo_90.fasta.clstr.sorted')
######
hemo_Folds=chunkify(hemo_CL)
Non_hemo_Folds=chunkify(Non_hemo_CL)
Y_t,Y_score=[],[]
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
       criterion = nn.MSELoss()
       optimizer = optim.Adam(Hemonet.parameters(),lr=0.001,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
       scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
       X_train=torch.FloatTensor(X_train).cuda()
       y_train=torch.FloatTensor( y_train).cuda()
       X_test=torch.FloatTensor(X_test).cuda()
       y_test=torch.FloatTensor( y_test).cuda()
       for epoch in range(15000):
#            pdb.set_trace()
#            print(Hemonet.parameters())
            output = Hemonet(X_train)
            output=torch.squeeze(output, 1)
            target =   y_train
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #        print("Loss=",loss)
            params = list(Hemonet.parameters())
            if ( epoch) % 1000== 0:
                scheduler.step()
                test_score = Hemonet(X_test)
                test_score=torch.squeeze(test_score, 1)
                Y_score.extend(test_score.cpu().data.numpy())
                Y_t.extend(y_test.cpu().data.numpy())
                test_score.tolist()
                y_test.tolist()
                auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
                print("Epoch",epoch,"auc_roc",auc_roc, "Loss=",np.average(LP))
                Y_t,Y_score,Loss=[],[],[]
            LP.append(loss.cpu().data.numpy())
            Loss.append(loss.cpu().data.numpy())
       Loss=[]
       test_score = Hemonet(X_test)
       test_score=torch.squeeze(test_score, 1)
       Y_score.extend(test_score.cpu().data.numpy())
       Y_t.extend(y_test.cpu().data.numpy())
       test_score.tolist()
       y_test.tolist()
       Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(avgmean))
plt.legend(loc='lower right')
plt.grid()
plt.show()
plt.figure()
plt.plot(LP)
plt.grid()  