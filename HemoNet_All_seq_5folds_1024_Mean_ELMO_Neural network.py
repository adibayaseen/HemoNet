#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:11:33 2019

@author: AdibaYaseen
"""
from sklearn.metrics import matthews_corrcoef
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.svm import SVC
import torch
import pandas as pd
from Bio import SearchIO
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pickle
from roc import roc_VA
from blaster import blasterNN
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score as auc_roc
from sklearn import metrics
from Bio import SeqIO
import torch.nn.functional as F
from sklearn.model_selection import KFold,StratifiedKFold
def All_Features(seq):
    N_terminous=pickle.load(open(path+'OHE_nTerminus_All_Dict.npy', "rb"))
    C_terminous=pickle.load(open(path+'OHE_cTerminus_All_Dict.npy', "rb"))
    ##########
    embedding=seqvec.embed_sentence( list(seq) ) 
    F=torch.tensor(embedding).sum(dim=0).mean(dim=0)
    name=records[i].id.split('#')[0] 
    Names.append(name)
    return Features,Names
def Seq_Name_list(file_name):
        names=[]
        sequences=[]
        All_seq = list(SeqIO.parse(file_name,'fasta'))
        for i in range(0,len(All_seq)): 
            sequences.append(str((All_seq[i].seq)))
            names.append(All_seq[i].id.split('#')[0])
        return sequences,names
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def Best_accuracy(y_ts, predictions):
    f, t, a=roc_curve(y_ts, predictions)
    AN=sum(x<0 for x in y_ts)
    AP=sum(x>0 for x in y_ts)
    TN=(1.0-f)*AN
    TP=t*AP
    Acc2=(TP+TN)/len(y_ts)
    acc=max(Acc2)
#    print ('best accuracy=',acc )
    return acc
def MCC_fromAUCROC(TPR,FPR, P,N):
    TP=TPR*P
    FN=((1-TPR)*TP)/TPR
    FP=FPR*N
    TN=(FP*(1-FPR))/FPR
    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return MCC
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        self.fc1 = nn.Linear(1076, 512)
        self.fc5 = nn.Linear(512, 1)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc5(x) 
        return x
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
plt.figure()                
plt.plot(LP)
plt.plot( AUC_list)
plt.grid() 

plt.figure()
plt.plot(Loss)
auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
plt.plot(fpr, tpr, color='c',marker=',',label='Our Proposed Model:{: .2f}'.format(auc_roc))
plt.legend(loc='lower right')
Senstivity_HELMO=np.max(tpr[np.where(fpr-0.3<0.01)])
#plt.grid()
#plt.show()
MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO,0.3, len(records_hemo),len(records_non_hemo))
print("MCC of OUR model",MCC_HELMO)
print("Senstivity of our model",Senstivity_HELMO)
####
path='D:/PhD/Hemo_All_SeQ/HemoPImodPdbfiles/'
hemopimod_hemo_features=np.load(path+"HemoPImod_Hemo_Features.npy")
hemopimod_Nonhemo_features=np.load(path+"HemoPImod_NonHemo_Features.npy")
hemopimod_Label=np.append(np.ones(len(hemopimod_hemo_features)),-1*np.ones(len(hemopimod_Nonhemo_features)))
#Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
hemopimod_Features=np.vstack((hemopimod_hemo_features,hemopimod_Nonhemo_features))
nc_mod=np.zeros((len(hemopimod_Features),52))
hemopimod_Features=np.hstack((hemopimod_Features,nc_mod))
hemopimod_Features=(hemopimod_Features-np.mean(hemopimod_Features, axis = 0))/(np.std(hemopimod_Features, axis = 0)+0.000001)
hemopimod_Features=torch.FloatTensor(hemopimod_Features).cuda()
hemopimod_score =Hemonet(hemopimod_Features).cpu().data
hemopimod_auc_roc=roc_auc_score(np.array(hemopimod_Label),np.array(hemopimod_score ))
print("hemopimod_auc_roc",hemopimod_auc_roc)
fpr, tpr, thresholds = roc_curve(np.array(hemopimod_Label),np.array(hemopimod_score ))
plt.plot(fpr, tpr, color='c',marker=',',label='External:{: .2f}'.format(hemopimod_auc_roc))
plt.legend(loc='lower right')
1/0
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
plt.legend(loc='lower right')
torch.save(Hemonet.state_dict(),path +'HELMOWithoutRedendencyREmovalModel'+"auc_roc"+str(round(auc_roc,3)))