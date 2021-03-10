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
from sklearn.metrics import accuracy_score
from Bio import SearchIO
from roc import roc_VA
from new_AAC_Features_Extract import *
from blaster import blasterNN
from sklearn.metrics import accuracy_score
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn import metrics
from Bio import SeqIO
import torch.nn.functional as F
import pickle
from sklearn.model_selection import KFold,StratifiedKFold
import shap
def NN(idx):
    P = {}
    Notfound=0
    for qresult in SearchIO.parse(idx+".pblast.txt","blast-text"):
            if len(qresult):
                P[qresult.id]=qresult[int(np.argmin([hsp.evalue for hit in qresult for hsp in hit]))].id
            else:
                Notfound+=1
                P[qresult.id]=100.0
    print("Total not found",Notfound)
    return P
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
def NaturalAAC(AllSeq, Score, Name):
    Natural_seq,Natural_score, Natural_name=[],[],[]
    for index in range(len(AllSeq)):
        Seq=''
        for AAC in range(len(AllSeq[index][0])):
            if AllSeq[index][0][AAC] in 'acdefghiklmnpqrstvwy':
                Seq=''
                break
                
            else:
                Seq=Seq+AllSeq[index][0][AAC]
        if len(Seq)>0 :
            Natural_seq.append(Seq)
            Natural_score.append(Score[index][0])
            Natural_name.append(Name[index][0])
    return Natural_seq,Natural_score, Natural_name
path="D:\PhD\Hemo_All_SeQ/"
"""
SeqVec Features
"""
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
"""
NC Smiles Features with 1024 features dimension for both N and C terminal modifications
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
###Non hemo
N_non_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_non_hemo]
C_non_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
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
        if loss<0.3:
            break;
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
#    
    test_score = Hemonet(X_test)
    test_score=torch.squeeze(test_score, 1)
    print("auc_roc",auc_roc)
    Y_p.extend(test_score.cpu().data.numpy())
    Y_test.extend(y_test.cpu().data.numpy())
    test_score.tolist()
    y_test.tolist()
    auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
    Roc_VA.append((test_score.cpu().data.numpy(),list(y_test)))
plt.figure()                
plt.plot(LP)
plt.plot( AUC_list)
plt.grid() 

plt.figure()
plt.plot(Loss)
plt.grid()
plt.figure() 
"""
Comparison with existing model HemoPI
"""
df_hemo = pd.read_csv(path+'new_HemoPI_predicted_results_hemo_AllSeq_data.csv')
hemo_score_HemoPI=df_hemo[['Prediction']].values
hemo_names_HemoPI=df_hemo[['Name']].values
hemo_HemoPI_seq=df_hemo[['Sequence']].values
Natural_hemo_HemoPI_seq,Natural_hemo_HemoPI_score, Natural_hemo_HemoPI_name=NaturalAAC(hemo_HemoPI_seq, hemo_score_HemoPI, hemo_names_HemoPI)
df_non_hemo = pd.read_csv(path+'new_HemoPI_predicted_results_Non_hemo_AllSeq_data.csv')
non_hemo_score_HemoPI=df_non_hemo[['Prediction']].values
non_hemo_names_HemoPI=df_non_hemo[['Name']].values
non_hemo_HemoPI_seq=df_non_hemo[['Sequence']].values
Natural_non_hemo_HemoPI_seq,Natural_non_hemo_HemoPI_score, Natural_non_hemo_HemoPI_name=NaturalAAC(non_hemo_HemoPI_seq,non_hemo_score_HemoPI,non_hemo_names_HemoPI)
#####
"""
Comparison with existing model HemoPred
"""
df_hemo = pd.read_csv(path+'new_HemoPred_predicted_results_hemo_AllSeq_data.csv')
hemo_score_HemoPred=df_hemo[['Prediction']].values
df_non_hemo = pd.read_csv(path+'new_HemoPred_predicted_results_NON_hemo_AllSeq_data.csv')
non_hemo_score_HemoPred=df_non_hemo[['Prediction']].values
######
HemoPred_score=np.append(hemo_score_HemoPred,non_hemo_score_HemoPred)
HemoPred_Label=np.append(np.ones(len(hemo_score_HemoPred)),np.zeros(len(non_hemo_score_HemoPred)))
HemoPred_auc_roc=roc_auc_score(np.array(HemoPred_Label), np.array(HemoPred_score))
print(HemoPred_auc_roc)
HemoPred_accuracy=accuracy_score(np.array(HemoPred_Label), np.array(HemoPred_score))
HemoPred_MCC=matthews_corrcoef(np.array(HemoPred_Label),np.array(HemoPred_score))
print("HemoPred_accuracy=",HemoPred_accuracy,"\nHemoPred_MCC=",HemoPred_MCC)
Hemopred_fpr, Hemopred_tpr, thresholds = roc_curve(np.array(HemoPred_Label), np.array(HemoPred_score))
Hemopred_average_precision=average_precision_score(np.array(HemoPred_Label), np.array(HemoPred_score))
#########
auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
plt.plot(fpr, tpr, color='c',marker=',',label='HemoNet:{: .2f}%'.format(auc_roc*100))
plt.legend(loc='lower right')
Senstivity_HELMO=np.max(tpr[np.where(fpr-Hemopred_fpr[1]<0.001)])
MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO,Hemopred_fpr[1], len(records_hemo),len(records_non_hemo))
print("MCC of OUR model",MCC_HELMO)
print("Senstivity of our model",Senstivity_HELMO)
####
from pylab import *
plt.scatter(Hemopred_fpr[1], Hemopred_tpr[1], color='k',marker='X' ,s=120,label='Hemopred')#.format(HemoPred_auc_roc*100))
axvline(x=Hemopred_fpr[1],color='k')
HemoPI_score=np.append(hemo_score_HemoPI,non_hemo_score_HemoPI)
HemoPI_Label=np.append(np.ones(len(hemo_score_HemoPI)),np.zeros(len(non_hemo_score_HemoPI)))
HemoPI_auc_roc=roc_auc_score(np.array(HemoPI_Label), np.array(HemoPI_score))
HemoPI_average_precision_score=average_precision_score(np.array(HemoPI_Label), np.array(HemoPI_score))
print(HemoPI_auc_roc)
HemoPI_accuracy=Best_accuracy(np.array(HemoPI_Label), np.array(HemoPI_score))
fpr, tpr, thresholds = roc_curve(np.array(HemoPI_Label), np.array(HemoPI_score))
plt.plot(fpr, tpr, color='blue',marker="8" ,label='HemoPI(All peptides):{: .2f}%'.format(HemoPI_auc_roc*100))
######For natural AAC only Hemopi
HemoPI_Natural_score=np.append(Natural_hemo_HemoPI_score,Natural_non_hemo_HemoPI_score)
HemoPI_Natural_Label=np.append(np.ones(len(Natural_hemo_HemoPI_score)),np.zeros(len(Natural_non_hemo_HemoPI_score)))
HemoPI_Natural_auc_roc=roc_auc_score(np.array(HemoPI_Natural_Label), np.array(HemoPI_Natural_score))
HemoPI_Natural_average_precision_score=average_precision_score(np.array(HemoPI_Natural_Label), np.array(HemoPI_Natural_score))
print(HemoPI_Natural_auc_roc)
HemoPI_Natural_accuracy=Best_accuracy(np.array(HemoPI_Natural_Label), np.array(HemoPI_Natural_score))
#print("HemoPI_accuracy=",HemoPI_accuracy)
Natural_fpr, Natural_tpr, thresholds = roc_curve(np.array(HemoPI_Natural_Label), np.array(HemoPI_Natural_score))
Senstivity_Natural_HemoPI=np.max(Natural_tpr[np.where(Natural_fpr-Hemopred_fpr[1]<0.01)])
Natural_MCC_HemoPI=MCC_fromAUCROC(Senstivity_Natural_HemoPI,Hemopred_fpr[1], len(hemo_score_HemoPI),len(non_hemo_score_HemoPI))
print("MCC of Natural HemoPI",Natural_MCC_HemoPI)
print("Senstivity_Hemopred",Senstivity_Natural_HemoPI)
###################################3
Senstivity_Hemopred=np.max(tpr[np.where(fpr-Hemopred_fpr[1]<0.01)])
MCC_HemoPI=MCC_fromAUCROC(Senstivity_Hemopred,Hemopred_fpr[1], len(hemo_score_HemoPI),len(non_hemo_score_HemoPI))
print("MCC of HemoPI",MCC_HemoPI)
print("Senstivity_Hemopred",Senstivity_Hemopred)
##################
seq1,name1=Seq_Name_list(path+'Hemo_dict.txt')
seq2,name2=Seq_Name_list(path+'Non_Hemo_dict.txt')
seqs=np.append(seq1,seq2)
names=np.append(name1,name2)
L=np.ones(len(name1))
NL=-1*np.ones(len(name2))
Label=np.append(L,NL)
A=[1,-1]
print ("Execution Completed")
cv = StratifiedKFold(n_splits=5, shuffle=True)
Y_t,Y_Predict=[],[]
fold,start=0,0
for train_index, test_index in cv.split(seqs,Label):
    fold+=1
    X_train, X_test, y_train, y_test = seqs[train_index], seqs[test_index], Label[train_index], Label[test_index]
    R=NN(str(fold))
    for v in R:
        if len(str(R[v]).split('#'))>1:
            Y_Predict.append(float(str(R[v]).split('#')[1]))
            Y_t.append(float(str(v).split('#')[1]))
        else:
            Y_Predict.append( np.random.choice(A))
            Y_t.append(float(str(v).split('#')[1]))
MCC=matthews_corrcoef(Y_t,Y_Predict)
AUC_ROC_blast=roc_auc_score(np.array(Y_t),np.array(Y_Predict))
PR_Blast=average_precision_score(np.array(Y_t),np.array(Y_Predict))
fpr_b,tpr_b,t_b=roc_curve(np.array(Y_t),np.array(Y_Predict))
Senstivity_Blast=np.max(tpr_b[np.where(fpr_b-Hemopred_fpr[1]<0.01)])
print("MCC of Baseline:",MCC,"AUC_ROC of baseline",AUC_ROC_blast,"PR",PR_Blast)
plt.scatter(fpr_b[1], tpr_b[1], color='m',marker='^' ,s=150,label='Baseline')#:{: .2f}'.format(AUC_ROC_blast))
#############
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.legend(loc='lower right')
plt.grid()
plt.savefig('All Method Comparison.png', dpi=300)
