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
###
from Bio import SearchIO
from roc import roc_VA
#from AAC_Features_Extract import *
#from  Clusterify import *
from blaster import blasterNN
###
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
#    print("Total not found",Notfound)
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
#        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=6, stride=1, padding=1)
#        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(1073, 512)
##        self.fc2 = nn.Linear(512, 128)
#        self.fc1 = nn.Linear(1076, 3076)
#        self.fc2 = nn.Linear(3076, 1076)
###        self.fc1 = nn.Linear(84,10)
###        self.fc2 = nn.Linear(256, 128)
####        self.fc5 = nn.Linear(1644, 1)
###        self.fc5 = nn.Linear(1644, 1)
#        self.fc4 = nn.Linear(1061, 512)
#        self.fc4 = nn.Linear(115, 330)
        
#        self.fc4 = nn.Linear(1024, 512)
##        self.fc5 = nn.Linear(256, 1)
##        self.fc5 = nn.Linear(3116, 1)
#        self.fc5 = nn.Linear(128, 1)
#        self.fc5 = nn.Linear(330, 1)
        self.fc5 = nn.Linear(512, 1)
    def forward(self, x):
###        x = F.relu(self.fc1(x))
###        x = F.relu(self.fc2(x))
#        x = F.sigmoid(self.fc1(x))
#        x = F.sigmoid(self.fc2(x))
#        x = F.tanh(self.fc1(x))
#        x = F.tanh(self.fc2(x))
        x = torch.tanh(self.fc1(x))
#        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x) 
        return x
path="D:\PhD\Hemo_All_SeQ/"
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
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
Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
#Label=np.append(np.ones(len(records_hemo)),-np.zeros(len(records_non_hemo)))
Features=np.vstack((records_hemo,records_non_hemo))
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features)
#path="D:\PhD\Hemo_All_SeQ/"
#records_hemo=np.load(path+'ELMO_1024_Hemo_Features.npy')
#records_non_hemo=np.load(path+'ELMO_1024_NonHemo_Features.npy')
#######For making zeros of NC to -1
#H_NC_mod=records_hemo[:,1024:]
#H_NC_mod[H_NC_mod==0]=-1
#NH_NC_mod=records_non_hemo[:,1024:]
#NH_NC_mod[NH_NC_mod==0]=-1
#records_hemo=records_hemo[:,0:1024]
#records_non_hemo=records_non_hemo[:,0:1024]
#records_hemo=np.hstack((records_hemo,H_NC_mod))
#records_non_hemo=np.hstack((records_non_hemo,NH_NC_mod))
####without N-modifications
#H_C_mod=records_hemo[:,1061:]
#1/0
##H_NC_mod[H_NC_mod==0]=-1
#NH_C_mod=records_non_hemo[:,1061:]
##NH_NC_mod[NH_NC_mod==0]=-1
#records_hemo=records_hemo[:,0:100]
#records_non_hemo=records_non_hemo[:,0:100]
#records_hemo=np.hstack((records_hemo,H_C_mod))
#records_non_hemo=np.hstack((records_non_hemo,NH_C_mod))
######without NC modifications 
#records_hemo=records_hemo[:,0:1024]
#records_non_hemo=records_non_hemo[:,0:1024]
#1/0
#Label=np.append(np.ones(len(records_hemo)),np.zeros(len(records_non_hemo)))
#Label=np.append(np.ones(len(records_hemo)),-1*np.ones(len(records_non_hemo)))
#Features=np.vstack((records_hemo,records_non_hemo))
#Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
#Features=torch.FloatTensor(Features)
#Features=F.normalize(Features, p=1, dim=1)
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
LP=[]
print ("Execution Completed")
all_losses=[]
#seq = 'SEQWENCE' # your amino acid sequence
#embedding = seqvec.embed_sentence( list(seq) ) 
Roc_VA, Y_pred,Y_t,Y_score=[],[],[],[]
avg_roc=[]
AUC_list=[]
Loss=[]
Y_test,Y_p=[],[]
cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(Features,Label):
    Roc_V=[]
    X_train, X_test, y_train, y_test = Features[train_index], Features[test_index], Label[train_index], Label[test_index]
#    R=blasterNN(testfile,trainfile)
    Hemonet = HemoNet().cuda()
    print(Hemonet)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.0001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    X_train=torch.FloatTensor(X_train).cuda()
    y_train=torch.FloatTensor( y_train).cuda()
    X_test=torch.FloatTensor(X_test).cuda()
    y_test=torch.FloatTensor( y_test).cuda()
 
#    for epoch in range(8000)
    for epoch in range(400):
#        pdb.set_trace()
        output = Hemonet(X_train)
        output=torch.squeeze(output, 1)
        target =   y_train
        loss = criterion(output, target)
#        if loss<0.3:
#            break;
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        print("Loss=",loss)
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
#            AUC_list.append(auc_roc)
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
#    #####Shape
#        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
#        # explain predictions of the model on four images
#        e = shap.DeepExplainer(Hemonet,X_train)
#        # ...or pass tensors directly
#        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
#        shap_values = e.shap_values(X_test)
#        
#        # plot the feature attributions
#        shap.summary_plot(shap_values, X_test.cpu().numpy())
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
#1/0
####Shap explainer
#e = shap.DeepExplainer(Hemonet,X_train)
#shap_values = e.shap_values(X_test)
#shap.summary_plot(shap_values, X_test.cpu().numpy())
# visualize the training set predictions
#shap.force_plot(Hemonet(X_test), shap_values, X_test.cpu().numpy())
# shap.summary_plot(shap_values, X_test.cpu().numpy(),max_display=1076)
#1/0
plt.figure()                
plt.plot(LP)
plt.plot( AUC_list)
plt.grid() 

plt.figure()
plt.plot(Loss)
#plt.plot( AUC_list)
plt.grid()
plt.figure() 
#########
################
#################
df_hemo = pd.read_csv(path+'new_HemoPI_predicted_results_hemo_AllSeq_data.csv')
hemo_score_HemoPI=df_hemo[['Prediction']].values
df_non_hemo = pd.read_csv(path+'new_HemoPI_predicted_results_Non_hemo_AllSeq_data.csv')
non_hemo_score_HemoPI=df_non_hemo[['Prediction']].values
#####
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
#plt.plot(fpr, tpr, color='k',marker='s' ,label='Hemopred_results= {:.2f}'.format(HemoPred_auc_roc))
#plt.scatter(Hemopred_fpr, Hemopred_tpr, color='k',marker='s' ,label='Hemopred_results= {:.2f}'.format(HemoPred_auc_roc))
#axvline(x=Hemopred_fpr[1],color='k')
#MCC_Hemopred=MCC_fromAUCROC(Hemopred_tpr[1],Hemopred_fpr[1], len(hemo_score_HemoPred),len(non_hemo_score_HemoPred))
#print("MCC of HemoPred",MCC_Hemopred)
###
##print("HemoPred_accuracy=",HemoPred_accuracy,"\nHemoPred_MCC=",HemoPred_MCC)
#Hemopred_fpr, Hemopred_tpr, thresholds = roc_curve(np.array(HemoPred_Label), np.array(HemoPred_score))
##specificity_Hemopred=0.69#np.min(1-Hemopred_fpr[np.where(Hemopred_fpr-0.3<0.01)])
##Senstivity_Hemopred=np.max(Hemopred_tpr[np.where(Hemopred_fpr-0.3<0.01)])
#MCC_Hemopred=MCC_fromAUCROC(Senstivity_Hemopred,0.3, len(records_hemo),len(records_non_hemo))
#print("MCC Hemopred",HemoPred_MCC)
#print("Senstivity_Hemopred",Senstivity_Hemopred)
#print("specificity _Hemopred",specificity_Hemopred)
#########
auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("Average_mean",avgmean)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
#ELMO_accuracy=Best_accuracy(Y_t, Y_score)
#print("ELMO_Accuracy=",ELMO_accuracy)
plt.plot(fpr, tpr, color='c',marker=',',label='Our Proposed Model:{: .2f}'.format(auc_roc))
#plt.scatter(fpr, tpr, color='c',marker=',',label='Hemo_with_ELMO= {:.2f}'.format(auc_roc))
plt.legend(loc='lower right')
Senstivity_HELMO=np.max(tpr[np.where(fpr-Hemopred_fpr[1]<0.01)])
#plt.grid()
#plt.show()
MCC_HELMO=MCC_fromAUCROC(Senstivity_HELMO,Hemopred_fpr[1], len(records_hemo),len(records_non_hemo))
print("MCC of OUR model",MCC_HELMO)
print("Senstivity of our model",Senstivity_HELMO)
####
from pylab import *
plt.scatter(Hemopred_fpr[1], Hemopred_tpr[1], color='k',marker='s' ,s=50,label='Hemopred: {: .2f}'.format(HemoPred_auc_roc))
axvline(x=Hemopred_fpr[1],color='k')
###
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) Curve')
##plt.legend()
#plt.legend(loc='lower right')
#plt.grid()
###
############
HemoPI_score=np.append(hemo_score_HemoPI,non_hemo_score_HemoPI)
HemoPI_Label=np.append(np.ones(len(hemo_score_HemoPI)),np.zeros(len(non_hemo_score_HemoPI)))
HemoPI_auc_roc=roc_auc_score(np.array(HemoPI_Label), np.array(HemoPI_score))
HemoPI_average_precision_score=average_precision_score(np.array(HemoPI_Label), np.array(HemoPI_score))
print(HemoPI_auc_roc)
HemoPI_accuracy=Best_accuracy(np.array(HemoPI_Label), np.array(HemoPI_score))
#print("HemoPI_accuracy=",HemoPI_accuracy)
fpr, tpr, thresholds = roc_curve(np.array(HemoPI_Label), np.array(HemoPI_score))
plt.plot(fpr, tpr, color='darkorange',marker='.' ,label='HemoPI:{: .2f}'.format(HemoPI_auc_roc))
Senstivity_Hemopred=np.max(tpr[np.where(fpr-Hemopred_fpr[1]<0.01)])
MCC_HemoPI=MCC_fromAUCROC(Senstivity_Hemopred,Hemopred_fpr[1], len(hemo_score_HemoPI),len(non_hemo_score_HemoPI))
print("MCC of HemoPI",MCC_HemoPI)
print("Senstivity_Hemopred",Senstivity_Hemopred)
##################
seq1,name1=Seq_Name_list(path+'hemo_All_seq.txt')
seq2,name2=Seq_Name_list(path+'Nonhemo_All_seq.txt')
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
        else:
            Y_Predict.append( np.random.choice(A))
    Y_t.extend(y_test)
MCC=matthews_corrcoef(Y_t,Y_Predict)
AUC_ROC_blast=roc_auc_score(np.array(Y_t),np.array(Y_Predict))
PR_Blast=average_precision_score(np.array(Y_t),np.array(Y_Predict))
fpr_b,tpr_b,t_b=roc_curve(np.array(Y_t),np.array(Y_Predict))
Senstivity_Blast=np.max(tpr_b[np.where(fpr_b-Hemopred_fpr[1]<0.01)])
print("MCC of Baseline:",MCC,"AUC_ROC of baseline",AUC_ROC_blast,"PR",PR_Blast)
plt.scatter(fpr_b[1], tpr_b[1], color='b',marker='^' ,s=150,label='Baseline:{: .2f}'.format(AUC_ROC_blast))
#############
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.legend(loc='lower right')
plt.grid()
plt.savefig('All Method Comparison.png', dpi=300)
################Testing Hemo1,2,3 problem is that id are not mentioned so how to calculate NC terminus
#records_hemo1=np.load(path+'server_hemo1_1024_data_Features.npy')
#records_non_hemo1=np.load(path+'server_Non_hemo1_1024_data_Features.npy')
