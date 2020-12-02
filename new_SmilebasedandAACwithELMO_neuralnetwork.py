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
from  Results import *
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
from platt import *
def Clinical_data_result(Classifier,FeatureName):
    ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
    DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
    DRAMP_Clinical_NC=np.zeros((28,2048))
    ALL_DRAMP_Clinical_data_Features=np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),DRAMP_Clinical_data_AAC))
    FeatureDict={'ELMO':ELMO_DRAMP_Clinical_data_Features,
                 'NC':DRAMP_Clinical_NC,
                 'AAC':DRAMP_Clinical_data_AAC,
                 'ELMO_NC':np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),
                 'AAC_NC':np.hstack((DRAMP_Clinical_data_AAC,DRAMP_Clinical_NC)),
                 'ELMO_Smile':np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),
                 'AAC_Smile':np.hstack((DRAMP_Clinical_data_AAC,DRAMP_Clinical_NC)),
                 'ELMO_AAC_Smile': ALL_DRAMP_Clinical_data_Features
                 }
    Features= FeatureDict[FeatureName]
    Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
    Score=Classifier.predict_proba(Features)[:,1]
    meanScore=np.mean(Score)
    plt.figure()
    plt.hist(np.sort(Score),bins=len(Score))
    plt.grid()
    print("Average score ALL features",meanScore)
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
class HemoNet(nn.Module):
    def __init__(self):
        super(HemoNet, self).__init__()
        
#        self.fc4 = nn.Linear(1024, 2150)
        self.fc4 = nn.Linear(3112,2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 100)
        self.fc7 = nn.Linear(100, 1)
        """
        self.fc4 = nn.Linear(1076, 2150)
        self.fc5 = nn.Linear(2150, 512)
        self.fc6 = nn.Linear(512, 1)
        """


    def forward(self, x):
#        x = torch.selu(self.fc4(x))
#        torch.sigmoid
#        torch.softmax
#        torch.selu
        x=torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
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
#    pdb.set_trace()
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
            if loss<0.1:
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
path="D:\PhD\Hemo_All_SeQ/"
#path='/content/drive/My Drive/ELMO_Embedding/'#Colab
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
ELMO_Hemo_Dict=dict(zip(Names_hemo,records_hemo))
ELMO_Non_hemo_Dict=dict(zip(Names_non_hemo,records_non_hemo))
#N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
#C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
#N_hemo=[N_terminous[int(n)] for n in Names_hemo]
#C_hemo=[C_terminous[int(n)] for n in Names_hemo]
#NC_hemo=np.hstack((N_hemo,C_hemo))
"""
NC Smiles Features
"""
N_terminous_names=pickle.load(open(path+'nTerminus_All_Dict.npy', "rb"))
C_terminous_names=pickle.load(open(path+'cTerminus_All_Dict.npy', "rb"))
#####
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
NC_hemo_dict=dict(zip(Names_hemo,NC_hemo))
#NC_hemo=(NC_hemo-np.mean(NC_hemo, axis = 0))/(np.std(NC_hemo, axis = 0)+0.000001)
#records_hemo=(records_hemo-np.mean(records_hemo, axis = 0))/(np.std(records_hemo, axis = 0)+0.000001)
#records_hemo=np.hstack((records_hemo,NC_hemo))
##Non hemo
N_non_hemo=[N_terminous_Smiles_features[N_terminous_names[int(n)]][1] for n in Names_non_hemo]
C_non_hemo=[C_terminous_Smiles_features[C_terminous_names[int(n)]][1] for n in Names_non_hemo]
#N_non_hemo=[N_terminous[int(n)] for n in Names_non_hemo]
#C_non_hemo=[C_terminous[int(n)] for n in Names_non_hemo]
NC_non_hemo=np.hstack((N_non_hemo,C_non_hemo))
NC_non_hemo_dict=dict(zip(Names_non_hemo,NC_non_hemo))
#NC_features=np.vstack((NC_hemo,NC_non_hemo))
NC_features=np.vstack((list(NC_hemo_dict.values()),list(NC_non_hemo_dict.values())))
#ELMO_features=np.vstack((records_hemo,records_non_hemo))
NC_features=(NC_features-np.mean(NC_features, axis = 0))/(np.std(NC_features, axis = 0)+0.000001)
ELMO_features=np.vstack((list(ELMO_Hemo_Dict.values()),list(ELMO_Non_hemo_Dict.values())))
ELMO_features=(ELMO_features-np.mean(ELMO_features, axis = 0))/(np.std(ELMO_features, axis = 0)+0.000001)
####
###AAC Features
mer=1
print("AAC features",mer,"mer")
AAC_hemo=list( All_FeaturesWithoutNC(path,path+'hemo_All_seq.txt',mer).values())
AAC_non_hemo= list( All_FeaturesWithoutNC(path,path+'Nonhemo_All_seq.txt',mer).values())
AAC_Features=np.vstack((AAC_hemo,AAC_non_hemo))
AAC_Features=(AAC_Features-np.mean(AAC_Features, axis = 0))/(np.std(AAC_Features, axis = 0)+0.000001)
#Features=torch.FloatTensor(Features)
Features=np.hstack((ELMO_features,NC_features))
Features=np.hstack((Features,AAC_Features))
Hemo_Dict=dict(zip(Names_hemo,np.hstack((np.hstack((list(ELMO_Hemo_Dict.values()),
list(NC_hemo_dict.values()))),AAC_hemo))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((np.hstack((list(ELMO_Non_hemo_Dict.values()),
list(NC_non_hemo_dict.values()))),AAC_non_hemo))))
"""
##only 1mer+ELMO
Features=np.hstack((ELMO_features,AAC_Features))#onli 1mer+ELMO
Hemo_Dict=dict(zip(Names_hemo,np.hstack((list(ELMO_Hemo_Dict.values()),AAC_hemo))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((list(ELMO_Non_hemo_Dict.values()),AAC_non_hemo))))
###############
"""

Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
#Label=np.append(np.ones(len(Hemo_Dict)),np.zeros(len(Non_hemo_Dict)))
Features=torch.FloatTensor(Features)
##
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA=[],[],[],[],[],[]
print ("Execution Completed")

UNames=new_RemoveDuplicates(path,'new_HemoltkAndDBAASP_all_seq.fasta.clstr.sorted')
###Total 10 times mean std  
##ResultMeanStd(Features,Label,120,11)#1mer
#ResultMeanStd(Features,Label,130,15)#2mer
##ResultMeanStd(Features,Label,135,5)#ELMO
#1/0
percent='90'
#ResultMeanStd(path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#90ELMO
Y_t,Y_score, Y_pred,names,avg_roc,Roc_VA,LP,Y_p,Y_test=[],[],[],[],[],[],[],[],[]
LP=[]
AUC_list=[]
print ("Execution Completed\n Total feature dimension=",len(Features[0]))

#cv = StratifiedKFold(n_splits=5, shuffle=True)
for i in range(5):
       X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
       print("Total Train:",len(X_train),"Total test",len(X_test))
       Hemonet = HemoNet().cuda()
       print(Hemonet)
       Loss=[]
       criterion = nn.MSELoss()
       optimizer = optim.Adam(Hemonet.parameters(),lr=0.0001,weight_decay=0.00001)#0.69 for 1mer single layer#, weight_decay=0.01, betas=(0.9, 0.999))
#       scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
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
            if loss<0.15:#0.45#new#0.55:#90%#0.74 #0.78 average for external
                break;
            if ( epoch) % 10== 0:
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
fpr, tpr, thresholds = roc_curve(np.array(Y_test), np.array(Y_p))
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
MCC=MCC_fromAUCROC(Senstivity, threshold, len(records_hemo),len(records_non_hemo))
print("senstivity specificity MCC,AUCROC and PR of OUR model",Senstivity,specificity, MCC,avgmean,average_precision_score(np.array(Y_test), np.array(Y_p)))
####Clinical using ELMO
ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
DRAMP_Clinical_NC=np.zeros((28,2048))
ALL_DRAMP_Clinical_data_Features=np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),DRAMP_Clinical_data_AAC))
ALL_DRAMP_Clinical_data_Features=(ALL_DRAMP_Clinical_data_Features-np.mean(ALL_DRAMP_Clinical_data_Features, axis = 0))/(np.std(ALL_DRAMP_Clinical_data_Features, axis = 0)+0.000001)
###
ALL_DRAMP_Clinical_data_Features=torch.FloatTensor(ALL_DRAMP_Clinical_data_Features).cuda()
DRAMP_Clinical_data_score =Hemonet(ALL_DRAMP_Clinical_data_Features).cpu().data.numpy()
print("Average score ALL features",np.mean(DRAMP_Clinical_data_score))
Classifier=Hemonet
V=Classifier(X_train).cpu().data.numpy()
L=y_train.cpu().data.numpy()
A,B = plattFit(V,L)
#A,B = plattFit(X_train,y_train) #rescling-coefficients
#print('A =',A,'B =',B)
pp = sigmoid(V,A,B)
from sklearn.metrics import roc_auc_score
#print("Print Ranges:")

#print("Original:",np.min(V),np.max(V))
#print("Rescaled:",np.min(pp),np.max(pp))
print("Calculate AUC-ROC (should not change):")
print(roc_auc_score(L,pp))
print(roc_auc_score(L,V))
V=DRAMP_Clinical_data_score
rasacaled_clinical = sigmoid(V,A,B)
print("Rescaled clinical",np.mean(rasacaled_clinical))
#####DRAMP Predicted from Hemopred
HemoPred_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPred_predicted_DRAMP_Clinical_data.csv')
Score_HemoPred_predicted_DRAMP_Clinical_data=HemoPred_predicted_DRAMP_Clinical_data[['Prediction']].values
#plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data,bins=len(Score_HemoPred_predicted_DRAMP_Clinical_data),color='r')
#plt.plot(np.histogram(Score_HemoPred_predicted_DRAMP_Clinical_data,bins=14)[0],color='r',marker='.',label='HemoPred:{: .2f}'.format(np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data)))
#plt.figure()
#plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data)
print("Average score HemoPred",np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data))
#plt.grid()
####DRAMP HEmoPI predicted results
HemoPI_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HemoPI_predicted_DRAMP_Clinical_data.csv')
Score_HemoPI_predicted_DRAMP_Clinical_data=HemoPI_predicted_DRAMP_Clinical_data[['PROB Score']].values
#plt.plot(np.histogram(Score_HemoPI_predicted_DRAMP_Clinical_data,bins=14)[0],color='g',marker='D',label='HemoPI:{: .2f}'.format(np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data)))
#plt.figure()
#plt.hist(Score_HemoPI_predicted_DRAMP_Clinical_data)
print("Average score HemoPI",np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data))
#plt.grid()
####HaPPeNN
####DRAMP HEmoPI predicted results
HaPPeNN_predicted_DRAMP_Clinical_data = pd.read_csv(path+'HaPPeNN_predicted_DRAMP_Clinical_data.csv')
Score_HaPPeNN_predicted_DRAMP_Clinical_data=HaPPeNN_predicted_DRAMP_Clinical_data[['PROB']].values
#plt.figure()
#plt.plot(Score_HaPPeNN_predicted_DRAMP_Clinical_data,color='b',marker='+',label='HaPPeNN :{: .2f}'.format(np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data)))
#plt.hist(Score_HaPPeNN_predicted_DRAMP_Clinical_data)
print("Average score HaPPeNN total 21/28",np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data))
#plt.plot(np.histogram(Score_HaPPeNN_predicted_DRAMP_Clinical_data,bins=14)[0],color='b',label='HaPPeNN :{: .2f}'.format(np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data)))
plt.grid()
####
plt.figure()
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
plt.hist(rasacaled_clinical,density=True, color='gray',label='Our method:{: .2f}'.format(np.mean(rasacaled_clinical)))
plt.grid()
plt.legend(loc='upper center')
plt.savefig('Ourmodel_ClinicalData.png', dpi=300)
plt.figure()
plt.hist(Score_HemoPred_predicted_DRAMP_Clinical_data, density=True,
 linestyle='--',color='darkseagreen', linewidth=3,label='HemoPred:{: .2f}'.format(np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data)))
plt.grid()
plt.legend(loc='upper center')
plt.savefig('HemoPrd_ClinicalData.png', dpi=300)
plt.figure()
plt.hist(Score_HemoPI_predicted_DRAMP_Clinical_data, density=True,
linestyle='solid',linewidth=3,label='HemoPI:{: .2f}'.format(np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data)))
plt.grid()
plt.legend(loc='upper right')
plt.savefig('HemoPI_ClinicalData.png', dpi=300)
#####
ELMO_F=np.load(path+'Hemolytic_External_validation_Features.npy')
ELMO_Features= ELMO_F[0:6]
ELMO_Features= np.vstack((ELMO_Features,ELMO_F[7:14]))
ELMO_Features= np.vstack((ELMO_Features,ELMO_F[15:]))
mer=1
AAC=list( All_FeaturesWithoutNC(path,path+'External.txt',mer).values())
AAC2=list( All_FeaturesWithoutNC(path,path+'External.txt',2).values())
NC=np.zeros((13,49))
Smiles_NC=np.zeros((13,2048))
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
             'ELMO_AAC_Smile': np.hstack((np.hstack(( ELMO_Features,Smiles_NC)),AAC))
             }
FeatureName='ELMO_AAC_Smile'
Features= FeatureDict[FeatureName]
Features=(Features-np.mean(Features, axis = 0))/(np.std(Features, axis = 0)+0.000001)
Features=torch.FloatTensor(Features).cuda()
Score=Classifier(Features).cpu().data.numpy()
External_label=np.append(np.zeros(8),np.ones(5))
AUCROC=roc_auc_score(np.array(External_label),np.array(Score))
print("External AUCROC",AUCROC)
#plt.figure()
#plt.hist(Score_HaPPeNN_predicted_DRAMP_Clinical_data, density=True,
#linestyle='-', color='gray',linewidth=3,label='HaPPeNN :{: .2f}'.format(np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data)))
#plt.legend(loc='upper center')
##plt.legend(frameon=False)
#plt.show()
#plt.grid()
#plt.savefig('HaPPeNN_ClinicalData.png', dpi=300)
###
#import seaborn as sb
#plt.legend(loc='upper right')
#sb.distplot(ALL_DRAMP_Clinical_data_score, hist=False,
#color='m',label='Our method:{: .2f}'.format(np.mean(ALL_DRAMP_Clinical_data_score)))
#sb.distplot(Score_HemoPred_predicted_DRAMP_Clinical_data,
#hist=False,color='r',label='HemoPred:{: .2f}'.format(np.mean(Score_HemoPred_predicted_DRAMP_Clinical_data)))
#sb.distplot(Score_HemoPI_predicted_DRAMP_Clinical_data,hist=False,color='g',
#label='HemoPI:{: .2f}'.format(np.mean(Score_HemoPI_predicted_DRAMP_Clinical_data)))
#sb.distplot(Score_HaPPeNN_predicted_DRAMP_Clinical_data,hist=False,color='b',
#label='HaPPeNN :{: .2f}'.format(np.mean(Score_HaPPeNN_predicted_DRAMP_Clinical_data)))
#plt.grid()
#####
#train_score = Hemonet(X_train)
#V=train_score.cpu().data.numpy()
#L=y_train.cpu().data.numpy()
#A,B = plattFit(V,L)
##A,B = plattFit(X_train,y_train) #rescling-coefficients
#print('A =',A,'B =',B)
#pp = sigmoid(V,A,B)
#from sklearn.metrics import roc_auc_score
#print("Print Ranges:")
#
#print("Original:",np.min(V),np.max(V))
#print("Rescaled:",np.min(pp),np.max(pp))
#print("Calculate AUC-ROC (should not change):")
#print(roc_auc_score(L,pp))
#print(roc_auc_score(L,V))
#V=ALL_DRAMP_Clinical_data_score
##L=np.zeros(28)
#pp = sigmoid(V,A,B)
####External testing
#path='D:/PhD/Hemo_All_SeQ/'
#External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
#mer=1
#print("External AAC features",mer,"mer")
#External_AAC=list( All_FeaturesWithoutNC(path,path+'Hemolytic_External_validation.txt',mer).values())
#External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
##Features=torch.FloatTensor(Features)
##External_features=External_f
#
#ELmo_External_features=External_f[:,0:1024]
######1mer+ELMO
##External_F=np.hstack((ELmo_External_features,External_AAC))
#
#
#
######
#External_NC=np.zeros((24,2048))
#External_Features=np.hstack((ELmo_External_features,External_NC))
#External_F=np.hstack((External_Features,External_AAC))
#####1mer+ELMO
##External_Features=np.hstack((ELmo_External_features,External_AAC))
#External_Features=np.vstack((External_F[:2],External_F[4:6]))
#External_Features=np.vstack((External_Features,External_F[8:12]))
#External_Features=np.vstack((External_Features,External_F[15]))
#External_Features=np.vstack((External_Features,External_F[17]))
#External_Features=np.vstack((External_Features,External_F[18:20]))
#External_Features=np.vstack((External_Features,External_F[21:]))
#####
##External_Features=External_F
##External_label=np.append(np.zeros(12),np.ones(12))
##
#External_Features=torch.FloatTensor(External_Features).cuda()
#External_score =Hemonet(External_Features).cpu().data
##External_score =svr_lin.predict_proba(External_Features)[:,1]
#print("External_score",External_score)
#External_label=np.append(np.zeros(8),np.ones(7))
##External_label=np.append(np.zeros(12),np.ones(12))
#External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
#print("External_auc_roc",External_auc_roc_90r)
#External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
#plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation at 90%:{: .2f}'.format(External_auc_roc_90r))
##plt.grid()
#plt.legend(loc='lower right')
#plt.grid()
####
##HemoPred_external_pred=[1,0,0,0,
##1,1,1,0,#5 randaom 5-12
###13,14
##1,#15
##0,0,1,#16-21 examples
##1,0,0]#22-24 random
##HemoPred_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPred_external_pred))
##print("HemoPred External_auc_roc=",HemoPred_External_auc_roc)
##from sklearn.metrics import accuracy_score
##score = accuracy_score(np.array(External_label),np.array(HemoPred_external_pred))
##print("Accuracy_HemoPred",score)
#####HemoPI
##HemoPi_external_pred=[0.51,0.47,0.5,0.45,0.44,0.44,0.48,0.44,0.44,0.46,0.47,0.53,0.44,0.44,0.44]
##HemoPi_External_auc_roc=roc_auc_score(np.array(External_label),np.array(HemoPi_external_pred))
##print("HemoPi External validation ",HemoPi_External_auc_roc)
#######
