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
from roc import roc_VA
from platt import *
def All_FeaturesWithoutNC(path,file_name,k):
    records = list(SeqIO.parse(file_name, "fasta"))
    Features={}
    for i in range(0,len(records)):
        F=[]
        Dict={}
        seq=str((records[i].seq))
        seq=seq.replace('U','')#2
        seq=seq.replace('u','')#2
        seq=seq.replace('b','')#2
        seq=seq.replace('(','')#2
        seq=seq.replace(')','')#2
        seq=seq.replace('Ψ[CH2NH]','')#2
        seq=seq.replace('¨[CH2NH]','')
        seq=seq.replace('Δ','')#2
        seq=seq.replace('/','')#1
        seq=seq.replace('”','')#1
        seq=seq.replace('Î','I')#1
        seq=seq.replace('Ψ[CH2OCONH]','')#1
        seq=seq.replace('¨[CH2OCONH]','')#1
        name=records[i].id.split('_')[0]
        Dict[name]=seq
        F=list(feature_extract(Dict,k).values())
        F=list(F[0][0])
        F=(F)/(np.sum(F, axis = 0))
        Features[name]=F
    return Features#,Names
import numpy as np
from Bio import SeqIO
import itertools
import pickle
import random
def feature_extract(recseq,k):
    dfv={} 
#    all_kmers=[''.join (i) for i in itertools.product("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy",repeat=k)]  
    #All Sequence 23
    all_kmers=[''.join (i) for i in itertools.product("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy",repeat=k)] 
    for i in recseq.keys():
        dic2 = dict(zip(all_kmers, np.zeros(len(all_kmers))))
        id1=i
        seq=recseq[i]
#        seq=seq.replace('X','')
#        seq=seq.replace('x','')
        ####
        seq=seq.replace('X',random.choice('ACDEFGHIKLMNPQRSTVWY'))
        seq=seq.replace('x',random.choice('acdefghiklmnpqrstvwy'))
        seq=seq.replace('O',random.choice('ACDEFGHIKLMNPQRSTVWY'))
        seq=seq.replace('o',random.choice('acdefghiklmnpqrstvwy'))
        #seq=seq1.encode("utf-8")
        D={}
#        print (seq, len(seq))
        for c in range(len(seq)-k+1):
#            pdb.set_trace()
            kmer = seq[c:c+k]
            try:
                D[kmer]+=1
            except:
                D[kmer]=1
        for key,val in D.items():
            dic2[key]=val
#        pdb.set_trace()
        fv=np.asarray(dic2.values()).reshape(1,-1)
        dfv[id1]=fv[0]
#        pdb.set_trace()
    return dfv
def Seq_Name_list(file_name):
        names=[]
        sequences=[]
        All_seq = list(SeqIO.parse(file_name,'fasta'))
        for i in range(0,len(All_seq)): 
            sequences.append(str((All_seq[i].seq)))
            names.append(All_seq[i].id.split('#')[0])
        return sequences,names
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
        #svr_lin = RandomForestClassifier(n_estimators=e, max_depth=d)
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
    plt.plot(fpr, tpr, color='c',marker=',',label='AUC_RF= {:.2f}'.format(auc_roc))
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
def chunkify(l, n=5):
    """
    Given a list of list of elements l, this function will create n folds with
    almost equal number of elements in each.
    """
    result = [[] for i in range(n)]
    sums   = [0]*n
    i = 0
    for e in l:
        result[i].extend(e)
        sums[i] += len(e)
        i = sums.index(min(sums))
    return result

def name2feature(bag,Dict):
    """
    bag is list of names that you require from given Names and crossponding features
    """
    fs=[]
    for b in bag:
#        import pdb;pdb.set_trace()
        if b in Dict:
            fs.append(Dict[b])
    #print(len(fs))
    return fs
def Make_Cluster(Names,path,file):
    import pdb
    names,Cluster=[],[]
    namescount,Clustercount=0,0
#    pdb.set_trace()
    with open(path+file) as f:
      content = f.readlines()
#    print("length of input file",len(content))
    for i in range(0,len(content),1):
        if len(content[i].split('Cluster'))>1:
            if len(names)>0:
                Clustercount=Clustercount+1
                namescount=namescount+len(names)
                Cluster.append(names)
                names=[]
        elif  len(content[i].split( '*' ))>1:
            name=content[i].split( '*' )[0].split('>')[1].split('_')[0]
#            names.append(name)
            if name  in Names:
#                pdb.set_trace()
                names.append(name)
        elif len(content[i].split('at'))>1:
            name=content[i].split('at')[0].split('>')[1].split('_')[0]
#            names.append(name)
            if name in Names:
                names.append(name)
#    print("Total sequence",namescount)
#    print("Total Clusters",Clustercount)
    return Cluster
def new_RemoveDuplicates(path,file):
    import pdb
    #path=''
    N_terminous=pickle.load(open(path+'nTerminus_All_Dict.npy', "rb"))
    C_terminous=pickle.load(open(path+'cTerminus_All_Dict.npy', "rb"))
    names,Cluster=[],[]
    count=0
    namescount=0
    NC_dict={}
    percent_list=[]
    with open(path+file) as f:
      content = f.readlines()
    print("Len of input data",len(content))
    for i in range(0,len(content),1):
      if len(content[i].split('Cluster'))>1:
#          cluster_name=content[i].split('Cluster')[1]
          if len(names)>0:
#              Cluster.append(names)
              namescount=namescount+len(names)
#              Cluster=np.append(Cluster,names)
              name_dict=dict(zip(names,percent_list))
#              pdb.set_trace()
              for s in name_dict:
                  if name_dict[s]==100.00:
#                      pdb.set_trace()
                      NC=N_terminous[int(s)]+'_'+C_terminous[int(s)]
                      NC_dict[NC]=s
#                      print(N_terminous[s],C_terminous[s],N_terminous[name_r] ,C_terminous[name_r])
#                      if N_terminous[s]!=N_terminous[name_r] or  C_terminous[s]!=C_terminous[name_r]:
#                        Cluster=np.append(Cluster,s)
                  elif name_dict[s]<100.00:
                         Cluster=np.append(Cluster,s)
#              [np.append(Cluster,s) for s in name_dict if name_dict[s]==100.00 and ( N_terminous[s]!=N_terminous[name_r] or  C_terminous[s]!=C_terminous[name_r] )]
              Cluster=np.append(Cluster,name_r)
#              pdb.set_trace()
              Cluster=np.append(Cluster,list(NC_dict.values()))
              names=[]
              NC_dict={}
#              pdb.set_trace()
#              names=[]
              percent_list=[]
          elif len(names)==1:
               Cluster=np.append(Cluster,name_r)
      elif  len(content[i].split( '*' ))>1:
#        pdb.set_trace()
        name=(content[i].split( '*' )[0].split('>')[1].split('_')[0])
        name_r=(content[i].split( '*' )[0].split('>')[1].split('_')[0])
        names.append(name_r)
      elif len(content[i].split('at'))>1:
        name=content[i].split('at')[0].split('>')[1].split('_')[0]
#        pdb.set_trace()
        percent=float(content[i].split('at')[-1].split('%')[0])
        percent_list.append(percent)
        names.append(name)
        
#        #print("name",name,"percent",percent)
##        C_terminous=content[i].split('_')[2]
##        N_terminous=content[i].split('_')[3]
#        if percent==100.00:
#            pdb.set_trace()
##            print("N_terminous[name]",N_terminous[name],"N_terminous[name_r]",N_terminous[name_r],"C_terminous[name]",C_terminous[name],"C_terminous[name_r]",C_terminous[name_r],np.any(N_terminous[name]!=N_terminous[name_r]), np.any(C_terminous[name]!=C_terminous[name_r]))
##            if np.any(N_terminous[name]!=N_terminous[name_r])or  np.any(C_terminous[name]!=C_terminous[name_r]):
#                names.append(name)
#            else:
#    #            print("Duplicate name=",name,N_terminous[name],C_terminous[name],"name_r=",name_r,N_terminous[name_r],C_terminous[name_r])
#                count=count+1
#        elif percent<100.00:
#            names.append(name)
    print("count=",count)
    print("Len of output data",namescount)
    return Cluster
def RedendencyRemoval(fold,path,UNames,hemo_file,nonhemo_file,Hemo_Dict,Non_hemo_Dict):
    #################90 new %
    hemo_CL=Make_Cluster(UNames,path,hemo_file)
    Non_hemo_CL=Make_Cluster(UNames,path,nonhemo_file)
    hemo_Folds=chunkify(hemo_CL)
    Non_hemo_Folds=chunkify(Non_hemo_CL)
    ####
    for i in range(5):
           X_train= np.array([], dtype=np.int64).reshape(0,len(list(Hemo_Dict.values())[0]))
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
"""
Features=np.hstack((ELMO_features,NC_features))
Features=np.hstack((Features,AAC_Features))
Hemo_Dict=dict(zip(Names_hemo,np.hstack((np.hstack((list(ELMO_Hemo_Dict.values()),
list(NC_hemo_dict.values()))),AAC_hemo))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((np.hstack((list(ELMO_Non_hemo_Dict.values()),
list(NC_non_hemo_dict.values()))),AAC_non_hemo))))
"""
#only 1mer+ELMO
Features=np.hstack((ELMO_features,AAC_Features))#onli 1mer+ELMO
Hemo_Dict=dict(zip(Names_hemo,np.hstack((list(ELMO_Hemo_Dict.values()),AAC_hemo))))
Non_hemo_Dict=dict(zip(Names_non_hemo,np.hstack((list(ELMO_Non_hemo_Dict.values()),AAC_non_hemo))))
###############

Label=np.append(np.ones(len(Hemo_Dict)),-1*np.ones(len(Non_hemo_Dict)))
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
##Roc= 0.7858955228880469 e= 130 d= 10 Average Mean= 0.730472339097517
#ResultMeanStd(130,10,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)#70%ELMO
#Roc= 0.7799307620671074 e= 160 d= 140 Average Mean= 0.7806737165905656
#ResultMeanStd(200,10,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
#ResultMeanStd(200,10,path,UNames,percent,Hemo_Dict,Non_hemo_Dict)
#1/0
print("Without 1 norm")
#Roc= 0.8830212585195861 e= 100 d= 5 Average Mean= 0.8980306401051673
cv = StratifiedKFold(n_splits=5, shuffle=True)
#C=[1,4,8,16,32,64,100,128,256,512,1024,2048,4096,8192,16384,32768,65535]
#Gamma=[0.00001,0.0001,0.001,0.01,0.1,1,2,4,8,16,32,64,128,256,512,1024,2056]
Depth=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
#Depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#XGBoost
Estimator=[1,5,10,15,20,25,30,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
###
"""
10 runs results
L=[0.866,0.866,0.867,0.871,0.867,0.872,0.87,0.867,0.868,0.87]
L=np.array(L)
print(np.mean(L),np.std(L))
"""
###Best parameters
#Depth=[6]#XGBoost
#Estimator=[120]
pre_roc,cc,gg,ee,dd=0,0,0,0,0

"""
XGboost
"""
#for d in Depth:
#   for e in Estimator:
#        Y_score,Y_t=[],[]
#        for i in range(5):
#            X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
#            svr_lin = XGBClassifier(learning_rate=0.1,max_depth=d, n_estimators=e)
##            svr_lin = RandomForestClassifier(n_estimators=e, max_depth=d,random_state=0)
#            svr_lin.fit(X_train, y_train)
##            Y_p=svr_lin.predict(X_test)
#            test_score=svr_lin.predict_proba(X_test)[:,1]
#            Y_score.extend(test_score)
#            Y_t.extend(y_test)
##            Y_pred.extend(Y_p) 
#            Roc_VA.append((test_score,list(y_test)))
#        avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
#        auc_roc=roc_auc_score(np.array(Y_t), np.array(Y_score))
#        print("auc_roc",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
#        if pre_roc<auc_roc:
#           print ("Prev_Roc=",pre_roc,"Roc=",auc_roc,"e=",e,"d=",d,"Average Mean=",avgmean)
#           pre_roc=auc_roc
#           ee=e
#           dd=d
##################
##1mer+ELMO+1024Smilebase
#ee=160
#dd=140
####ELMO+1mer
#Roc= 0.7852178831798914 e= 165 d= 10 Average Mean= 0.7315340775520628
#ee=165
#dd=10
ee= 200 
dd= 10
Roc_VA,Y_p,Y_test=[],[],[]
for i in range(5):
    X_train,X_test, y_train, y_test=RedendencyRemoval(i,path,UNames,'new_hemo_'+percent+'.txt','new_Nonhemo_'+percent+'.txt',Hemo_Dict,Non_hemo_Dict)
    svr_lin = XGBClassifier(max_depth=dd, n_estimators=ee)
    svr_lin.fit(X_train, y_train)
    test_score=svr_lin.predict_proba(X_test)[:,1]
    Y_p.extend(test_score)
    Y_test.extend(y_test)
    Roc_VA.append((test_score,list(y_test)))
auc_roc=roc_auc_score(np.array(Y_test), np.array(Y_p))
print("auc_roc",auc_roc)
avgfpr,avgtpr,avgmean=roc_VA( Roc_VA)
print("avgmean",avgmean)
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
######
####External testing
#####
###External testing
path='D:/PhD/Hemo_All_SeQ/'
External=np.load(path+"Hemolytic_External_validation_Features.npy")
#mer=1
print("External AAC features",mer,"mer")
External_AAC=list( All_FeaturesWithoutNC(path,path+'External.txt',mer).values())
External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
ELmo_External_features=External[:,0:1024]
#####
External_NC=np.zeros((15,2048))
External_features=np.hstack((np.hstack((ELmo_External_features,External_NC)),External_AAC))
####
#External_features=np.vstack((External_f[:2],External_f[4:6]))
#External_features=np.vstack((External_features,External_f[8:12]))
#External_features=np.vstack((External_features,External_f[15]))
#External_features=np.vstack((External_features,External_f[17]))
#External_features=np.vstack((External_features,External_f[18:20]))
#External_features=np.vstack((External_features,External_f[21:]))
####1mer+ELMO
#External_Features=np.hstack((ELmo_External_features,External_AAC))
External_score =svr_lin.predict_proba(External_features)[:,1]
print("External_score",External_score)
External_label=np.append(np.zeros(8),np.ones(7))
#External_label=np.append(np.zeros(12),np.ones(12))
External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
print("External_auc_roc",External_auc_roc_90r)
External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation 5-fold:{: .2f}'.format(External_auc_roc_90r))
plt.legend(loc='lower right')
plt.grid()
####Clinical using ELMO
ELMO_DRAMP_Clinical_data_Features=np.load(path+'DRAMP_Clinical_data_Features.npy')
DRAMP_Clinical_data_AAC=list( All_FeaturesWithoutNC(path,path+'DRAMP_Clinical_data.txt',mer).values())
DRAMP_Clinical_NC=np.zeros((28,2048))
ALL_DRAMP_Clinical_data_Features=np.hstack((np.hstack((ELMO_DRAMP_Clinical_data_Features,DRAMP_Clinical_NC)),DRAMP_Clinical_data_AAC))
ALL_DRAMP_Clinical_data_score=svr_lin.predict_proba(ALL_DRAMP_Clinical_data_Features)[:,1]
plt.figure()
plt.hist(np.sort(ALL_DRAMP_Clinical_data_score),bins=len(ALL_DRAMP_Clinical_data_score))
plt.grid()
print("Average score ALL features",np.mean(ALL_DRAMP_Clinical_data_score))
train_score = svr_lin.predict_proba(X_train)[:,1]
V=train_score
L=y_train
A,B = plattFit(V,L)
#A,B = plattFit(X_train,y_train) #rescling-coefficients
print('A =',A,'B =',B)
pp = sigmoid(V,A,B)
from sklearn.metrics import roc_auc_score
print("Print Ranges:")

print("Original:",np.min(V),np.max(V))
print("Rescaled:",np.min(pp),np.max(pp))
print("Calculate AUC-ROC (should not change):")
print(roc_auc_score(L,pp))
print(roc_auc_score(L,V))
V=ALL_DRAMP_Clinical_data_score
#L=np.zeros(28)
pp = sigmoid(V,A,B)
1/0
#path='D:/PhD/Hemo_All_SeQ/'
#External_f=np.load(path+"1076_Hemolytic_External_validation_Features.npy")
#mer=2
#print("External AAC features",mer,"mer")
#External_AAC=list( All_FeaturesWithoutNC(path,path+'Hemolytic_External_validation.txt',mer).values())
#External_AAC=(External_AAC-np.mean(External_AAC, axis = 0))/(np.std(External_AAC, axis = 0)+0.000001)
##Features=torch.FloatTensor(Features)
##External_features=External_f
#ELmo_External_features=External_f[:,0:1024]
#External_NC=np.zeros((24,2048))
#External_features=np.hstack((ELmo_External_features,External_NC))
#External_features=np.hstack((External_features,External_AAC))
#############
#External_features=np.vstack((External_f[:2],External_f[4:6]))
#External_features=np.vstack((External_features,External_f[8:12]))
#External_features=np.vstack((External_features,External_f[15]))
#External_features=np.vstack((External_features,External_f[17]))
#External_features=np.vstack((External_features,External_f[18:20]))
#External_features=np.vstack((External_features,External_f[21:]))
#External_label=np.append(np.zeros(8),np.ones(7))
##External_NC=np.zeros((24,2048))
##External_label=np.append(np.zeros(12),np.ones(12))
#####
##External_NC=np.zeros((24,2048))
##External_Features=np.hstack((ELmo_External_features,External_NC))
##External_Features=np.hstack((External_Features,External_AAC))
#
#####1mer+ELMO
##External_Features=np.hstack((ELmo_External_features,External_AAC))
#
##External_features=torch.FloatTensor(External_features).cuda()
##External_score =Hemonet(External_features).cpu().data
#External_score =svr_lin.predict_proba(External_features)[:,1]
#print("External_score",External_score)
##External_label=np.append(np.zeros(8),np.ones(7))
##External_label=np.append(np.zeros(12),np.ones(12))
#External_auc_roc_90r=roc_auc_score(np.array(External_label),np.array(External_score))
#print("External_auc_roc",External_auc_roc_90r)
#External_fpr_90r, External_tpr_90r, thresholds = roc_curve(np.array(External_label),np.array(External_score))
#plt.plot(External_fpr_90r, External_tpr_90r, color='m',marker=',',label='External Validation at 90%:{: .2f}'.format(External_auc_roc_90r))
##plt.grid()
#plt.legend(loc='lower right')
#plt.grid()