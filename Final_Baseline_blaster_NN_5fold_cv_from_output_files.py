# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 02:49:26 2019
blast output file to scores
@author: 92340
"""
import numpy as np
from Bio import SeqIO
from Bio import SearchIO
from blaster import *
import pdb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef
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
path="D:\PhD\Hemo_All_SeQ/"
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
AUC_ROC=roc_auc_score(np.array(Y_t),np.array(Y_Predict))
tpr_b,fpr_b,t_b=roc_curve(np.array(Y_t),np.array(Y_Predict))
print("MCC:",MCC,"AUC_ROC",AUC_ROC)