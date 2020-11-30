#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 04:02:09 2019

@author: adiba
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
def Best_accuracy(y_ts, predictions):
    f, t, a=roc_curve(y_ts, predictions)
    AN=sum(x<0 for x in y_ts)
    AP=sum(x>0 for x in y_ts)
    TN=(1.0-f)*AN
    TP=t*AP
    Acc2=(TP+TN)/len(y_ts)
    acc=max(Acc2)
    print ('best accuracy=',acc )
    return acc
path='D:/PhD/Hemo_All_SeQ/'
"""
All Data by Hemo_PI
"""
###########
#df_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/Hemo_server_Prediction.csv')
#hemo_score=df_hemo[['PROB Score.1']]
#hemo_score=hemo_score.values
##Y_t=np.ones(len(Y_score))
#df_non_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/Non_Hemo_server_Prediction.csv')
#non_hemo_score=df_non_hemo[['PROB Score.1']]
#non_hemo_score=non_hemo_score.values
################3All data
df_hemo = pd.read_csv(path+'HemoPI_predicted_results_hemo_AllSeq_data.csv')
hemo_score_HemoPI=df_hemo[['Prediction']].values
df_non_hemo = pd.read_csv(path+'HemoPI_predicted_results_Non_hemo_AllSeq_data.csv')
non_hemo_score_HemoPI=df_non_hemo[['Prediction']].values

###################333
#"""
#Only Upper from hemoPI
#"""
#df_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/Hemo_Upper_server_Prediction.csv')
#hemo_score=df_hemo[['PROB Score']]
#hemo_score=hemo_score.values
#df_non_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/Non_Hemo_Upper_server_Prediction.csv')
#non_hemo_score=df_non_hemo[['PROB Score']]
#non_hemo_score=non_hemo_score.values
#############
"""
All Data by Hemo_pred
"""
#df_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/HemoPred_predicted_results_hemo_data.csv')
#hemo_score=df_hemo[["'Score'"]]
#hemo_score=hemo_score.values
#df_non_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/HemoPred_predicted_results_Non_hemo_data.csv')
#non_hemo_score=df_non_hemo[["'Score'"]]
#non_hemo_score=non_hemo_score.values
#################33
df_hemo = pd.read_csv(path+'new_HemoPred_predicted_results_hemo_AllSeq_data.csv')
hemo_score_HemoPred=df_hemo[['Prediction']].values
#hemo_score=hemo_score.values
df_non_hemo = pd.read_csv(path+'new_HemoPred_predicted_results_NON_hemo_AllSeq_data.csv')
non_hemo_score_HemoPred=df_non_hemo[['Prediction']].values
#non_hemo_score=non_hemo_score.values
########################3
#hemo_score=df_hemo[["'Score'"]]
#hemo_score=hemo_score.values
#df_non_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/HemoPred_predicted_results_Non_hemo_data.csv')
#non_hemo_score=df_non_hemo[["'Score'"]]
#non_hemo_score=non_hemo_score.values
################33
"""
Only Upper by Hemo_pred
"""
#df_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/HemoPred_predicted_results_hemo_Upper_data.csv')
#hemo_score=df_hemo[['+ACI-Score+ACI-']]
#hemo_score=hemo_score.values
#df_non_hemo = pd.read_csv('/home/adiba/Desktop/PhD/4rth/Hemolytic/HemoPred_predicted_results_Non_hemo_Upper_data.csv')
#non_hemo_score=df_non_hemo[['+ACI-Score+ACI-']]
#non_hemo_score=non_hemo_score.values
#1/0

##############33
HemoPI_score=np.append(hemo_score_HemoPI,non_hemo_score_HemoPI)
HemoPI_Label=np.append(np.ones(len(hemo_score_HemoPI)),np.zeros(len(non_hemo_score_HemoPI)))
HemoPI_auc_roc=roc_auc_score(np.array(HemoPI_Label), np.array(HemoPI_score))
HemoPI_accuracy=Best_accuracy(np.array(HemoPI_Label), np.array(HemoPI_score))
print(HemoPI_auc_roc)
fpr, tpr, thresholds = roc_curve(np.array(HemoPI_Label), np.array(HemoPI_score))
plt.plot(fpr, tpr, color='darkorange',marker='.' ,label='HemoPI_results= {:.2f}'.format(HemoPI_auc_roc))
################
HemoPred_score=np.append(hemo_score_HemoPred,non_hemo_score_HemoPred)
HemoPred_Label=np.append(np.ones(len(hemo_score_HemoPred)),np.zeros(len(non_hemo_score_HemoPred)))
HemoPred_auc_roc=roc_auc_score(np.array(HemoPred_Label), np.array(HemoPred_score))
print(HemoPred_auc_roc)
HemoPred_accuracy=accuracy_score(np.array(HemoPred_Label), np.array(HemoPred_score))
fpr, tpr, thresholds = roc_curve(np.array(HemoPred_Label), np.array(HemoPred_score))
plt.plot(fpr, tpr, color='b',marker='.' ,label='Hemopred_Accuracy= {:.2f}'.format(HemoPred_accuracy))
##################
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.legend(loc='lower right')
plt.grid()

#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend()
#fpr, tpr, thresholds = roc_curve(np.array(Y_t), np.array(Y_score))
#plt.plot(fpr, tpr, color='darkorange',marker='.',label='AUC= {:.2f}'.format(auc_roc))
#plt.grid()
#plt.figure()
#plt.plot(avgfpr, avgtpr, color='b',marker='.',label='AUC_avgmean= {:.2f}'.format(avgmean))
#plt.legend(loc='lower right')
#plt.grid()
#plt.show() 






