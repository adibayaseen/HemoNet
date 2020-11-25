# -*- coding: utf-8 -*-
"""
Created on wednesday 1 january 2020
Happy new year

@author: 92340
"""
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
sns.set(style='white', context='notebook', rc={'figure.figsize':(5,5)})
path="D:\PhD\Hemo_All_SeQ/"
####ELMO Features
records_hemo=np.load(path+'new_Hemo_Features.npy')
records_non_hemo=np.load(path+'new_NonHemo_Features.npy')
Names_hemo=np.load(path+'new_Hemo_Names.npy')
Names_hemo=[str(n).split('_')[0] for n in Names_hemo]
Names_non_hemo=np.load(path+'new_NonHemo_Names.npy')
Names_non_hemo=[str(n).split('_')[0] for n in Names_non_hemo]
"""
NC Smiles Features
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
Label=np.append(np.ones(len(Hemo_Dict)),np.zeros(len(Non_hemo_Dict)))
import umap
neighbours=6#5 #best
embedding=umap.UMAP(a=neighbours,learning_rate=0.3).fit_transform(Features, y=Label)
embedding.shape
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
colors=[]
for i in Label:
    if i==1.0:
        colors.append('g')
    elif i==0.:
        colors.append('r')
plt.figure()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
for j, c in zip( range(len(Label)), colors):
    plt.scatter(embedding[j][ 0], embedding[j][1], c=c,marker='.',s=3)
plt.scatter(embedding[0][ 0], embedding[0][1], c='g',marker='.',label='Hemolytic')
plt.scatter(embedding[-1][ 0], embedding[-1][1], c='r',marker='.', label='NonHemolytic')
plt.legend()
plt.grid()
plt.show()
plt.savefig('Ourmodel_UMAP.png', dpi=200)