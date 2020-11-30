# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:38:25 2019

@author: Adiba Yaseen
"""
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
def All_Features(path,file_name,k):
    records = list(SeqIO.parse(file_name, "fasta"))
#    pdb.set_trace()
    Features={}
#    Names=[]
    for i in range(0,len(records)):
#        import pdb;pdb.set_trace()
        F=[]
        Dict={}
        N_terminous=pickle.load(open(path+'Onehot_nTerminus_All_Dict.npy', "rb"))
        C_terminous=pickle.load(open(path+'Onehot_cTerminus_All_Dict.npy', "rb"))
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
        F=np.append(F,N_terminous[int(name)])
        F=np.append(F,C_terminous[int(name)])
        F=(F)/(np.sum(F, axis = 0))
        Features[name]=F
    return Features#,Names
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
#        if len(F)==40:
#            import pdb;pdb.set_trace()
#            F=(F)/(np.sum(F, axis = 0))
        Features[name]=F
#        else:
#            print("name",name,"seq",seq)
    return Features#,Names
