#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:49:56 2018
An implementation of a BLAST-based nearest neighbor search
@author: fminhas
"""

from Bio import SeqIO
from Bio import SearchIO
import os
import numpy as np
from scipy.stats import rankdata
import pdb
#C:\Program Files\NCBI\blast-2.10.0+\bin
#BLASTDir = "/mirror/pairpred_tools/ncbi-blast-2.2.30+-x64-linux/ncbi-blast-2.2.30+/bin/"
BLASTDir =r"C:\Program Files\NCBI\blast-2.10.0+\bin/"
#BLASTDir = "C:\Program Files\NCBI\blast-2.10.0+\bin/"
#BLASTDir='https://blast.ncbi.nlm.nih.gov/'
from joblib import Parallel, delayed
def blasterNN(testfile,trainfile,idx = "out"):
    try: 
        print("training")   
        pdb.set_trace()
        cmd = BLASTDir+"makeblastdb -in "+trainfile+" -dbtype prot"
        os.system(cmd)
        pdb.set_trace()
        cmd = BLASTDir+"blastp -db "+trainfile+" -query "+testfile+" -evalue 100 -out "+str(idx)+".pblast.txt"
        pdb.set_trace()
        os.system(cmd)
#        
#        P = {}
#        for qresult in SearchIO.parse(idx+".pblast.txt","blast-text"):
#            pdb.set_trace()
#            if len(qresult):
#                P[qresult.id]=np.min([hsp.evalue for hit in qresult for hsp in hit])
#            else:
#                P[qresult.id]=100.0
#        return P
    except Exception:        
        print ("Error Processing",idx)
        return None
    
#if __name__=='__main__':
#    path="D:\PhD\Hemo_All_SeQ/"
#    path='D:\PhD\Blast'
##    explist = [(path+'hemo_All_seq.txt',path+'hemo_All_seq.txt',"1"),(path+'Nonhemo_All_seq.txt',path+'Nonhemo_All_seq.txt',"2")]
#    testfile=path+'NonHemo_train.fasta'
#    trainfile=path+'NonHemo_test.fasta'
#    R=blasterNN(testfile,trainfile)

