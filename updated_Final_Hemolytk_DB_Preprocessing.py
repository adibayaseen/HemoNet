#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:25:09 2019

@author: adiba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 04:33:01 2019

@author: adiba
"""
import pandas as pd
from Bio.SeqUtils import molecular_weight
import random
import numpy as np
from Bio import SeqIO
import pickle
import pdb
def Micro_gram2MicroMole(seq,i):
    G1='μg/ml'#430
    G2='μg/mL'#28
    G3='µg ml-1'#7
    G4='µg/mL'#6
    G5='µg/ml'#128
    G6='μM'#697#1060
    G7='μmol/L'
    G8='µM'
    G9='mg/ml'
    #    ###73
    G10='mM'
    #    ###51
    G11='M'
    #    ###4
    G12='g/l'
        ###41
    G13='μg'
    G14='g/ml'
    cn='NaN'
    if len(i[i.find(G1):])>1 or len(i[i.find(G2):])>1 or len(i[i.find(G3):])>1 or len(i[i.find(G4):])>1 or len(i[i.find(G5):])>1:
        if len(i.split(G1))>1:
    #    if len(i[i.find(G1):])>1 or len(i[i.find(G2):])>1 or len(i[i.find(G3):])>1 or len(i[i.find(G4):])>1 or len(i[i.find(G5):])>1:
                cn=(i.split(G1)[0])
    #            mw=molecular_weight(seq[index], "protein")
    #            C.append((float(cn)/mw)*(10**3))
        elif len(i.split(G2))>1:
            cn=(i.split(G2)[0])
    #            mw=molecular_weight(seq[index], "protein")
    #            C.append((float(cn)/mw)*(10**3))
        elif len(i.split(G3))>1:
            cn=(i.split(G3)[0])
    #            mw=molecular_weight(seq[index], "protein")
    #            C.append((float(cn)/mw)*(10**3))
        elif len(i.split(G4))>1:
            cn=(i.split(G4)[0])
    #            mw=molecular_weight(seq[index], "protein")
    #            C.append((float(cn)/mw)*(10**3))
        elif len(i.split(G5))>1:
            cn=(i.split(G5)[0])
        if cn!='NaN':
            cn=Eliminate_symbols(cn)
            mw=molecular_weight(seq, "protein")
            cn=(float(cn)/mw)*(10**3)
            print("mw",mw)
    
    elif len(i.split(G6))>1 or len(i.split(G7))>1 or len(i.split(G8))>1:
        if len(i.split(G6))>1:
            cn=i.split(G6)[0]
        elif len(i.split(G7))>1:
            cn=i.split(G7)[0]
        elif len(i.split(G8))>1:
            cn=i.split(G8)[0]
        if cn!='NaN' and len(cn.split('.'))<3:
            cn=Eliminate_symbols(cn)
        elif cn!='NaN':
            cn=Eliminate_symbols(cn)
    elif len(i.split(G9))>1:
      print(G9)
      cn=(i.split(G9)[0])
      cn=Eliminate_symbols(cn)
      mw=molecular_weight(seq, "protein")
      cn=(float(cn)/mw)*(10**6)
      print("mw",mw)
    elif len(i.split(G10))>1 or  len(i.split(G12))>1:
      if len(i.split(G10))>1:
        cn=(i.split(G10)[0])
      elif len(i.split(G12))>1:
        cn=(i.split(G12)[0])
      cn=Eliminate_symbols(cn)
      #mw=molecular_weight(seq[index], "protein")
      cn=(float(cn)*(10**3))
    elif len(i.split(G11))>1:
      cn=(i.split(G11)[0])
      cn=Eliminate_symbols(cn)
      #mw=molecular_weight(seq[index], "protein")
      cn=(float(cn)*(10**6))
    elif len(i.split(G13))>1:
      cn=(i.split(G13)[0])
      cn=Eliminate_symbols(cn)
      mw=molecular_weight(seq, "protein")
      cn=(float(cn)/mw)
    elif len(i.split(G14))>1:
      cn=(i.split(G14)[0])
      cn=Eliminate_symbols(cn)
      mw=molecular_weight(seq, "protein")
      cn=(float(cn)/mw)*(10**9)
    return cn
def Eliminate_symbols(cn):
    if '±' in cn:
      print(cn)
      cn=float(cn.split('±')[0]) 
    elif 'x' in cn and '>' in cn:
      cn=str(cn)
      cn=cn.split('>')[1]
      p=cn.split('x')[1].split('10')[1]
      print("p=",p)
      if '-' in p:
        cn=float(cn.split('x')[0])/(10**float(p.split('-')[1]))
    elif 'x' in cn:
      cn=str(cn)
      p=cn.split('x')[1].split('10')[1]
      print("p=",p)
      if '-' in p:
        cn=float(cn.split('x')[0])/(10**float(p.split('-')[1]))
    elif '>' in cn and '-' in cn:
        cn=cn.split('>')[1]
        cn=0.5*(float(cn.split('-')[0])+float(cn.split('-')[1]))
    elif '~' in cn and '-' in cn:
        cn=cn.split('~')[1]
        cn=0.5*(float(cn.split('-')[0])+float(cn.split('-')[1]))
    elif '>' in cn:
        cn=cn.split('>')[1]
    elif 'above' in cn:
        cn=cn.split('above')[1]
    elif '<' in cn:
        cn=cn.split('<')[1]
    elif '-' in cn:
        cn=0.5*(float(cn.split('-')[0])+float(cn.split('-')[1]))
    elif '–' in cn:
        cn=0.5*(float(cn.split('–' )[0])+float(cn.split('–' )[1]))
    elif 'to' in cn:
        cn=0.5*(float(cn.split('to' )[0])+float(cn.split('to' )[1]))
    elif '~' in cn:
        cn=cn.split('~')[1]
    elif '≥' in cn:
        cn=cn.split('≥')[1]
    else:
        cn=float(cn)
    return float(cn)
def NC_names(path,file_name):
    records = list(SeqIO.parse(file_name, "fasta"))
    cNames,nNames,IDS,SeQ=[],[],[],[]
    for i in range(0,len(records)):
      N_terminous=pickle.load(open(path+"/DBAASP_dicNterminus.npy", "rb"))
      C_terminous=pickle.load(open(path+"DBAASP_dicCterminus.npy", "rb"))
      """
      N_terminous=pickle.load(open('./DBAASP_nTerminus.txt', "rb"))
      C_terminous=pickle.load(open('./DBAASP_cTerminus.txt', "rb"))
      """
      name=records[i].id.split('#')[0]
      seq=str(records[i].seq)
#      import pdb;pdb.set_trace()
      cNames.append(C_terminous[int(name)])
      nNames.append(N_terminous[int(name)])
      IDS.append(name)
      SeQ.append(seq)
    return cNames,nNames,IDS, SeQ
def Verify(final_dict,intial_dict):
    correct_count=0
    incorrect=[]
    for i in final_dict:
        if i in intial_dict:
            if intial_dict[i]==final_dict[i]:
                correct_count=correct_count+1
            else:
                incorrect.append((i,ID_dict[i],final_dict))
    return correct_count,incorrect
def onehotencoding(dic):
    from numpy import array
    from numpy import argmax
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    # define example
    data = list(set(list(dic.values())))
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    onehot_encoded_dict=dict(zip(data,onehot_encoded))
    new_dict={}
    for d in dic:
        new_dict[d]= onehot_encoded_dict[dic[d]]
    return  new_dict,data
path="D:\PhD\Hemo_All_SeQ/"
#url=r'http://crdd.osdd.net/raghava/hemolytik/submitkey_adv.php?ran=1938'
url=r'http://crdd.osdd.net/raghava/hemolytik/submitkey_adv.php?ran=5803'
#path='/home/adiba/Desktop/PhD/4rth/Hemolytic/Hymolytic_DB_data/
tables = pd.read_html(url) 
Mydata=tables[2]
ID=Mydata[['Result']].values[2:-2]
SeQ=Mydata[['Unnamed: 3_level_0']].values[2:-2]
C_ter_MOD=Mydata[['Unnamed: 5_level_0']].values[2:-2]
N_ter_MOD=Mydata[['Unnamed: 6_level_0']].values[2:-2]
Seq,IDs,Cmod,Nmod,Act=[],[],[],[],[]
[Seq.append(s[0]) for s  in SeQ]
[IDs.append(s[0]) for s  in ID]
[Cmod.append(c[0]) for c in C_ter_MOD]
[Nmod.append(n[0]) for n in N_ter_MOD]
Activity=Mydata[['Unnamed: 12_level_0']].values[2:-2]   
[Act.append(a[0]) for a in Activity]
unknown,All_unknown,M,CT,NT=[],[],[],[],[]
###
for index in range(len(Seq)):
    Seq[index]=Seq[index].replace('-','')
    [unknown.append(s) for s in Seq[index] if s  not in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy'  ]
U=set(unknown)
ID_dict=dict(zip(IDs,zip(Seq,Cmod,Nmod,Act)))

######Total C-modification total 10
###Write Dictionary for hemolytk database N and C terminous   
NonHemo,Hemo,HemoId,NonHemoId,HemoSeq,NonHemoSeq,percent,equal,greater,lesser,concentration,percentage=[],[],[],[],[],[],[],[],[],[],[],[]
seq,ids,Mhc_Seq,Mhc_id,Mhc_Concentration,F=[],[],[],[],[],[]
count=0
NonHemo_dict,Hemo_dict={},{}
Seq_dict={}
for i in range(len(Activity)):
    S=str(Activity[i])
    #    if len(S.split("'")[1].split('(non-hemolytic)'))>1 or len(S.split("'")[1].split('(non hemolytic)')) or len(S.split("'")[1].split('(Non hemolytic)'))>1 or len(S.split("'")[1].split('Non-hemolytic'))>1 or len(S.split("'")[1].split('No hemolysis'))>1 or len(S.split("'")[1].split('little hemolysis'))>1 or len(S.split("'")[1].split('Low hemolytic activity'))>1 or len(S.split("'")[1].split('Poor hemolytic'))>1 or len(S.split("'")[1].split('Little hemolysis'))>1 or len(S.split("'")[1].split('Negligible hemolysis'))>1 or len(S.split("'")[1].split('No hemolytic'))>1 or len(S.split("'")[1].split('Weak hemolysis'))>1  or len(S.split("'")[1].split('Low hemolysis'))>1 :
    if  len(S.split("'")[1].split('Weak hemolytic'))>1 or  len(S.split("'")[1].split('(non-hemolytic)'))>1 or len(S.split("'")[1].split('Non-hemolytic'))>1 or len(S.split("'")[1].split('(non hemolytic)'))>1 or len(S.split("'")[1].split('Non hemolytic'))>1 or len(S.split("'")[1].split('No hemolysis'))>1 or len(S.split("'")[1].split('little hemolysis'))>1 or len(S.split("'")[1].split('Low hemolytic activity'))>1 or len(S.split("'")[1].split('Poor hemolytic'))>1 or len(S.split("'")[1].split('Little hemolysis'))>1 or len(S.split("'")[1].split('Negligible hemolysis'))>1 or len(S.split("'")[1].split('No hemolytic'))>1 or len(S.split("'")[1].split('Weak hemolysis'))>1  or len(S.split("'")[1].split('Low hemolysis'))>1 :
        NonHemo.append(S.split("'")[1])
        NonHemoId.append(ID[i][0])
        NonHemoSeq.append(Seq[i])
        NonHemo_dict[ID[i][0]]=Seq[i]
    elif len(S.split("'")[1].split('%'))>1:
            percent.append(S.split("'")[1].split('%'))
            if len(S.split("'")[1].split('%')[1].split("at"))>1:
                concentration.append(S.split("'")[1].split('%')[1].split("at")[1].strip())
                Seq_dict[ID[i][0]]=(Seq[i],S.split("'")[1].split('%')[0],S.split("'")[1].split('%')[1].split("at")[1].strip())
                seq.append(Seq[i])
                ids.append(ID[i][0])
                percentage.append(S.split("'")[1].split('%')[0])
            elif len(S.split("'")[1].split('%')[1].split("upto"))>1:
                concentration.append(S.split("'")[1].split('%')[1].split("upto")[1].strip())
                seq.append(Seq[i])
                percentage.append(S.split("'")[1].split('%')[0])
                ids.append(ID[i][0])
                Seq_dict[ID[i][0]]=(Seq[i],S.split("'")[1].split('%')[0],S.split("'")[1].split('%')[1].split("upto")[1].strip())
            elif len(S.split("'")[1].split('%')[1].split("up to"))>1:
                concentration.append(S.split("'")[1].split('%')[1].split("up to")[1].strip())
                seq.append(Seq[i])
                percentage.append(S.split("'")[1].split('%')[0])
                ids.append(ID[i][0])
                Seq_dict[ID[i][0]]=(Seq[i],S.split("'")[1].split('%')[0],S.split("'")[1].split('%')[1].split("up to")[1].strip())
            elif len(S.split("'")[1].split('%')[1].split(" hemolysis"))>1:
                concentration.append(S.split("'")[1].split('%')[1].split(" hemolysis")[1].strip())
                seq.append(Seq[i])
                percentage.append(S.split("'")[1].split('%')[0])
                ids.append(ID[i][0])
                Seq_dict[ID[i][0]]=(Seq[i],S.split("'")[1].split('%')[0],S.split("'")[1].split('%')[1].split(" hemolysis")[1].strip())
            
            else:
                print(S.split("'")[1].split('%')[1])
    elif len(S.split("'")[1].split("fractions hemolysis at "))>1:
        F.append(S.split("'")[1].split("fractions hemolysis at "))
        concentration.append(S.split("'")[1].split("fractions hemolysis at ")[1].strip())
        seq.append(Seq[i])
        ids.append(ID[i][0])
        percentage.append(float(S.split("'")[1].split("fractions hemolysis at ")[0])*100)
        Seq_dict[ID[i][0]]=(Seq[i],float(S.split("'")[1].split("fractions hemolysis at ")[0])*100,S.split("'")[1].split("fractions hemolysis at ")[1].strip())
    elif len(S.split("'")[1].split('='))>1  :
            equal.append(S.split("'")[1].split('='))
            Mhc_Concentration.append(S.split("'")[1].split('='))
            Mhc_Seq.append(Seq[i])
            Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('~'))>1:
            equal.append(S.split("'")[1].split('~'))
            Mhc_Concentration.append(S.split("'")[1].split('~'))
            Mhc_Seq.append(Seq[i])
            Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('>'))>1 :
         greater.append(S.split("'")[1].split('>'))
         Mhc_Concentration.append(S.split("'")[1].split('>'))
         Mhc_Seq.append(Seq[i])
         Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('>>'))>1:
        Mhc_Concentration.append(S.split("'")[1].split('>>'))
        greater.append(S.split("'")[1].split('>>'))
        Mhc_Seq.append(Seq[i])
        Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('≥'))>1:
            greater.append(S.split("'")[1].split('≥'))
            Mhc_Concentration.append(S.split("'")[1].split('≥'))
            Mhc_Seq.append(Seq[i])
            Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('<'))>1:
            lesser.append(S.split("'")[1].split('<'))
            Mhc_Concentration.append(S.split("'")[1].split('<'))
            Mhc_Seq.append(Seq[i])
            Mhc_id.append(ID[i][0])
    elif len(S.split("'")[1].split('Strong hemolysis'))>1 or len(S.split("'")[1].split('Very strong hemolysis'))>1 or len(S.split("'")[1].split('Strongly hemolytic'))>1 or len(S.split("'")[1].split("Hemolytic"))>1  or len(S.split("'")[1].split('hemolytic'))>1:
         Hemo.append(S.split("'")[1])
         HemoId.append(ID[i][0])
         HemoSeq.append(Seq[i])
         Hemo_dict[ID[i][0]]=Seq[i]
    else:
        count=count+1
        print(S.split("'")[1])

print(len(NonHemo))
print(len(Hemo))
####
wrong=[]
correct_count=0
for i in Seq_dict:
    if i in ID_dict:
            s1,c1,n1,a1=ID_dict[i]
            s2,p,c=Seq_dict[i]
            if s1==s2:
                correct_count=correct_count+1
                wrong.append((i,a1,p,c))
#                wrong.append((i,s1,s2,a1,a2))
            else:
                print("id=",i,ID_dict[i],Seq_dict[i])
#                wrong.append((i,ID_dict[i],nonhemo_final_dict[i]))
                wrong.append((i,s1,s2,a1,p,c))
#1/0
print("%len",len(percent),"len of equal",len(equal),"greater",len(greater),"lesser",len(lesser),"count",count) 
#C,Cn,SEQUENCE,IDS,PERCENT=[],[],[],[],[]
#cp=0
#cnc=0
#for index in range(len( Seq_dict)):
##    i=concentration[index]
#    unknown=[]
#    [unknown.append(s) for s in seq[index] if s in U]
#    if len(unknown)==0:
#       print("concentartion",concentration[index])
#       cp=cp+1
##       if len(i[i.find(G1):])>1 or len(i[i.find(G2):])>1 or len(i[i.find(G3):])>1 or len(i[i.find(G4):])>1 or len(i[i.find(G5):])>1 or len(i[i.find(G6):])>1 or len(i[i.find(G7):])>1 or len(i[i.find(G8):])>1:
#       c=Micro_gram2MicroMole(seq[index],concentration[index])
#       if c!='NaN':
##               print(i)
#           C.append(c)
#           SEQUENCE.append(Seq[index])
#           IDS.append(ID[index][0])
#           p=Eliminate_symbols(str(percentage[index]))
#           PERCENT.append(p)
##           print("after conversion",c,"percentage",p)
#       else:
#             print(concentration[index])
####
C,Cn,SEQUENCE,IDS,PERCENT=[],[],[],[],[]
cp=0
cnc=0
for name in Seq_dict:
    unknown=[]
    [unknown.append(s) for s in Seq_dict[name][0] if s in U]
    if len(unknown)==0:
       print("concentartion",Seq_dict[name][2])
       cp=cp+1
#       if len(i[i.find(G1):])>1 or len(i[i.find(G2):])>1 or len(i[i.find(G3):])>1 or len(i[i.find(G4):])>1 or len(i[i.find(G5):])>1 or len(i[i.find(G6):])>1 or len(i[i.find(G7):])>1 or len(i[i.find(G8):])>1:
       c=Micro_gram2MicroMole(Seq_dict[name][0],str(Seq_dict[name][2]))
       if c!='NaN':
#               print(i)
           C.append(c)
           SEQUENCE.append( Seq_dict[name][0])
           IDS.append(name)
           p=Eliminate_symbols(str(Seq_dict[name][1]))
           PERCENT.append(p)
#           print("after conversion",c,"percentage",p)
       else:
             print(Seq_dict[name][2])
#1/0
#######
print(cp)
Seq_dict={}
Seq_dict=dict(zip(IDS,zip(SEQUENCE,PERCENT, C)))
wrong=[]
correct_count=0
for i in Seq_dict:
    if i in ID_dict:
            s1,c1,n1,a1=ID_dict[i]
            s2,p2,c=Seq_dict[i]
            if s1==s2:
                correct_count=correct_count+1
                wrong.append((i,a1,p2,c))
            else:
                print("id=",i,ID_dict[i],Seq_dict[i])
#                wrong.append((i,ID_dict[i],nonhemo_final_dict[i]))
                wrong.append((i,s1,s2,a1,p2,c))
for index in range(len(SEQUENCE)):
#    print("PERCENT[index]",PERCENT[index],"C[index]",C[index])
    if PERCENT[index]<2 or PERCENT[index]==2 and C[index]>10 or C[index]==10 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<5 or PERCENT[index]==5 and C[index]>20 or C[index]==20 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<10 or PERCENT[index]==10 and C[index]>50 or C[index]==50 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<15 or PERCENT[index]==15 and C[index]>100 or C[index]==100 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<20 or PERCENT[index]==20 and C[index]>200 or C[index]==200 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<30 or PERCENT[index]==30 and C[index]>300 or C[index]==300 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]<50 or PERCENT[index]==50 and C[index]>500 or C[index]==500 :
        NonHemoId.append(IDS[index])
        NonHemoSeq.append(SEQUENCE[index]) 
    #####################Hemo#################33
            #####check this condition
    elif PERCENT[index]>5 or PERCENT[index]==5 and C[index]<10 or C[index]==10 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]>10 or PERCENT[index]==10 and C[index]<20 or C[index]==20 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]>15 or PERCENT[index]==15 and C[index]<50 or C[index]==50 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]>20 or PERCENT[index]==20 and concentration[index]<100 or concentration[index]==100 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]>30 or PERCENT[index]==30 and C[index]<200 or C[index]==200 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    elif PERCENT[index]>50 or PERCENT[index]==50 and C[index]<300 or C[index]==300 :
        HemoId.append(IDS[index])
        HemoSeq.append(SEQUENCE[index]) 
    print("len of hemo",len(HemoId))
    print("len of Nonhemo",len(NonHemoId))
count=0
for m in  range(len(Mhc_id)):
  unknown=[]
  [unknown.append(s) for s in seq[index] if s in U]
  if len(unknown)==0:
    i=Mhc_Concentration[m]
    print(i)
    if i[0]=='MHC' or i[0]=='MHC ':
      count=count+1
      if i[1]!='':
        MHC=Micro_gram2MicroMole(seq[index],str(i[1]))
        print(i[1],MHC )
        if MHC<50 or MHC==50:
#          print("hemo with mhc=",MHC)
          HemoId.append(Mhc_id[m])
          HemoSeq.append(Mhc_Seq[m])
        elif MHC>100 or MHC==100:
#          print("NonHemo with mhc=",MHC)
          NonHemoId.append(Mhc_id[m])
          NonHemoSeq.append(Mhc_Seq[m])
print(count)
print("len of hemo After MHC",len(HemoId))
print("len of Nonhemo after MHC",len(NonHemoId))
#C_ter=Cmod
#N_ter=Nmod
#NC_Modification_dictionary
nTerminus_All={}
cTerminus_All={}
C_ter=dict(zip(IDs,Cmod))
N_ter=dict(zip(IDs,Nmod))
Act_dict=dict(zip(IDs,Act))
  ###########Only for hemolytk
"""
hemo_seq =open(path+'hemo_seq.txt','w')
Nonhemo_seq =open(path+'Nonhemo_seq.txt','w')
CT_mod=open(path+'CT_mod_hemolytk',"w")
NT_mod=open(path+'NT_mod_hemolytk',"w")
"""
hemo_seq =open(path+'hemo_All_seq.txt','w', encoding="utf-8")
Nonhemo_seq =open(path+'Nonhemo_All_seq.txt', "w+", encoding="utf-8")
#CT_mod=open(path+'CT_mod_All',"w")
#NT_mod=open(path+'NT_mod_All',"w")

for h in range(len(HemoId)):
#  hemo_seq.write((">"+str(HemoId[h])+"#"+str(len(HemoSeq[h])) +"\n"+HemoSeq[h]+"\n"))
  hemo_seq.write((">"+str(HemoId[h])+"_Hemolytik_"+ C_ter[HemoId[h]]+'_'+N_ter[HemoId[h]]+"_"+str(Act_dict[HemoId[h]])+"\n"+HemoSeq[h]+"\n"))
  print("id, C,N",HemoId[h],C_ter[HemoId[h]],N_ter[HemoId[h]])
#  if int(HemoId[h]) in nTerminus_All:
      #pdb.set_trace()
  nTerminus_All[int(HemoId[h])]=N_ter[HemoId[h]]
  cTerminus_All[int(HemoId[h])]=C_ter[HemoId[h]]
#  if  HemoId[h]=='1312':
#      pdb.set_trace()
for n in range(len(NonHemoId)):
#  Nonhemo_seq.write((">"+str(NonHemoId[n])+"#"+str(len(NonHemoSeq[n])) +"\n"+NonHemoSeq[n]+"\n"))
  Nonhemo_seq.write((">"+str(NonHemoId[n])+"_Hemolytik_"+ C_ter[NonHemoId[n]]+'_'+N_ter[NonHemoId[n]]+"_"+Act_dict[NonHemoId[n]]+"\n"+NonHemoSeq[n]+"\n"))
  print("id, C,N",NonHemoId[n],C_ter[NonHemoId[n]],N_ter[NonHemoId[n]])
  cTerminus_All[int(NonHemoId[n])]=C_ter[NonHemoId[n]]
  nTerminus_All[int(NonHemoId[n])]=N_ter[NonHemoId[n]]
####DBAASP###
import json
#import pickle
#path='/content/drive/My Drive/ELMO_Embedding/'
#DBAASP_Writing_names
with open(path+'alldata1.json') as jsonfile:
    data = json.load(jsonfile)
ncount=0
Ccount=0
CT_Dict,NT_Dict={},{}
for x in data:
  if len(data[x]['peptideCard'])>1:
    S=str(data[x]['peptideCard'])
    if len(S.split("nTerminus"))>1:
      ncount=ncount+1
      NT_Dict[int(x)]=S.split("nTerminus")[1].split(",")[0].split(":")[1].split("'")[1]
    if len(S.split("cTerminus"))>1:
      Ccount=Ccount+1
      CT_Dict[int(x)]=S.split("cTerminus")[1].split(",")[0].split(":")[1].split("'")[1]
#      1/0
############## Write Names of presnet modifications
chemo,nhemo,idhemo,seqhemo,cNonhemo,nNonhemo,idNonhemo,seqNonhemo=[],[],[],[],[],[],[],[]
chemo,nhemo,idhemo,seqhemo=NC_names(path,'./DBAASP_Hemo.txt')
cNonhemo,nNonhemo,idNonhemo,seqNonhemo=NC_names(path,path+'./DBAASP_NonHemo.txt')
for h in range(len(idhemo)):
#  if  str(idhemo[h])=='1312':
#     import pdb;pdb.set_trace()
  
  hemo_seq.write((">"+str(idhemo[h])+"_DBASSP_"+CT_Dict[int(idhemo[h])]+'_'+NT_Dict[int(idhemo[h])]+"_hem"+"\n"+seqhemo[h]+"\n"))
  print("id, C,N",idhemo[h],chemo[h],nhemo[h])
#  pdb.set_trace()
  nTerminus_All[int(idhemo[h])]=NT_Dict[int(idhemo[h])]
  cTerminus_All[int(idhemo[h])]=CT_Dict[int(idhemo[h])]
#  C_ter[HemoId[h]]
#  CT_mod.write( (str(idhemo[h])+"\t"+chemo[h]+"\n"))
#  CT.append(cTerminus[idhemo[h]])
#  NT_mod.write(str(idhemo[h])+"\t"+nhemo[h]+"\n")
#  NT.append(nTerminus[idhemo[h]])
for n in range(len(idNonhemo)):
#  if NT_Dict[int(idNonhemo[n])]=='C2':
##      1/0
#      pdb.set_trace()
  Nonhemo_seq.write((">"+str(idNonhemo[n])+"_DBASSP_"+CT_Dict[int(idNonhemo[n])]+'_'+NT_Dict[int(idNonhemo[n])]+"_Nonhemo"+"\n"+seqNonhemo[n]+"\n"))
  print("id, C,N",idNonhemo[n],cNonhemo[n],nNonhemo[n])
  nTerminus_All[int(idNonhemo[n])]=NT_Dict[int(idNonhemo[n])]
  cTerminus_All[int(idNonhemo[n])]=CT_Dict[int(idNonhemo[n])]
  
#  CT_mod.write((str(idNonhemo[n])+"\t"+cNonhemo[n]+"\n"))
#  NT_mod.write( (str(idNonhemo[n])+"\t"+nNonhemo[n]+"\n"))
#  CT.append(N_ter[idNonhemo[n]])
#  NT.append(N_ter[idNonhemo[n]])
#cTerminusfile=open(path+'cTerminus_All_Dict.npy',"wb")
#pickle.dump(cTerminus_All, cTerminusfile)
#cTerminusfile.close()
#nTerminusfile=open(path+'nTerminus_All_Dict.npy',"wb")
#pickle.dump(nTerminus_All, nTerminusfile)
#nTerminusfile.close() 
###processing on files
with open(path+'Nonhemo_All_seq.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    s= [data_list[d] for d in range(1,len(data_list),2)]
    N = len(data_list)
    D =[data_list[d].split('>')[1] for d in range(0,len(data_list),2)]
    DD=[str(d).split('_') for d in D]
    DD=np.array(DD)
    ids,db,c,n,a= [str(d).split('_')[0] for d in D],[str(d).split('_')[1] for d in D],[str(d).split('_')[2] for d in D],[str(d).split('_')[3] for d in D],[str(d).split('_')[4] for d in D]
nonhemo_final_dict=dict(zip(ids,zip(s,c,n,a)))
db_dict=dict(zip(ids,db))
####hemo finl dict
with open(path+'hemo_All_seq.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    s= [data_list[d] for d in range(1,len(data_list),2)]
    N = len(data_list)
    D =[data_list[d].split('>')[1] for d in range(0,len(data_list),2)]
    DD=[str(d).split('_') for d in D]
    DD=np.array(DD)
    ids,db,c,n,a= [str(d).split('_')[0] for d in D],[str(d).split('_')[1] for d in D],[str(d).split('_')[2] for d in D],[str(d).split('_')[3] for d in D],[str(d).split('_')[4] for d in D]
hemo_final_dict=dict(zip(ids,zip(s,c,n,a)))
#####
correct_count=0
wrong=[]
for i in nonhemo_final_dict:
    if i in ID_dict:
        if db_dict[i]=='Hemolytik':
            s1,c1,n1,a1=ID_dict[i]
            s2,c2,n2,a2=nonhemo_final_dict[i]
            if a1==a2:
                correct_count=correct_count+1
            else:
                print("id=",i,ID_dict[i],nonhemo_final_dict[i])
#                wrong.append((i,ID_dict[i],nonhemo_final_dict[i]))
                wrong.append((i,a1,a2))
####1-hot-Encoding
#CTerminousS
Free_c=0
for t in cTerminus_All:
    print(t,cTerminus_All[t])
    if cTerminus_All[t]=='Free' or cTerminus_All[t]=='#':
        print(cTerminus_All[t])
        cTerminus_All[t]='Free'
        Free_c=Free_c+1
            #AMD=Amidation'
    elif cTerminus_All[t]=='AMD' or cTerminus_All[t]=='Amidation':
        print(cTerminus_All[t])
        cTerminus_All[t]='AMD'
#        pdb.set_trace()
    elif cTerminus_All[t]=='Ome' or cTerminus_All[t]=='Methoxy (OMe)': 
        cTerminus_All[t]='Ome'
    elif cTerminus_All[t]=='EN'or cTerminus_All[t]=='[NH(CH2)2NH2]2' or cTerminus_All[t]=='NH(CH2)2NH2' :
        print(cTerminus_All[t])
        cTerminus_All[t]='EN'
#####N-Terminous
NT_count=0
Free_n=0
for n in nTerminus_All:
#    print(n,nTerminus_All[n])
    if nTerminus_All[n]=='Free' or nTerminus_All[n]=='#':
        Free_n=Free_n+1
        nTerminus_All[n]='Free'
#        print(cTerminus_All[n])
    elif nTerminus_All[n]=='Dansylation' or nTerminus_All[n]== 'DNS': #or cTerminus_All[t]=='#':
        print(nTerminus_All[n])
        nTerminus_All[n]= 'DNS'
    elif nTerminus_All[n]=='BZO' or nTerminus_All[n]=='Benzoylation':
        nTerminus_All[n]='BZO'
        print(nTerminus_All[n])
    elif nTerminus_All[n]=='ACT' or nTerminus_All[n]=='Acetylation (CH3(CH2)4CO)' or nTerminus_All[n]=='Acetylation (CH3(CH2)6CO)' or nTerminus_All[n]=='Acetylation'or nTerminus_All[n]=='Acetylated by n-octanoyl':   
        print(nTerminus_All[n])
        nTerminus_All[n]='ACT'
    elif nTerminus_All[n]=='Tos' or nTerminus_All[n]=='Tosylation' or nTerminus_All[n]=='TOS':
        print(nTerminus_All[n])
        nTerminus_All[n]='Tos'
    elif nTerminus_All[n]=='Formylation'or nTerminus_All[n]== 'FOR':# or cTerminus_All[t]=='#':
        print(nTerminus_All[n])
        nTerminus_All[n]= 'FOR'
    elif nTerminus_All[n]== 'Conjugated with lauric acid' or nTerminus_All[n]== 'C12':# or cTerminus_All[t]=='#':
        print(nTerminus_All[n])
        nTerminus_All[n]= 'C12'
    elif nTerminus_All[n]== 'Palmitoylation' or nTerminus_All[n]== 'C16':# or cTerminus_All[t]=='#':
        print(nTerminus_All[n])
        nTerminus_All[n]= 'C16'
    elif nTerminus_All[n]== 'ch' or nTerminus_All[n]== 'Ch':
#        pdb.set_trace()
        nTerminus_All[n]= 'Ch'
print("NT_count=",NT_count)
print("total Free in C",Free_c,"total free in N",Free_n)
cTerminusfile=open(path+'cTerminus_All_Dict.npy',"wb")
pickle.dump(cTerminus_All, cTerminusfile)
cTerminusfile.close()
nTerminusfile=open(path+'nTerminus_All_Dict.npy',"wb")
pickle.dump(nTerminus_All, nTerminusfile)
nTerminusfile.close() 
nTerminus_onehotencode_dict,nTerminus_unique =onehotencoding(nTerminus_All)
cTerminus_onehotencode_dict,cTerminus_unique=onehotencoding(cTerminus_All)
#######
Onehot_cTerminusfile=open(path+'Onehot_cTerminus_All_Dict.npy',"wb")
pickle.dump(Onehot_cTerminusfile, cTerminus_onehotencode_dict)
cTerminusfile.close()
Onehot_nTerminusfile=open(path+'Onehot_nTerminus_All_Dict.npy',"wb")
pickle.dump(Onehot_nTerminusfile, nTerminus_onehotencode_dict)
nTerminusfile.close() 
#1/0    
       
    
#1/0  
#find union of both list
#C_list = list(set(chemo + cNonhemo))
#N_list = list(set(nhemo + nNonhemo))
#print("Cterminous:",len(C_list),C_list, "\n Ntermionus: ",len(N_list),N_list)

"""
Remove duplicates if same sequence and NC modification
#import pickle
#path='/home/adiba/Desktop/PhD/4rth/Hemolytic/hemodatapapersandcode/'
N_terminous=pickle.load(open(path+"nTerminus_All_Dict.npy", "rb"))
C_terminous=pickle.load(open(path+"cTerminus_All_Dict.npy", "rb"))
#with open(path+'./cTerminus_All_Dict') as f:
#  C_terminous = f.readlines()
#with open(path+'./nTerminus_All_Dict') as f:
#  N_terminous = f.readlines()
#path='/content/drive/My Drive/ELMO_Embedding/'
#N_terminous=pickle.load(open('./CT_mod_hemolytk', 'r'))
#C_terminous=pickle.load(open('./NT_mod_hemolytk', 'r'))
#1/0
#file='CD_hit_hemolytkDB.fasta.clstr.sorted'
file='HemoltkAndDBAASP_all_seq.fasta.clstr.sorted'
names,Cluster,P=[],[],[]
count=0
with open(path+file) as f:
  content = f.readlines()
for i in range(0,len(content),1):
  if len(content[i].split('Cluster'))>1:
    cluster_name=content[i].split('Cluster')[1]
    if len(names)>0:
     print("count =",count)
     Cluster.append(names)
     print(names)
     names=[]
#     count=0
  elif len(content[i].split('at'))>1:
    name=int(content[i].split('at')[0].split('>')[1].split('#')[0])
    percent=float(content[i].split('at')[1].split('%')[0])
    print("name=",name,"percent=",percent,"cluster name=",cluster_name)
#    count=count+1
    print("NT_mod:",N_terminous[name],"CT_mod:",C_terminous[name])
    if percent==100.00:#and (N_terminous[name]==N_terminous[name_r]) and (C_terminous[name]==C_terminous[name_r]):
        print("Duplicate name=",name,N_terminous[name],C_terminous[name],"name_r=",name_r,N_terminous[name_r],C_terminous[name_r])
        count=count+1
        #import pdb;pdb.set_trace()
    elif percent==100.00 and (N_terminous[name]!=N_terminous[name_r] or  C_terminous[name]!=C_terminous[name_r]):
#          print("name=",name,N_terminous[name],C_terminous[name],"name_r=",name_r,N_terminous[name_r],C_terminous[name_r])
          names.append(name)
#          count=count+1
    elif percent<100.00:
        names.append(name)
#        count=count+1
  elif  len(content[i].split( '*' ))>1:
    print(content[i])
    name=(content[i].split( '*' )[0].split('>')[1].split('#')[0])
    name_r=int((content[i].split( '*' )[0].split('>')[1].split('#')[0]))
    names.append(name)
print("count=",count)
#NC_Modification_dictionary
#C_ter={}
#N_ter={}
#import pickle
#for c in range(len(C_ter_MOD)):
#  C_ter[ID[c][0]]=C_ter_MOD[c][0]
#  N_ter[ID[c][0]]=N_ter_MOD[c][0]
##for  n in range(len())
#CT_mod=open(path+'CT_mod.txt',"wb")
#pickle.dump(C_ter, CT_mod)
#CT_mod.close()
#NT_mod=open(path+'NT_mod.txt',"wb")
#pickle.dump(N_ter, NT_mod)
#NT_mod.close()
"""