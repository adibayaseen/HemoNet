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
"""
Conversion from all availble units to µM
input:sequence, availble units
Output:units to µM
"""
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
                cn=(i.split(G1)[0])
        elif len(i.split(G2))>1:
            cn=(i.split(G2)[0])
        elif len(i.split(G3))>1:
            cn=(i.split(G3)[0])
        elif len(i.split(G4))>1:
            cn=(i.split(G4)[0])
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
      cn=(float(cn)*(10**3))
    elif len(i.split(G11))>1:
      cn=(i.split(G11)[0])
      cn=Eliminate_symbols(cn)
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
"""
Conversion prefixes to a value, eliminating all non-digital signs
input:hemolytic concentration with symbols
Output:hemolytic concentration with numerical value
"""
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
"""
Conversion from one hot encoding(OHE) of DBAASP's N/C terminals to its names
input:OHE data
Output:OHE Names
"""
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
      cNames.append(C_terminous[int(name)])
      nNames.append(N_terminous[int(name)])
      IDS.append(name)
      SeQ.append(seq)
    return cNames,nNames,IDS, SeQ
"""
Verification at every steps
"""
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
    for d in dic:
        dic[d]= onehot_encoded_dict[dic[d]]
    return  dic,onehot_encoded_dict
path="D:\PhD\Hemo_All_SeQ/"
url=r'http://crdd.osdd.net/raghava/hemolytik/submitkey_adv.php?ran=7254'
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
######Total C-modification total 9  
NonHemo,Hemo,HemoId,NonHemoId,HemoSeq,NonHemoSeq,percent,equal,greater,lesser,concentration,percentage=[],[],[],[],[],[],[],[],[],[],[],[]
seq,ids,Mhc_Seq,Mhc_id,Mhc_Concentration,F=[],[],[],[],[],[]
count=0
NonHemo_dict,Hemo_dict={},{}
Seq_dict={}
for i in range(len(Activity)):
    S=str(Activity[i])
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
"""
Verification step start
"""
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
"""
Verification step End
"""
#print("%len",len(percent),"len of equal",len(equal),"greater",len(greater),"lesser",len(lesser),"count",count) 
####
C,Cn,SEQUENCE,IDS,PERCENT=[],[],[],[],[]
cp=0
cnc=0
else_count=0
for name in Seq_dict:
    unknown=[]
    [unknown.append(s) for s in Seq_dict[name][0] if s in U]
    if len(unknown)==0:
       print("concentartion",Seq_dict[name][2],Seq_dict[name][0])
       cp=cp+1
#       if len(i[i.find(G1):])>1 or len(i[i.find(G2):])>1 or len(i[i.find(G3):])>1 or len(i[i.find(G4):])>1 or len(i[i.find(G5):])>1 or len(i[i.find(G6):])>1 or len(i[i.find(G7):])>1 or len(i[i.find(G8):])>1:
       c=Micro_gram2MicroMole(Seq_dict[name][0],str(Seq_dict[name][2]))
       if c!='NaN':
           C.append(c)
           SEQUENCE.append( Seq_dict[name][0])
           IDS.append(name)
           p=Eliminate_symbols(str(Seq_dict[name][1]))
           PERCENT.append(p)
       else:
             print("else",Seq_dict[name][2])
             else_count=else_count+1
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
          HemoId.append(Mhc_id[m])
          HemoSeq.append(Mhc_Seq[m])
        elif MHC>100 or MHC==100:
          NonHemoId.append(Mhc_id[m])
          NonHemoSeq.append(Mhc_Seq[m])
print(count)
print("len of hemo After MHC",len(HemoId))
print("len of Nonhemo after MHC",len(NonHemoId))
#NC_Modification_dictionary
nTerminus_All={}
cTerminus_All={}
C_ter=dict(zip(IDs,Cmod))
N_ter=dict(zip(IDs,Nmod))
Act_dict=dict(zip(IDs,Act))
  ###########Only for hemolytk
hemo_seq =open(path+'hemo_All_seq.txt','w', encoding="utf-8")
Nonhemo_seq =open(path+'Nonhemo_All_seq.txt', "w+", encoding="utf-8")
for h in range(len(HemoId)):
  hemo_seq.write((">"+str(HemoId[h])+"_Hemolytik_"+ C_ter[HemoId[h]]+'_'+N_ter[HemoId[h]]+"_"+str(Act_dict[HemoId[h]])+"\n"+HemoSeq[h]+"\n"))
  print("id, C,N",HemoId[h],C_ter[HemoId[h]],N_ter[HemoId[h]])
  nTerminus_All[int(HemoId[h])]=N_ter[HemoId[h]]
  cTerminus_All[int(HemoId[h])]=C_ter[HemoId[h]]
for n in range(len(NonHemoId)):
#  Nonhemo_seq.write((">"+str(NonHemoId[n])+"#"+str(len(NonHemoSeq[n])) +"\n"+NonHemoSeq[n]+"\n"))
  Nonhemo_seq.write((">"+str(NonHemoId[n])+"_Hemolytik_"+ C_ter[NonHemoId[n]]+'_'+N_ter[NonHemoId[n]]+"_"+Act_dict[NonHemoId[n]]+"\n"+NonHemoSeq[n]+"\n"))
  print("id, C,N",NonHemoId[n],C_ter[NonHemoId[n]],N_ter[NonHemoId[n]])
  cTerminus_All[int(NonHemoId[n])]=C_ter[NonHemoId[n]]
  nTerminus_All[int(NonHemoId[n])]=N_ter[NonHemoId[n]]
####DBAASP###
import json
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
############## Write Names of presnet modifications
chemo,nhemo,idhemo,seqhemo,cNonhemo,nNonhemo,idNonhemo,seqNonhemo=[],[],[],[],[],[],[],[]
chemo,nhemo,idhemo,seqhemo=NC_names(path,'./DBAASP_Hemo.txt')
cNonhemo,nNonhemo,idNonhemo,seqNonhemo=NC_names(path,path+'./DBAASP_NonHemo.txt')
for h in range(len(idhemo)):
#  hemo_seq.write((">"+str(idhemo[h])+"_DBASSP_"+CT_Dict[int(idhemo[h])]+'_'+NT_Dict[int(idhemo[h])]+"_hem"+"\n"+seqhemo[h]+"\n"))
#  print("id, C,N",idhemo[h],chemo[h],nhemo[h])
  nTerminus_All[int(idhemo[h])]=NT_Dict[int(idhemo[h])]
  cTerminus_All[int(idhemo[h])]=CT_Dict[int(idhemo[h])]
for n in range(len(idNonhemo)):
#  Nonhemo_seq.write((">"+str(idNonhemo[n])+"_DBASSP_"+CT_Dict[int(idNonhemo[n])]+'_'+NT_Dict[int(idNonhemo[n])]+"_Nonhemo"+"\n"+seqNonhemo[n]+"\n"))
#  print("id, C,N",idNonhemo[n],cNonhemo[n],nNonhemo[n])
  nTerminus_All[int(idNonhemo[n])]=NT_Dict[int(idNonhemo[n])]
  cTerminus_All[int(idNonhemo[n])]=CT_Dict[int(idNonhemo[n])]
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
        cTerminus_All[t]='Free'
        Free_c=Free_c+1
    elif cTerminus_All[t]=='AMD' or cTerminus_All[t]=='Amidation':
        cTerminus_All[t]='AMD'
    elif cTerminus_All[t]=='Ome' or cTerminus_All[t]=='Methoxy (OMe)': 
        cTerminus_All[t]='Ome'
    elif cTerminus_All[t]=='EN'or cTerminus_All[t]=='[NH(CH2)2NH2]2' or cTerminus_All[t]=='NH(CH2)2NH2' :
        cTerminus_All[t]='EN'
#####N-Terminous
NT_count=0
Free_n=0
for n in nTerminus_All:
    if nTerminus_All[n]=='Free' or nTerminus_All[n]=='#':
        Free_n=Free_n+1
        nTerminus_All[n]='Free'
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
        nTerminus_All[n]= 'Ch'
print("total Free in C",Free_c,"total free in N",Free_n)
#######
for h in range(len(idhemo)):
  hemo_seq.write((">"+str(idhemo[h])+"_DBASSP_"+cTerminus_All[int(idhemo[h])]+'_'+nTerminus_All[int(idhemo[h])]+"_hem"+"\n"+seqhemo[h]+"\n"))
  print("id, C,N",idhemo[h],chemo[h],nhemo[h])
#  nTerminus_All[int(idhemo[h])]=NT_Dict[int(idhemo[h])]
#  cTerminus_All[int(idhemo[h])]=CT_Dict[int(idhemo[h])]
for n in range(len(idNonhemo)):
  Nonhemo_seq.write((">"+str(idNonhemo[n])+"_DBASSP_"+cTerminus_All[int(idNonhemo[n])]+'_'+nTerminus_All[int(idNonhemo[n])]+"_Nonhemo"+"\n"+seqNonhemo[n]+"\n"))
  print("id, C,N",idNonhemo[n],cNonhemo[n],nNonhemo[n])
#  nTerminus_All[int(idNonhemo[n])]=NT_Dict[int(idNonhemo[n])]
#  cTerminus_All[int(idNonhemo[n])]=CT_Dict[int(idNonhemo[n])]
cTerminusfile=open(path+'cTerminus_All_Dict.npy',"wb")
pickle.dump(cTerminus_All, cTerminusfile)
cTerminusfile.close()
nTerminusfile=open(path+'nTerminus_All_Dict.npy',"wb")
pickle.dump(nTerminus_All, nTerminusfile)
nTerminusfile.close() 
nTerminus_onehotencode_dict,nTerminus_unique_dict =onehotencoding(nTerminus_All)
cTerminus_onehotencode_dict,cTerminus_unique_dict=onehotencoding(cTerminus_All)

#######
name2Onehot_cTerminusfile=open(path+'cTerminus_name2Onehot_Dict.npy',"wb")
pickle.dump(cTerminus_unique_dict,name2Onehot_cTerminusfile)
name2Onehot_cTerminusfile.close()
name2Onehot_nTerminusfile=open(path+'nTerminus_name2Onehot_Dict.npy',"wb")
pickle.dump(nTerminus_unique_dict,name2Onehot_nTerminusfile)
name2Onehot_nTerminusfile.close() 
############
Onehot_cTerminusfile=open(path+'Onehot_cTerminus_All_Dict.npy',"wb")
pickle.dump(cTerminus_onehotencode_dict,Onehot_cTerminusfile)
Onehot_cTerminusfile.close()
Onehot_nTerminusfile=open(path+'Onehot_nTerminus_All_Dict.npy',"wb")
pickle.dump(nTerminus_onehotencode_dict,Onehot_nTerminusfile)
Onehot_nTerminusfile.close() 
###Common between two databases
DBAASP_id=idhemo+idNonhemo
Hemolytic_ids=HemoId+NonHemoId
Hemolytic_seq=HemoSeq+NonHemoSeq

dbaasp_seq=seqhemo+seqNonhemo
dbaasp_dict=dict(zip(DBAASP_id,dbaasp_seq))
hemolytic_dict=dict(zip(Hemolytic_ids,Hemolytic_seq))
Common=[]
my_count=0
for d_id in DBAASP_id:
    if d_id in Hemolytic_ids:# and dbaasp_dict[d_id]==ID_dict[d_id][0]:
#        my_count=my_count+1
#        if dbaasp_dict[d_id]==hemolytic_dict[d_id]:
        Common.append((d_id,hemolytic_dict[d_id],dbaasp_dict[d_id]))
    else:
        my_count=my_count+1

Common=[]
my_count=0
for hemo in idhemo:
    if hemo in HemoId:# and dbaasp_dict[d_id]==ID_dict[d_id][0]:
    #        my_count=my_count+1
    #        if dbaasp_dict[d_id]==hemolytic_dict[d_id]:
        Common.append((hemo ,hemolytic_dict[hemo],dbaasp_dict[hemo]))
else:
    my_count=my_count+1
Common=[]
my_count=0
for nonhemo in idNonhemo:
    if nonhemo in NonHemoId:# and dbaasp_dict[d_id]==ID_dict[d_id][0]:
#        my_count=my_count+1
#        if dbaasp_dict[d_id]==hemolytic_dict[d_id]:
        Common.append((nonhemo ,hemolytic_dict[nonhemo],dbaasp_dict[nonhemo]))
else:
    my_count=my_count+1
Hemo_dict=dict(zip(idhemo+HemoId,seqhemo+HemoSeq))
print(len(Hemo_dict))
Nonhemo_dict=dict(zip(idNonhemo+NonHemoId,seqNonhemo+NonHemoSeq))
print(len(Nonhemo_dict))#,Nonhemo_dict)
#####
for index in Hemo_dict:
    [unknown.append(s) for s in Hemo_dict[index] if s not in 'ACDEFGHIKLMNPQRSTVW']#acdefghiklmnpqrstvwy'  ]#'ACDEFGHIKLMNPQRSTVW'
U=set(unknown)
hemo_D=0
nonhemo_D=0
for index in Hemo_dict:
  unknown=[]
  [unknown.append(s) for s in Hemo_dict[index] if s in 'acdefghiklmnpqrstvwy']#U]
  if len(unknown)!=0:
      hemo_D=hemo_D+1
print(hemo_D, "out of", len(Hemo_dict),"Have D amino acid") 
for index in Nonhemo_dict:
  unknown=[]
  [unknown.append(s) for s in Nonhemo_dict[index] if s in 'acdefghiklmnpqrstvwy']
  if len(unknown)!=0:
      nonhemo_D=nonhemo_D+1
print(nonhemo_D, "out of", len(Nonhemo_dict),"Have D amino acid") 