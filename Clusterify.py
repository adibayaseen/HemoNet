# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:45:36 2019

@author: Adiba Yaseen
"""
import pickle
import numpy as np
#def Make_Cluster(path,file):
#    names,Cluster=[],[]
#    with open(path+file) as f:
#        content = f.readlines()
#    for i in range(1,len(content),1):
#        name=content[i-1].split(' ')[1].split("\n")[0]
#        cluster_prv=content[i-1].split(' ')[0].split(">")[1]
#        cluster_new=content[i].split(' ')[0].split(">")[1]
#        names.append(name)
#        if cluster_prv!=cluster_new:
#           Cluster.append(names)
#           names=[]
#    return Cluster

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
def RemoveDuplicates(path,file):
    #path=''
    N_terminous=pickle.load(open(path+'OHE_nTerminus_All_Dict.npy', "rb"))
    C_terminous=pickle.load(open(path+'OHE_cTerminus_All_Dict.npy', "rb"))
    names,Cluster=[],[]
    count=0
    namescount=0
    NC_dict={}
    with open(path+file) as f:
      content = f.readlines()
    print("Len of input data",len(content))
    for i in range(0,len(content),1):
      if len(content[i].split('Cluster'))>1:
#          cluster_name=content[i].split('Cluster')[1]
          if len(names)>0:
#              Cluster.append(names)
              namescount=namescount+len(names)
              Cluster=np.append(Cluster,names)
              pdb.set_trace()
              Cluster=np.append(Cluster,list(NC_dict.values()))
              names=[]
              NC_dict={}
      elif  len(content[i].split( '*' ))>1:
        name=(content[i].split( '*' )[0].split('>')[1].split('#')[0])
        name_r=int((content[i].split( '*' )[0].split('>')[1].split('#')[0]))
        names.append(name)
      elif len(content[i].split('at'))>1:
        name=int(content[i].split('at')[0].split('>')[1].split('#')[0])
        percent=float(content[i].split('at')[1].split('%')[0])
        #print("name",name,"percent",percent)
        if percent==100.00:
            NC=N_terminous[name]+'_'+C_terminous[name]
            NC_dict[NC]=name
            
#            print("N_terminous[name]",N_terminous[name],"N_terminous[name_r]",N_terminous[name_r],"C_terminous[name]",C_terminous[name],"C_terminous[name_r]",C_terminous[name_r],np.any(N_terminous[name]!=N_terminous[name_r]), np.any(C_terminous[name]!=C_terminous[name_r]))
#            if np.any(N_terminous[name]!=N_terminous[name_r])or  np.any(C_terminous[name]!=C_terminous[name_r]):
##                names.append(name)
#            else:
#    #            print("Duplicate name=",name,N_terminous[name],C_terminous[name],"name_r=",name_r,N_terminous[name_r],C_terminous[name_r])
#                count=count+1
        elif percent<100.00:
            names.append(name)
    print("count=",count)
    print("Len of output data",namescount)
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
