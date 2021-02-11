# HemoNet
HemoNet: Predicting Hemolytic Activity with Integrated Feature Learning
A neural network that perdict hemolytic activity classification of a peptide sequence along with inforamtion of N/C terminal modifications.This network captures context of amino acid with SeqVec embedding, Circular Fingerprint repersatation of N/C terminal modification's SMILES.
##Set Up Environment
```
python=3.6
conda install -c conda-forge rdkit
```
**Dataset**
Dataset can be downloaded from this link [https://github.com/adibayaseen/HemoNet/blob/main/Datasets.docx]
Seprate Hemolytic examples can be downloaded from here https://github.com/adibayaseen/HemoNet/blob/main/hemo_All_seq.txt
and non-hemolytic examples can be downloaded from here https://github.com/adibayaseen/HemoNet/blob/main/Nonhemo_All_seq.txt
**Code Structure**
weights.hdf file have link of google drive in which neural network weights are stored
UMAP.py file is used for visulaization of the data
Clusterify.py used for making non-redendend clusters
Final_Baseline_blaster_NN_5fold_cv_from_output_files.py used for Baseline Blast search


