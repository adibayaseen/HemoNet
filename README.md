# HemoNet

HemoNet: Predicting Hemolytic Activity with Integrated Feature Learning
A neural network that perdict hemolytic activity classification of a peptide sequence along with inforamtion of N/C terminal modifications.This network captures context of amino acid with SeqVec embedding, Circular Fingerprint repersatation of N/C terminal modification's SMILES.
# Abstract
Quantifying the hemolytic activity of peptides is a crucial step in the discovery of novel therapeutic peptides. Computational methods are attractive in this domain due to their ability to guide wet-lab experimental discovery or screening of peptides based on their hemolytic activity. However, existing methods are unable to accurately model various important aspects of this predictive problem such as the role of N/C terminal modifications, D- and L- amino acids, etc. In this work we have developed a novel neural network-based approach called HemoNet for predicting the hemolytic activity of peptides. The proposed method captures the contextual importance of different amino acids in a given peptide sequence using a specialized feature embedding in conjunction with SMILES-based fingerprint representation of N/C terminal modifications. We have analyzed the predictive performance of the proposed method using stratified cross-validation for comparison with previous methods, non-redundant cross-validation as well as validation on external peptides and clinical antimicrobial peptides. Our analysis shows the proposed approach achieves significantly better predictive performance (AUC-ROC of 88%) in comparison to previous approaches (HemoPI and HemoPred with AUC-ROC of 73%). HemoNet can be a useful and much-needed tool in the search for novel therapeutic peptides. The python implementation of the proposed method is available at the URL: https://github.com/adibayaseen/HemoNet
Keywords: Hemolytic activity prediction, peptide toxicity classification, Antimicrobial activity, Machine learning guided drug discovery
## Set Up Environment
```
python=3.6
conda install -c conda-forge rdkit
```
## Dataset
Dataset can be downloaded from this link [https://github.com/adibayaseen/HemoNet/blob/main/Datasets.docx]<br/>
Seprate Hemolytic examples can be downloaded from here https://github.com/adibayaseen/HemoNet/blob/main/hemo_All_seq.txt<br/>
Non-hemolytic examples can be downloaded from here https://github.com/adibayaseen/HemoNet/blob/main/Nonhemo_All_seq.txt <br/>
## Code Structure
[weights.hdf ](https://github.com/adibayaseen/HemoNet/blob/main/weights.hdf)file have link of google drive in which neural network weights  of SeqVec features are stored <br/>
[5-foldResultsComparison.py](https://github.com/adibayaseen/HemoNet/blob/code/5-foldResultsComparison.py) is used for 5-fold comparison with existing models and basline results.<br/>
[HemoNet10RunsResults](https://github.com/adibayaseen/HemoNet/blob/code/HemoNet10RunsResults)( 5-fold and Non-redundant) is used for HemoNet's all methods(5-fold and Non-redundant Cross-Validation Analysis) for 10 runs <br/>
[UMAP.py](https://github.com/adibayaseen/HemoNet/blob/code/UMAP.py) file is used for visulaization of the data <br/>
[Clusterify.py](https://github.com/adibayaseen/HemoNet/blob/code/Clusterify.py) used for making non-redendend clusters <br/>
[Final_Baseline_blaster_NN_5fold_cv_from_output_files.py](https://github.com/adibayaseen/HemoNet/blob/code/Final_Baseline_blaster_NN_5fold_cv_from_output_files.py) used for Baseline Blast search <br/>
