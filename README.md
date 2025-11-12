Multi-level Functional Network-based PD Identification via Graph Deep Learning
This repository contains the implementation for the paper "Multi-level functional network-based PD identification via graph deep learning". The proposed framework integrates functional connectivity networks with patient similarity networks using Graph Convolutional Networks (GCN) for Parkinson's Disease (PD) identification.

Project Structure
Core Files
merged_time_series.pkl: Preprocessed fMRI time series data extracted after preprocessing

all_correlation_matrices103.npy: Functional connectivity matrices computed from time series data

fused_similarity_matrix_k20.npy: Subject-subject similarity matrix with K=20 neighbors

Code Files
1. Data Processing & Visualization
one_FC_copy.py: Computes functional connectivity matrices from time series data

Input: merged_time_series.pkl (preprocessed fMRI time series)

Output: all_correlation_matrices103.npy (functional connectivity matrices)

Method: Pearson correlation coefficients between brain regions

Features: 116×116 symmetric connectivity matrices for 103 subjects

AAL_brain_six_lobes_AAL_copy.py: Visualizes brain connectivity networks for six major lobes

Input: all_correlation_matrices103.npy

Output: 2×3 subplot visualization of connectivity in Frontal, Occipital, Parietal, Subcortical, Temporal, and Cerebellum lobes

Features: Orthogonal views (sagittal, axial, coronal) for comprehensive visualization

2. Similarity Network Construction
SNF_Similarity_Matrix_k=20.py: Constructs subject-subject similarity network

Input: all_correlation_matrices103.npy

Output: fused_similarity_matrix_k20.npy

Method: Similarity Network Fusion (SNF) with Gaussian similarity

Parameters: K=20 nearest neighbors, sigma=1.0

Features: Builds graph where nodes represent subjects and edges represent FC pattern similarities

3. Main GCN Models
GCN_renew_copy.py: Base Graph Convolutional Network model

Architecture: 3-layer GCN (128-64-2 dimensions)

Features: Layer normalization, ELU activation, dropout regularization

Input: Functional connectivity features + demographic data (age, gender)

Training: Focal loss, class-weighted optimization, advanced feature engineering

GCN_LR_renew_copy.py: Enhanced GCN with Laplacian Regularization

Extends base GCN with Laplacian regularization term

Regularization: Adds graph smoothness constraint to prevent overfitting

Optimal λ: 0.1 (determined through hyperparameter tuning)

Features: Combined classification loss + Laplacian regularization

4. Interpretability Analysis
OS_copy.py: Occlusion Sensitivity analysis for model interpretability

Method: Systematically blocks different brain lobes and measures accuracy changes

Output: Box plots showing importance of each brain lobe in classification

Features: Identifies frontal lobe as most critical for PD identification

5. Comparative Models
GAT_copy_2.py: Graph Attention Network for comparison

Features: Attention mechanisms for weighted neighbor aggregation

Graph_Transformer_copy.py: Graph Transformer model for comparison

Features: Self-attention mechanisms across the graph

SVM_copy.py: Support Vector Machine baseline

Features: Traditional machine learning approach with flattened FC matrices

MLP_copy.py: Multi-Layer Perceptron baseline

Features: Deep neural network without graph structure

Key Features
Multi-level Network Framework
Individual Functional Connectivity: Brain region interactions within subjects

Subject-Similarity Network: FC pattern similarities between subjects

Integrated Analysis: Combines both levels for robust PD identification

Model Advantages
Handles small fMRI datasets effectively

Incorporates demographic covariates (age, gender)

Provides interpretability through occlusion sensitivity analysis

Achieves 76.2% accuracy, 72.2% precision, and 75.3% recall

Visualization & Interpretability
Comprehensive brain lobe connectivity visualization

Quantitative assessment of regional importance

Identification of frontal lobe as key biomarker for PD

Data Requirements
Input Data
Preprocessed fMRI time series (BOLD signals)

Clinical information: age, gender, diagnosis labels

AAL atlas parcellation (116 brain regions)

Data Format
Time series: num_subjects × num_timepoints × num_regions

FC matrices: num_subjects × num_regions × num_regions

Labels: Binary classification (PD=0, Control=1)

Dependencies
text
Python 3.7+
PyTorch 1.9+
PyTorch Geometric
scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
SciPy
Nilearn
Usage Pipeline
Preprocessing: Ensure fMRI data is preprocessed and time series are extracted

FC Calculation: Run one_FC_copy.py to generate connectivity matrices

Visualization: Execute AAL_brain_six_lobes_AAL_copy.py for connectivity visualization

Similarity Network: Run SNF_Similarity_Matrix_k=20.py for subject similarities

Model Training: Train with either GCN_renew_copy.py or GCN_LR_renew_copy.py

Interpretability: Run OS_copy.py for occlusion sensitivity analysis

Comparison: Execute comparative models (GAT_copy_2.py, SVM_copy.py, etc.)

Experimental Results
Best Performance: GCN with Laplacian Regularization (76.2% accuracy)

Key Finding: Frontal lobe plays crucial role in PD identification

Advantage: Outperforms traditional ML and other GNN architectures
