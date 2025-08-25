# BraVL Multimodal Classification

Classifying object labels by fusing **brain signals (EEG)**, **visual features**, and **textual descriptions** using the [BraVL dataset](https://github.com/ChangdeDu/BraVL).

Note: This repository does not provide the dataset, including the seen/unseen data. Please visit the original repository for access.

Visit `BraVL_documentation.pdf` on this repository to view the methodology and output from the Colab notebook.  
See `documentation.py` for the basic logic and structure behind this project.

---

## Overview

This project explores multimodal learning using EEG signals, CLIPText embeddings, and image features to classify object labels.  
The approach combines traditional machine learning methods with preprocessing and evaluation techniques to handle complex data from three modalities.

Key components:

- Data exploration and visualization
- Outlier detection and removal (Isolation Forest)
- Standardization and PCA-based dimensionality reduction
- Stratified train/test split and class balancing (Random Over Sampling)
- Baseline and custom model implementation (Logistic Regression with gradient descent)

Skills demonstrated:

- Machine Learning (Logistic Regression, Isolation Forest, PCA)
- Multimodal Feature Engineering
- Data Imbalance Handling
- Custom Gradient Descent and Regularization
- EEG Signal Preprocessing

---

## Origin of Dataset

The [BraVL dataset](https://github.com/ChangdeDu/BraVL) (Brain Vision Language) is a **trimodal** dataset designed to study the relationship between human brain activity, visual perception, and language understanding.  
It contains:

- EEG brain responses recorded from human participants while viewing images
- Visual features extracted from those images
- Textual embeddings from descriptions of the images

The classification task is to predict the object label using the fused multimodal information.

---

## BraVL Dataset Structure

The BraVL dataset provides:

- EEG features: 17-channel recordings, sliced from 70–400 ms (indices 27–60)
- Image features: 100-dimensional PCA-reduced embeddings
- Text features: CLIPText embeddings
- Labels: Integer-encoded words representing the object in each image

Original BraVL dataset repository: [https://github.com/ChangdeDu/BraVL](https://github.com/ChangdeDu/BraVL)


---

## Model Pipeline

1. **Load data**  
   - EEG: slice relevant time window, flatten, scale by 2.0  
   - Image: load, scale by 50.0, keep first 100 dimensions  
   - Text: load, scale by 2.0  
   - Labels: loaded from `class_idx` in EEG `.mat` files

2. **Standardize per modality**  
   StandardScaler is applied independently to EEG, image, and text features.

3. **Outlier removal**  
   Isolation Forest is used for each modality. Only samples passing all filters are kept.

4. **Feature fusion and dimensionality reduction**  
   Concatenate `[EEG | Image | Text]` features and apply PCA (retain 95% variance)

5. **Split and balance**  
   Stratified train/test split (80/20). Random Over Sampling applied to balance training classes.

6. **Model training**  
   - Baseline: Multinomial Logistic Regression using scikit-learn  
   - Custom: Logistic Regression from scratch with softmax, L1/L2 regularization, mini-batch gradient descent, and weight initialization

---

## Results

| Model               | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
|--------------------|----------|----------------------|--------------------|----------------------|
| Baseline (scikit-learn) | 55.00%   | 0.53                 | 0.55               | 0.54                 |
| Custom Model        | 56.48%   | 0.62                 | 0.56               | 0.55                 |

---

## BraVL_Analysis.ipynb

This notebook contains the full analysis pipeline for the BraVL dataset:

- EEG brain signal processing
- Visual and text feature handling
- Data exploration and visualization
- Standardization and PCA
- Outlier removal (Isolation Forest)
- Dataset balancing (RandomOverSampler)
- Baseline and custom model implementation
- Evaluation and improvements (accuracy, F1, regularization, batch training)

Open the interactive notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HnUmp5_yvMdXwpvP01HpqK2TT0OJzOl5?usp=sharing)
