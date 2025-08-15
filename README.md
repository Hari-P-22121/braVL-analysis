Classifying object labels by fusing **brain signals (EEG)**, **visual features**, and **textual descriptions** using the [BraVL dataset](https://github.com/ChangdeDu/BraVL).

NOTE: This repository does not provided the dataset, including the seen/unseen data. Please visit the original repo for the data.

Visit "collab" on this repository to view the code and it's output on Google Collab.

Read documentation.py to see the basic analysis behind this project.

---

## Origin of dataset
The **[BraVL dataset](https://github.com/ChangdeDu/BraVL)** (Brain Vision Language) is a **trimodal** dataset designed to study the relationship between human brain activity, visual perception, and language understanding.  
It contains:
- **EEG brain responses** recorded from human participants while they viewed images.
- **Visual features** extracted from those images.
- **Textual embeddings** from descriptions of the images.

The task is to classify each sample into an **object label** by leveraging the combined information from all three modalities (Brain EEG, Textual, Image, and Label features).

---

## BraVL Dataset
The BraVL dataset provides:
- **EEG features**: 17-channel recordings, sliced from 70–400 ms (indices 27–60).
- **Image features**: 100-dimensional PCA-reduced embeddings.
- **Text features**: CLIPText embeddings.
- **Labels**: Integer-encoded words representing the object in each image.

Oringinal BraVL dataset repository: [BraVL on GitHub](https://github.com/ChangdeDu/BraVL).

Expected folder structure for this project:

data_root/
brain_feature/{roi}/{subject}/eeg_train_data_within.mat
visual_feature/ThingsTrain/{image_model}/{subject}/feat_pca_train.mat
textual_feature/ThingsTrain/text/{text_model}/{subject}/text_feat_train.mat


---

## Model Pipeline
1. **Load data**  
   - EEG: slice relevant time window, flatten, scale by 2.0.  
   - Image: load, scale by 50.0, keep first 100 dims.  
   - Text: load, scale by 2.0.  
   - Labels: loaded from `class_idx` in EEG `.mat` file.

2. **Standardize per modality**  
   StandardScaler on EEG, image, and text **separately**.

3. **Outlier removal**  
   Isolation Forest per modality, keep only samples that pass all three filters.

4. **Feature fusion & dimensionality reduction**  
   Concatenate [EEG | Image | Text] features → PCA (retain 95% variance).

5. **Split & balance**  
   Stratified train/validation split. Random Over Sampling on training set to fix imbalance.

6. **Model training**  
   - **Baseline**: scikit-learn multinomial Logistic Regression.  
   - **Custom**: softmax logistic regression with L1/L2 regularization, mini-batch gradient descent, better weight initialization.

---

## Result
| Model        | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 (Weighted) |
|--------------|----------|----------------------|--------------------|---------------|
| Baseline     | 55.00%   | 0.53                 | 0.55               | 0.54          |
| Custom Model | 56.48%   | 0.62                 | 0.56               | 0.55          |

---

## Installation
```bash
git clone https://github.com/yourusername/neurovision-classifier.git
cd neurovision-classifier
pip install -r requirements.txt
