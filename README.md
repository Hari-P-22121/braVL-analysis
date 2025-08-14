
---

## 🛠 Pipeline
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

## 📈 Results
| Model        | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 (Weighted) |
|--------------|----------|----------------------|--------------------|---------------|
| Baseline     | 55.00%   | 0.53                 | 0.55               | 0.54          |
| Custom Model | 56.48%   | 0.62                 | 0.56               | 0.55          |

---

## 📦 Installation
```bash
git clone https://github.com/yourusername/neurovision-classifier.git
cd neurovision-classifier
pip install -r requirements.txt
