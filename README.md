# Chronic Kidney Disease Prediction using K-Nearest Neighbors

This repository contains a complete machine learning pipeline for **predicting Chronic Kidney Disease (CKD)** using the **K-Nearest Neighbors (KNN)** algorithm on a **merged clinical dataset**.

The project focuses on:
- Integrating two public CKD datasets
- Building a robust preprocessing pipeline (imputation + scaling + feature selection)
- Training and tuning a KNN classifier with cross-validation
- Achieving high diagnostic performance suitable for decision-support use

---

## 🔍 Project Overview

Chronic Kidney Disease is a major global health burden and is often detected late, when kidney damage is already severe. This project explores how a carefully designed machine learning pipeline can help **predict CKD early** using commonly available clinical parameters.

Key aspects:

- Two datasets merged:
  - **UCI CKD dataset** (400 samples, 25 attributes)
      - URL: {https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease}
  - **Risk factor CKD dataset (CKD-dataset-v2)** (200 samples, 29 attributes)
      - URL: {https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease}
- Final merged dataset: **600 records**, **28 features**, binary target: `affected` (1 = CKD, 0 = Non-CKD)
- Main model: **K-Nearest Neighbors (KNN)**
- Best configuration (from GridSearchCV, 5-fold CV):  
  `algorithm='auto'`, `leaf_size=20`, `metric='manhattan'`, `n_neighbors=3`, `weights='uniform'`
- Test performance:
  - **Accuracy:** 98.33%  
  - Zero false positives, only two false negatives
  - Confusion Matrix: 
```text 
    [[44  0]
     [ 2 74]]
 ``` 

---

## Steps to run project

- First download the datasets and keep them in 'data/raw/' folder.
- After installing all the libraries run the "merge_and_clean.ipynb" file, to merge both the datasets.
- Then open "knn_modelling.ipynb" file, install all the requirements, then run to implement the pipeline, at last the model pipeline will be saved in "models/" folder.

---

## 📁 Repository Structure

```text
PL PROJECT/
├─ data/
│  ├
│  └─ processed/
│     └─ ckd_merged_corrected.csv # Final merged & cleaned dataset used for modeling
│
├─ models/
│  ├─ ckd_knn_pipeline.joblib     # Saved KNN pipeline (production-ready)
│  └─ ckd_knn_pipeline1.joblib    # Alternate/experimental pipeline (optional)
│
├─ notebooks/
   ├─ merge_and_clean.ipynb       # Data merging, cleaning, preprocessing
   ├─ knn_modelling.ipynb         # Main KNN training & evaluation notebook
   └─ knn_modelling1.ipynb        # Experimental / variant modelling notebook

