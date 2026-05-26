# NephroAI: Chronic Kidney Disease Prediction System

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80-Live%20Demo-purple?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/dmist36/NephroAI)

NephroAI is a state-of-the-art, publication-grade machine learning system designed to predict and diagnose **Chronic Kidney Disease (CKD)** using an optimized, leakage-free clinical K-Nearest Neighbors (KNN) pipeline.

This repository features standard Scikit-Learn data science engineering, a modular Python architecture, and a stunning, interactive local web application with a glassmorphism design for real-time pathology screening.

---

## 🔍 Key Engineering & Methodology Upgrades

This project represents a complete, rigorous refactoring of an initial proof-of-concept modeling codebase. Key engineering gaps were addressed:

1. **Elimination of Data Leakage (The Cardinal Sin of ML):**
   * *The Problem:* The original notebooks fit the `IterativeImputer`, scaling steps, and `SelectFromModel` feature selector on the *entire merged dataset* before splitting into training and test sets. This leaked parameters (e.g. mean, standard deviation, imputation models, and feature importances) from the test split, producing artificially inflated metrics.
   * *The Solution:* Preprocessing and feature selection are now strictly fit **only** on the training split (`X_train`, `y_train`). Test features (`X_test`) are transformed using the fitted state of training estimators, ensuring a mathematically rigorous validation.
2. **Consolidation of Redundant Scaling Chaining:**
   * *The Problem:* The original codebase sequentially chained `RobustScaler` $\rightarrow$ `StandardScaler` $\rightarrow$ `MinMaxScaler` on top of each other. This is statistically meaningless and distorted the feature space.
   * *The Solution:* Streamlined into a single `StandardScaler` layer to normalize features properly.
3. **Unified scikit-learn Pipeline Serialization:**
   * *The Problem:* Preprocessing steps and model classifiers were serialized as a clunky ad-hoc dictionary. During inference, this required manual, sequential unpacking and transformation.
   * *The Solution:* Integrated all steps directly into a standard unified `sklearn.pipeline.Pipeline` object. The serialized `ckd_knn_pipeline.joblib` artifact abstracts all preprocessing, imputation, scaling, feature selection, and classification. Predicting on raw data is now as simple as `pipeline.predict(X_new)`.
4. **Interactive Glassmorphic Web Dashboard:**
   * Transitioned from Jupyter-only workflows to a modular, production-ready system featuring a beautiful Flask web server and a premium Glassmorphic HTML5/JS dashboard.

---

## 📈 Model Performance & Metrics

Using the leakage-free pipeline configuration trained via **Stratified 5-Fold Cross-Validation** on the integrated clinical dataset (600 samples, 27 features), we achieved highly robust validation metrics:

### Best Model Parameters (from GridSearchCV):
* **Algorithm:** `auto`
* **Leaf Size:** `20`
* **Metric:** `euclidean`
* **N Neighbors:** `3`
* **Weights:** `distance`
* **Best CV Training Accuracy:** **99.17%**

### Independent Test Performance (Zero Leakage):
* **Test Accuracy:** **97.50%**
* **Precision:** **100%** (for CKD detection)
* **False Positives:** **0** (Critical for clinical diagnostic safety)

#### Confusion Matrix:
```text
               Predicted Healthy    Predicted CKD
True Healthy          44                 0
True CKD               3                73
```

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Healthy)** | 0.9362 | 1.0000 | 0.9670 | 44 |
| **1 (CKD)** | 1.0000 | 0.9605 | 0.9799 | 76 |
| **Accuracy** | | | **0.9750** | 120 |

---

## 📁 Repository Structure

```text
CHRONIC_KIDNEY_DISEASE/
├─ data/
│  ├─ raw/                      # Raw datasets (excluded in .gitignore)
│  │  ├─ ckd_full.csv           # UCI Dataset (400 samples)
│  │  └─ ckd-dataset-v2.csv     # Risk Factor Dataset (202 samples)
│  └─ processed/
│     └─ ckd_merged_corrected.csv # Preprocessed and merged clinical dataset
│
├─ models/
│  └─ ckd_knn_pipeline.joblib   # Unified scikit-learn Pipeline (Production Artifact)
│
├─ notebooks/
│  └─ eda_and_modelling.ipynb   # Streamlined EDA and pathology parameter check
│
├─ src/                         # Modular Python library
│  ├─ __init__.py
│  ├─ data_preprocessing.py     # Range conversion and dataset merging
│  ├─ train.py                  # Leakage-free cross-validation training pipeline
│  └─ predict.py                # Command-line prediction utility
│
├─ templates/
│  └─ index.html                # Premium interactive prediction dashboard UI
│
├─ app.py                       # Flask web app server
├─ README.md                    # System documentation
├─ .gitignore
└─ .gitattributes
```

---

## 🚀 Getting Started

### 1. Installation & Environment Setup
Clone the repository and install the standard machine learning dependencies (e.g. Anaconda or PyPI):
```powershell
pip install pandas numpy scikit-learn joblib flask matplotlib seaborn
```

### 2. Run Data Preprocessing
Integrate, clean, and merge the raw clinical datasets:
```powershell
python src/data_preprocessing.py
```

### 3. Run Pipeline Training & Validation
Execute leakage-free pipeline training, hyperparameter optimization, test evaluation, and model serialization:
```powershell
python src/train.py
```

### 4. Make Predictions via Command Line (CLI)
Test single predictions on clinical parameters directly from the terminal:
```powershell
python src/predict.py --age 55 --sg 1.015 --al 3.0 --hemo 10.5 --grf 42.0 --stage 3
```

### 5. Launch the Web Application Dashboard
Run the Flask server locally to launch the interactive, responsive Glassmorphic clinical dashboard:
```powershell
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to access the dashboard.

---

## 🔬 Clinical Datasets Integrated

1. **UCI CKD Dataset** (400 samples, 25 attributes): [Source Link](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
2. **Risk Factor Prediction of CKD (v2)** (200 samples, 29 attributes): [Source Link](https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease)
