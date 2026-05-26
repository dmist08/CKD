import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load

# Config
DATA_PATH = "data/processed/ckd_merged_corrected.csv"
MODEL_PATH = "models/ckd_knn_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def train_pipeline():
    print(f"Loading processed dataset: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. Please run data_preprocessing.py first.")
        
    df = pd.read_csv(DATA_PATH)
    
    target_col = "affected"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int)
    
    # 1. Leakage-free split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Split data into train {X_train.shape} and test {X_test.shape}")
    
    # 2. Sequential Preprocessing on Train Split only (Zero Leakage!)
    print("\nFitting preprocessing steps on Train Split...")
    
    # Step A: Imputation
    imputer = IterativeImputer(max_iter=20, random_state=RANDOM_STATE)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Step B: Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Step C: Feature Selection
    print("Fitting RandomForest for feature selection...")
    rf_selector = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf_selector.fit(X_train_scaled, y_train)
    
    selector = SelectFromModel(rf_selector, threshold="median", prefit=True)
    support_mask = selector.get_support()
    selected_features = np.array(X.columns)[support_mask].tolist()
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}:")
    print(selected_features)
    
    X_train_sel = X_train_scaled[:, support_mask]
    X_test_sel = X_test_scaled[:, support_mask]
    
    # 3. Fast Grid Search for KNN on Preprocessed Train Features
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "metric": ["euclidean", "manhattan"],
        "weights": ["uniform", "distance"],
        "leaf_size": [20, 30],
        "algorithm": ["auto", "kd_tree"]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print("\nStarting hyperparameter tuning via GridSearchCV on preprocessed train split...")
    grid = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train_sel, y_train)
    
    print("\nBest Parameters Found:")
    for param_name, param_val in grid.best_params_.items():
        print(f"  {param_name}: {param_val}")
    print(f"Best CV Training Accuracy: {grid.best_score_ * 100:.2f}%")
    
    # 4. Evaluate on Independent Test Set (Zero Leakage Validation!)
    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X_test_sel)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n=========================================")
    print(f"Independent Test Accuracy (Leakage-Free): {test_acc * 100:.2f}%")
    print(f"=========================================")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 5. Fit unified production pipeline on FULL dataset for deployment
    print("\nRefitting best pipeline configuration on full dataset...")
    
    production_pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=20, random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("feature_selector", SelectFromModel(
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample"
            ),
            threshold="median"
        )),
        ("model", KNeighborsClassifier(**grid.best_params_))
    ])
    
    production_pipeline.fit(X, y)
    
    # Save the pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(production_pipeline, MODEL_PATH)
    print(f"Successfully serialized and saved final production pipeline to: {MODEL_PATH}")
    
    # 6. Quick verification of the serialized model
    print("\nVerifying serialized pipeline loading and execution...")
    loaded_pipeline = load(MODEL_PATH)
    test_record = X.iloc[[0]].copy()
    pred = loaded_pipeline.predict(test_record)
    pred_prob = loaded_pipeline.predict_proba(test_record)
    print(f"Test record prediction: {pred[0]} (Probability: {pred_prob[0]})")
    print("Pipeline verified successfully!")

if __name__ == "__main__":
    train_pipeline()
