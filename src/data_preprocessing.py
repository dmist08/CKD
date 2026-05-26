import os
import re
import numpy as np
import pandas as pd

def convert_range_to_numeric(x):
    """
    Convert text ranges to their numeric midpoints.
    Examples:
        '1.019 - 1.021' -> 1.020
        '≥ 227.944' -> 227.944
        '< 12' -> 6.0 (midpoint from 0 to 12)
        's1' -> 1.0 (stage value)
    """
    if pd.isna(x):
        return np.nan
    
    s = str(x).strip().lower()

    # Stage values: s1 - s5
    if re.match(r"s[1-5]", s):
        return float(s[1:])  # "s3" -> 3.0

    # Greater than equal: "≥ X"
    if s.startswith("≥"):
        try:
            return float(s.replace("≥", "").strip())
        except ValueError:
            pass

    # Less than: "< X"
    if s.startswith("<"):
        try:
            num = float(s.replace("<", "").strip())
            return num / 2.0  # midpoint from 0 to X
        except ValueError:
            pass

    # Range: "a - b"
    if "-" in s:
        parts = s.split("-")
        try:
            a = float(parts[0])
            b = float(parts[1])
            return (a + b) / 2.0
        except ValueError:
            pass

    # Single numeric string
    try:
        return float(s)
    except ValueError:
        return np.nan

def map_class(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "ckd":
        return 1
    if s == "notckd":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return np.nan

def normalize_binary(val):
    if pd.isna(val):
        return np.nan
    binary_map = {
        "yes": 1.0, "y": 1.0, "present": 1.0, "abnormal": 1.0, "good": 1.0,
        "no": 0.0, "n": 0.0, "notpresent": 0.0, "normal": 0.0, "poor": 0.0
    }
    s = str(val).strip().lower()
    return binary_map.get(s, np.nan if s not in ["1", "0", "1.0", "0.0"] else float(s))

def preprocess_datasets(uci_path, v2_path, output_path):
    print(f"Loading raw datasets:\n  UCI: {uci_path}\n  V2:  {v2_path}")
    
    if not os.path.exists(uci_path):
        raise FileNotFoundError(f"UCI dataset not found at {uci_path}")
    if not os.path.exists(v2_path):
        raise FileNotFoundError(f"V2 dataset not found at {v2_path}")
        
    df_full = pd.read_csv(uci_path)
    df_v2 = pd.read_csv(v2_path)
    
    print(f"Original shapes -> UCI: {df_full.shape}, V2: {df_v2.shape}")
    
    # 1. Clean V2 Dataset
    # Drop first two metadata rows (indices 0 and 1)
    df_v2_clean = df_v2.drop(index=[0, 1]).reset_index(drop=True)
    
    # Convert ranges and stages to midpoints
    df_v2_numeric = df_v2_clean.copy()
    for col in df_v2_numeric.columns:
        if col not in ["class", "affected"]:
            df_v2_numeric[col] = df_v2_numeric[col].apply(convert_range_to_numeric)
            
    # Map target column
    df_v2_numeric["affected"] = df_v2_clean["class"].apply(map_class)
    df_v2_numeric = df_v2_numeric.drop(columns=["class"])
    
    # 2. Clean UCI Dataset
    df_full_clean = df_full.copy()
    df_full_clean.columns = df_full_clean.columns.str.lower().str.strip()
    
    uci_to_v2_mapping = {
        "age": "age",
        "blood pressure": "bp limit",
        "specific gravity": "sg",
        "albumin": "al",
        "sugar": "su",
        "red blood cells": "rbc",
        "pus cell": "pc",
        "pus cell clumps": "pcc",
        "bacteria": "ba",
        "blood glucose random": "bgr",
        "blood urea": "bu",
        "serum creatinine": "sc",
        "sodium": "sod",
        "potassium": "pot",
        "hemoglobin": "hemo",
        "packed cell volume": "pcv",
        "red blood cell count": "rbcc",
        "white blood cell count": "wbcc",
        "hypertension": "htn",
        "diabetes mellitus": "dm",
        "coronary artery disease": "cad",
        "appetite": "appet",
        "pedal edema": "pe",
        "anemia": "ane",
        "class": "affected",
    }
    
    df_full_clean = df_full_clean.rename(columns={col: uci_to_v2_mapping.get(col, col) 
                                                  for col in df_full_clean.columns})
    
    # Map text values to binaries
    binary_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    for col in binary_cols:
        if col in df_full_clean.columns:
            df_full_clean[col] = df_full_clean[col].apply(normalize_binary)
            
    df_full_clean["affected"] = df_full_clean["affected"].apply(map_class)
    
    # 3. Merge Datasets
    v2_cols = df_v2_numeric.columns.tolist()
    full_cols = df_full_clean.columns.tolist()
    
    # Align columns
    for col in v2_cols:
        if col not in df_full_clean.columns:
            df_full_clean[col] = np.nan
            
    for col in full_cols:
        if col not in df_v2_numeric.columns:
            df_v2_numeric[col] = np.nan
            
    df_full_clean = df_full_clean[v2_cols]
    df_v2_numeric = df_v2_numeric[v2_cols]
    
    # Concatenate both dataframes
    df_merged_final = pd.concat([df_v2_numeric, df_full_clean], ignore_index=True)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged_final.to_csv(output_path, index=False)
    print(f"Successfully preprocessed and merged datasets. Merged shape: {df_merged_final.shape}")
    print(f"Saved processed dataset to: {output_path}")

if __name__ == "__main__":
    # If run standalone, use standard relative paths from project root
    uci = "data/raw/ckd_full.csv"
    v2 = "data/raw/ckd-dataset-v2.csv"
    out = "data/processed/ckd_merged_corrected.csv"
    
    try:
        preprocess_datasets(uci, v2, out)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
