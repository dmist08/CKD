import os
import argparse
import pandas as pd
from joblib import load

MODEL_PATH = "models/ckd_knn_pipeline.joblib"

def predict_single_record(record_dict):
    """
    Predict CKD status for a single record dictionary.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model pipeline not found at {MODEL_PATH}. Please run train.py first.")
        
    pipeline = load(MODEL_PATH)
    
    # Construct a DataFrame with a single row
    # Use pandas to load since the pipeline expects a DataFrame structure
    df_new = pd.DataFrame([record_dict])
    
    # Run prediction
    pred = pipeline.predict(df_new)[0]
    prob = pipeline.predict_proba(df_new)[0]
    
    result = "Chronic Kidney Disease (CKD)" if pred == 1 else "Healthy (Non-CKD)"
    confidence = prob[pred] * 100
    
    return pred, result, confidence

def main():
    parser = argparse.ArgumentParser(description="Chronic Kidney Disease Prediction System CLI Utility")
    parser.add_argument("--age", type=float, default=50.0, help="Age in years")
    parser.add_argument("--bp", type=float, default=80.0, help="Diastolic Blood Pressure")
    parser.add_argument("--sg", type=float, default=1.020, help="Specific Gravity (e.g. 1.005 - 1.025)")
    parser.add_argument("--al", type=float, default=0.0, help="Albumin level (0 to 5)")
    parser.add_argument("--su", type=float, default=0.0, help="Sugar level (0 to 5)")
    parser.add_argument("--rbc", type=float, default=0.0, help="Red Blood Cells (0: normal, 1: abnormal)")
    parser.add_argument("--pc", type=float, default=0.0, help="Pus Cell (0: normal, 1: abnormal)")
    parser.add_argument("--pcc", type=float, default=0.0, help="Pus Cell Clumps (0: not present, 1: present)")
    parser.add_argument("--ba", type=float, default=0.0, help="Bacteria (0: not present, 1: present)")
    parser.add_argument("--bgr", type=float, default=120.0, help="Blood Glucose Random (mg/dl)")
    parser.add_argument("--bu", type=float, default=40.0, help="Blood Urea (mg/dl)")
    parser.add_argument("--sc", type=float, default=1.2, help="Serum Creatinine (mg/dl)")
    parser.add_argument("--sod", type=float, default=138.0, help="Sodium (mEq/L)")
    parser.add_argument("--pot", type=float, default=4.1, help="Potassium (mEq/L)")
    parser.add_argument("--hemo", type=float, default=15.0, help="Hemoglobin (g/dl)")
    parser.add_argument("--pcv", type=float, default=44.0, help="Packed Cell Volume (percentage)")
    parser.add_argument("--rbcc", type=float, default=5.2, help="Red Blood Cell Count (millions/cmm)")
    parser.add_argument("--wbcc", type=float, default=7800.0, help="White Blood Cell Count (cells/cmm)")
    parser.add_argument("--htn", type=float, default=0.0, help="Hypertension (0: no, 1: yes)")
    parser.add_argument("--dm", type=float, default=0.0, help="Diabetes Mellitus (0: no, 1: yes)")
    parser.add_argument("--cad", type=float, default=0.0, help="Coronary Artery Disease (0: no, 1: yes)")
    parser.add_argument("--appet", type=float, default=0.0, help="Appetite (0: good, 1: poor)")
    parser.add_argument("--pe", type=float, default=0.0, help="Pedal Edema (0: no, 1: yes)")
    parser.add_argument("--ane", type=float, default=0.0, help="Anemia (0: no, 1: yes)")
    parser.add_argument("--grf", type=float, default=95.0, help="Glomerular Filtration Rate")
    parser.add_argument("--stage", type=float, default=1.0, help="CKD Stage (1.0 to 5.0)")
    
    args = parser.parse_args()
    
    # Build complete record matching the 27 features format
    record = {
        "bp (Diastolic)": args.bp,
        "bp limit": args.bp,  # aligned mapping helper
        "sg": args.sg,
        "al": args.al,
        "rbc": args.rbc,
        "su": args.su,
        "pc": args.pc,
        "pcc": args.pcc,
        "ba": args.ba,
        "bgr": args.bgr,
        "bu": args.bu,
        "sod": args.sod,
        "sc": args.sc,
        "pot": args.pot,
        "hemo": args.hemo,
        "pcv": args.pcv,
        "rbcc": args.rbcc,
        "wbcc": args.wbcc,
        "htn": args.htn,
        "dm": args.dm,
        "cad": args.cad,
        "appet": args.appet,
        "pe": args.pe,
        "ane": args.ane,
        "grf": args.grf,
        "stage": args.stage,
        "age": args.age
    }
    
    try:
        pred, result, confidence = predict_single_record(record)
        print("\n=== Chronic Kidney Disease Prediction System ===")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("================================================\n")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
