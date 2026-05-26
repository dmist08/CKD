import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.experimental import enable_iterative_imputer  # noqa

app = Flask(__name__, template_folder="templates")
MODEL_PATH = "models/ckd_knn_pipeline.joblib"

# Load the production ML pipeline
if os.path.exists(MODEL_PATH):
    print(f"Loading final serialized ML pipeline from {MODEL_PATH}...")
    pipeline = load(MODEL_PATH)
else:
    print(f"WARNING: Serialized model not found at {MODEL_PATH}. Please run train.py to train it.")
    pipeline = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = load(MODEL_PATH)
        else:
            return jsonify({"error": "Serialized ML pipeline not found. Please train the model first."}), 500
            
    try:
        data = request.get_json()
        
        # Align incoming inputs with the exact 27 features expected by the training pipeline
        record = {
            "bp (Diastolic)": data.get("bp", 80.0),
            "bp limit": data.get("bp", 80.0),
            "sg": data.get("sg", 1.020),
            "al": data.get("al", 0.0),
            "rbc": data.get("rbc", 0.0),
            "su": data.get("su", 0.0),
            "pc": data.get("pc", 0.0),
            "pcc": data.get("pcc", 0.0),
            "ba": data.get("ba", 0.0),
            "bgr": data.get("bgr", 120.0),
            "bu": data.get("bu", 40.0),
            "sod": data.get("sod", 138.0),
            "sc": data.get("sc", 1.2),
            "pot": data.get("pot", 4.1),
            "hemo": data.get("hemo", 15.0),
            "pcv": data.get("pcv", 44.0),
            "rbcc": data.get("rbcc", 5.2),
            "wbcc": data.get("wbcc", 7800.0),
            "htn": data.get("htn", 0.0),
            "dm": data.get("dm", 0.0),
            "cad": data.get("cad", 0.0),
            "appet": data.get("appet", 0.0),
            "pe": data.get("pe", 0.0),
            "ane": data.get("ane", 0.0),
            "grf": data.get("grf", 95.0),
            "stage": data.get("stage", 1.0),
            "age": data.get("age", 45.0)
        }
        
        # Convert to single-row DataFrame
        df_new = pd.DataFrame([record])
        
        # Run prediction on the pipeline
        pred = int(pipeline.predict(df_new)[0])
        prob = pipeline.predict_proba(df_new)[0]
        confidence = float(prob[pred] * 100)
        
        return jsonify({
            "prediction": pred,
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
