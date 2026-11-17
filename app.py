from pathlib import Path
import math
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# 1. KONFIGURASI DAN LOAD MODEL (V4 STACKING)
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# File paths
PREP_PATH = MODEL_DIR / "preprocessor.joblib"
XGB_PATH = MODEL_DIR / "xgb_model.json"
LGBM_PATH = MODEL_DIR / "lgbm_model.txt"
HGB_PATH = MODEL_DIR / "hgb_model.joblib"
META_PATH = MODEL_DIR / "meta_model.joblib"
META_DATA_PATH = MODEL_DIR / "metadata.json"

# Cek keberadaan file
for path in [PREP_PATH, XGB_PATH, LGBM_PATH, HGB_PATH, META_PATH, META_DATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

# Load Preprocessor & Meta
prep = joblib.load(PREP_PATH)
meta = joblib.load(META_PATH)
hgb_model = joblib.load(HGB_PATH)

# Load XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

# Load LightGBM (sebagai Booster)
lgbm_model = lgb.Booster(model_file=LGBM_PATH)

# Load Metadata
with open(META_DATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
BIAS = np.array(metadata["bias"])
RAW_COLS = metadata["raw_cols"]

# =========================================================
# 2. INISIALISASI FASTAPI
# =========================================================

app = FastAPI(
    title="DiabeCheck API v4",
    description="API prediksi risiko diabetes menggunakan Stacking (XGB, LGBM, HGB + Meta-LR).",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabe-check-omega.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# =========================================================
# 3. FUNGSI FEATURE ENGINEERING (SAMA PERSIS DENGAN COLAB V4)
# =========================================================

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    eps = 1e-6
    d["eAG_from_HbA1c"] = 28.7 * d["HbA1c"] - 46.7
    d["Glucose_minus_eAG"] = d["Blood_Glucose"] - d["eAG_from_HbA1c"]
    d["Glucose_minus_Fasting"] = d["Blood_Glucose"] - d["Fasting_Blood_Sugar"]
    d["HOMA_IR"] = d["Insulin_Level"] * d["Fasting_Blood_Sugar"] / 405.0
    d["BMI_calc"] = d["Weight_kg"] / ((d["Height_cm"] / 100.0) ** 2 + eps)
    d["BMI_gap"] = d["BMI"] - d["BMI_calc"]
    d["Waist_to_Height"] = d["Waist_Circumference_cm"] / (d["Height_cm"] + eps)
    d["Chol_HDL_ratio"] = d["Total_Cholesterol"] / (d["HDL"] + eps)
    d["TG_HDL_ratio"] = d["Triglycerides"] / (d["HDL"] + eps)
    d["LDL_HDL_ratio"] = d["LDL"] / (d["HDL"] + eps)
    d["Non_HDL"] = d["Total_Cholesterol"] - d["HDL"]
    d["Pulse_Pressure"] = d["Blood_Pressure_Systolic"] - d["Blood_Pressure_Diastolic"]
    d["MAP"] = d["Blood_Pressure_Diastolic"] + d["Pulse_Pressure"] / 3.0
    d["Active_Min_Week"] = d["Exercise_Hours_Per_Week"] * 60.0 + d["Daily_Walking_Minutes"] * 7.0
    d["Age_x_BMI"] = d["Age"] * d["BMI"]
    d["Age_x_HbA1c"] = d["Age"] * d["HbA1c"]

    def stage(col, lo, hi):
        v = d[col].to_numpy(dtype=float)
        out = np.where(v >= hi, 2.0, np.where(v >= lo, 1.0, 0.0))
        return np.where(np.isnan(v), np.nan, out)

    d["HbA1c_stage"] = stage("HbA1c", 5.7, 6.5)
    d["FBS_stage"] = stage("Fasting_Blood_Sugar", 100, 126)
    d["Glucose_stage"] = stage("Blood_Glucose", 140, 200)
    
    stages = d[["HbA1c_stage", "FBS_stage", "Glucose_stage"]]
    d["Glycemic_max_stage"] = stages.max(axis=1)
    d["Glycemic_sum_stage"] = stages.sum(axis=1)

    present = [c for c in ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"] if c in d.columns]
    for c in present:
        d[c + "_bin"] = d[c].map({"Yes": 1.0, "No": 0.0})
    d["Comorbidity_count"] = d[[c + "_bin" for c in present]].sum(axis=1)
    
    return d

# =========================================================
# 4. VALIDASI DAN PENYUSUNAN INPUT
# =========================================================

def prepare_patient_data(payload: dict):
    patient = {}
    
    # Inisialisasi dengan NaN agar diimputasi oleh preprocessor
    for col in RAW_COLS:
        patient[col] = np.nan

    warnings = []
    
    for feature, value in payload.items():
        if feature not in patient:
            continue # Abaikan field yang tidak dikenal model (seperti field form tambahan)
            
        if value is None or value == "" or value == "Unknown":
            continue

        # Jika numerik, pastikan float
        try:
            if isinstance(value, (int, float)):
                patient[feature] = float(value)
            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                patient[feature] = float(value)
            else:
                patient[feature] = str(value)
        except:
            patient[feature] = value

    # Buat DataFrame
    df = pd.DataFrame([patient])
    
    # Lakukan Feature Engineering
    df_fe = add_features(df)
    
    return df_fe, warnings

# =========================================================
# 5. ENDPOINT UTAMA
# =========================================================

@app.get("/")
def home():
    return {
        "name": "DiabeCheck API",
        "version": "4.0.0",
        "status": "running",
        "model": "Stacking (XGB + LGBM + HGB -> Meta LR)"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# =========================================================
# 6. ENDPOINT PREDIKSI
# =========================================================

@app.post("/predict")
def predict(payload: dict):
    try:
        # 1. Siapkan data
        patient_df, warnings = prepare_patient_data(payload)

        # 2. Preprocess (Ordinal Encoding + Imputasi)
        X_proc = prep.transform(patient_df).astype(np.float32)

        # 3. Base Models Prediction
        p_xgb = xgb_model.predict_proba(X_proc)
        p_lgbm = lgbm_model.predict(X_proc) # Booster returns proba directly
        p_hgb = hgb_model.predict_proba(X_proc)

        # 4. Meta Features (Log of probabilities)
        Z = np.hstack([
            np.log(np.clip(p_xgb, 1e-6, 1.0)),
            np.log(np.clip(p_lgbm, 1e-6, 1.0)),
            np.log(np.clip(p_hgb, 1e-6, 1.0))
        ])

        # 5. Meta Learner Prediction
        proba = meta.predict_proba(Z)[0]

        # 6. Apply Class Bias (bobot kelas dari OOF)
        final_proba = proba * BIAS
        pred_idx = int(final_proba.argmax())

        prediction = CLASSES[pred_idx]

        # 7. Format output
        probabilities = {
            cls: round(float(proba[i]), 6) 
            for i, cls in enumerate(CLASSES)
        }

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "threshold_applied": False, # Di v4, bias kelas menggantikan threshold manual
            "disclaimer": "Skrining awal berbasis AI. Hasil ini bukan diagnosis medis resmi.",
            "warnings": warnings
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat melakukan prediksi: {str(error)}"
        )
