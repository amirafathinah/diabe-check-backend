from pathlib import Path
import math
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# 1. KONFIGURASI DAN LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_risk_model_bundle.joblib"
SCHEMA_PATH = BASE_DIR / "model_schema.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

pipelines = bundle["pipelines"]
weights = bundle["weights"]
label_encoder = bundle["label_encoder"]

FEATURE_NAMES = bundle["feature_names"]
NUMERIC_FEATURES = bundle["numeric_features"]
CATEGORICAL_FEATURES = bundle["categorical_features"]

DEFAULT_NUMERIC = bundle["default_numeric"]
DEFAULT_CATEGORICAL = bundle["default_categorical"]
VALID_RANGES = bundle["valid_ranges"]

LOW_CLASS_INDEX = bundle["low_class_index"]
LOW_CLASS_THRESHOLD = bundle["low_class_threshold"]

CLASSES = [str(c) for c in label_encoder.classes_]


# =========================================================
# 2. INISIALISASI FASTAPI
# =========================================================

app = FastAPI(
    title="DiabeCheck API",
    description="API prediksi risiko diabetes menggunakan ensemble XGBoost dan LightGBM.",
    version="3.0.0"
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
# 3. FUNGSI ENSEMBLE
# =========================================================

def ensemble_proba(data: pd.DataFrame) -> np.ndarray:
    """
    Menggabungkan probabilitas XGBoost dan LightGBM
    menggunakan bobot dari hasil training.
    """

    p_xgb = pipelines["xgb"].predict_proba(data)
    p_lgbm = pipelines["lgbm"].predict_proba(data)

    return (
        float(weights["xgb"]) * p_xgb
        + float(weights["lgbm"]) * p_lgbm
    )


# =========================================================
# 4. VALIDASI DAN PENYUSUNAN INPUT
# =========================================================

def prepare_patient_data(payload: dict):
    """
    Menggabungkan input pengguna dengan nilai default model.
    Nilai numerik di luar rentang akan disesuaikan (clipping).
    """

    unknown_fields = set(payload.keys()) - set(FEATURE_NAMES)

    if unknown_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Field tidak dikenal: {sorted(unknown_fields)}"
        )

    patient = {}

    # Isi semua fitur dengan nilai default dari bundle
    patient.update(DEFAULT_NUMERIC)
    patient.update(DEFAULT_CATEGORICAL)

    warnings = []

    for feature, value in payload.items():

        # Abaikan nilai kosong
        if value is None or value == "" or value == "Unknown":
            continue

        # Validasi fitur numerik
        if feature in NUMERIC_FEATURES:

            if isinstance(value, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"{feature} harus berupa angka."
                )

            try:
                number = float(value)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Nilai {feature} harus berupa angka."
                )

            if not math.isfinite(number):
                raise HTTPException(
                    status_code=422,
                    detail=f"Nilai {feature} tidak valid."
                )

            if feature in VALID_RANGES:
                minimum, maximum = VALID_RANGES[feature]

                clipped = min(max(number, minimum), maximum)

                if clipped != number:
                    warnings.append(
                        f"{feature} disesuaikan dari {number} "
                        f"menjadi {clipped} sesuai rentang model."
                    )

                number = clipped

            patient[feature] = number

        # Validasi fitur kategorikal
        elif feature in CATEGORICAL_FEATURES:
            patient[feature] = str(value)

    # Susun DataFrame sesuai urutan fitur ketika training
    patient_df = pd.DataFrame(
        [patient],
        columns=FEATURE_NAMES
    )

    return patient_df, warnings


# =========================================================
# 5. ENDPOINT UTAMA
# =========================================================

@app.get("/")
def home():
    return {
        "name": "DiabeCheck API",
        "version": "3.0.0",
        "status": "running",
        "model": "XGBoost + LightGBM Ensemble"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/schema")
def get_schema():
    if not SCHEMA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="model_schema.json tidak ditemukan."
        )

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 6. ENDPOINT PREDIKSI
# =========================================================

@app.post("/predict")
def predict(payload: dict):

    try:
        # Siapkan data pasien
        patient_df, warnings = prepare_patient_data(payload)

        # Prediksi probabilitas ensemble
        probabilities_array = ensemble_proba(patient_df)[0]

        # Prediksi kelas berdasarkan probabilitas tertinggi
        predicted_index = int(np.argmax(probabilities_array))

        # Terapkan threshold khusus kelas Low
        threshold_applied = (
            probabilities_array[LOW_CLASS_INDEX]
            >= LOW_CLASS_THRESHOLD
        )

        if threshold_applied:
            predicted_index = LOW_CLASS_INDEX

        prediction = str(
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        # Susun probabilitas berdasarkan nama kelas
        probabilities = {
            class_name: round(float(probabilities_array[index]), 6)
            for index, class_name in enumerate(CLASSES)
        }

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "threshold_applied": bool(threshold_applied),
            "disclaimer": (
                "Skrining awal berbasis model AI. "
                "Hasil ini bukan diagnosis medis resmi."
            ),
            "warnings": warnings
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat melakukan prediksi: {str(error)}"
        )
