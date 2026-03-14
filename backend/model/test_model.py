"""
TriageAI — Model Test
Place at: backend/model/test.py
"""

import pickle
import pandas as pd
import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")

# ============================================================
# FEATURES (MUST MATCH TRAINING)
# ============================================================

ALL_SYMPTOMS = [
    "abdominal_pain", "anemia", "appetite_changes", "blurred_vision",
    "body_ache", "bone_deformities", "breathlessness", "burning_urination",
    "chest_pain", "chills", "cough", "cramps", "dehydration",
    "delayed_growth", "diarrhea", "dizziness", "excessive_worry",
    "fatigue", "fever", "frequent_infections", "headache", "irritability",
    "joint_pain", "loss_of_appetite", "loss_of_interest",
    "loss_of_smell_or_taste", "muscle_pain", "nausea", "numbness",
    "pain_episodes", "pale_skin", "panic_attacks", "persistent_sadness",
    "rash", "sleep_disturbance", "sore_throat", "sweating", "swelling",
    "swollen_lymph_nodes", "vomiting", "weight_loss", "wheezing"
]

ALL_CONDITIONS = [
    "diabetes", "heart_disease", "hypertension", "kidney_disease", "obesity"
]

# Encoded mappings (LabelEncoder alphabetical order)
GENDER_MAP        = {"Female": 0, "Male": 1, "Other": 2}
REGION_MAP        = {"Central": 0, "East": 1, "North": 2, "Northeast": 3, "South": 4, "West": 5}
URBAN_RURAL_MAP   = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}
SMOKING_MAP       = {"Current": 0, "Former": 1, "Never": 2}
ALCOHOL_MAP       = {"Heavy": 0, "Never": 1, "Occasional": 2, "Regular": 3}
DISEASE_CAT_MAP   = {
    "Genetic": 0, "Infectious": 1, "Mental Health": 2,
    "Non-Communicable": 3, "Respiratory": 4,
    "Vector-Borne": 5, "Waterborne": 6
}
SEASON_MAP        = {"Monsoon": 0, "Post-Monsoon": 1, "Summer": 2, "Winter": 3}

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)


def predict(patient_json: dict) -> dict:

    patient_id = patient_json.get("patient_id", f"PT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    name = patient_json.get("name", "Unknown")
    now = datetime.now()

    symptoms   = patient_json.get("symptoms", [])
    conditions = patient_json.get("conditions", [])

    # Build model input row
    row = {
        "age":             patient_json["age"],
        "gender":          GENDER_MAP.get(patient_json["gender"], 1),
        "region":          REGION_MAP.get(patient_json["region"], 0),
        "urban_rural":     URBAN_RURAL_MAP.get(patient_json["urban_rural"], 2),
        "disease_category": DISEASE_CAT_MAP.get(patient_json["disease_category"], 1),
        "season":          SEASON_MAP.get(patient_json["season"], 0),
        "smoking_status":  SMOKING_MAP.get(patient_json["smoking_status"], 2),
        "alcohol_use":     ALCOHOL_MAP.get(patient_json["alcohol_use"], 1),
        "bmi":             patient_json["bmi"],
    }

    for s in ALL_SYMPTOMS:
        row[f"symptom_{s}"] = 1 if s in symptoms else 0

    row["num_symptoms"] = len(symptoms)

    for c in ALL_CONDITIONS:
        row[f"condition_{c}"] = 1 if c in conditions else 0

    row["has_pre_existing"] = 1 if len(conditions) > 0 else 0
    row["num_conditions"]   = len(conditions)

    df = pd.DataFrame([row])

    risk_code    = model.predict(df)[0]
    risk_label   = le.inverse_transform([risk_code])[0]
    probabilities = model.predict_proba(df)[0]
    confidence   = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(le.classes_, probabilities)
    }

    output = {
        "patient_id": patient_id,
        "name":       name,
        "date":       now.strftime("%Y-%m-%d"),
        "time":       now.strftime("%H:%M"),
        "age":        patient_json["age"],
        "gender":     patient_json["gender"],
        "region":     patient_json["region"],
        "urban_rural": patient_json["urban_rural"],
        "disease_category": patient_json["disease_category"],
        "season":     patient_json["season"],
        "smoking_status": patient_json["smoking_status"],
        "alcohol_use": patient_json["alcohol_use"],
        "bmi":        patient_json["bmi"],
        "symptoms":   symptoms,
        "conditions": conditions,
        "result": {
            "prediction": risk_label,
            "confidence": {k: f"{v}%" for k, v in confidence.items()}
        }
    }

    return output


# ============================================================
# TEST PATIENTS
# ============================================================

if __name__ == "__main__":

    test_patients = [
        {
            "patient_id": "PT-2026-00125",
            "name": "Ramanathan Iyer",
            "age": 65,
            "gender": "Male",
            "region": "South",
            "urban_rural": "Urban",
            "disease_category": "Non-Communicable",
            "season": "Summer",
            "smoking_status": "Former",
            "alcohol_use": "Occasional",
            "bmi": 28.5,
            "symptoms": ["chest_pain", "breathlessness", "sweating", "dizziness"],
            "conditions": ["diabetes", "hypertension"]
        },
        {
            "patient_id": "PT-2026-00126",
            "name": "Priya Sharma",
            "age": 28,
            "gender": "Female",
            "region": "North",
            "urban_rural": "Urban",
            "disease_category": "Infectious",
            "season": "Winter",
            "smoking_status": "Never",
            "alcohol_use": "Never",
            "bmi": 22.1,
            "symptoms": ["fever", "cough", "sore_throat", "headache"],
            "conditions": []
        },
        {
            "patient_id": "PT-2026-00127",
            "name": "Lakshmi Devi",
            "age": 72,
            "gender": "Female",
            "region": "South",
            "urban_rural": "Rural",
            "disease_category": "Non-Communicable",
            "season": "Post-Monsoon",
            "smoking_status": "Never",
            "alcohol_use": "Never",
            "bmi": 31.2,
            "symptoms": ["dizziness", "fatigue", "nausea", "blurred_vision"],
            "conditions": ["diabetes", "hypertension", "kidney_disease"]
        }
    ]

    print("=" * 60)
    print("🏥 TriageAI — Model Test")
    print("=" * 60)

    for patient in test_patients:
        result = predict(patient)
        print(f"\n{'─' * 60}")
        print(f"Patient  : {result['name']} ({result['patient_id']})")
        print(f"Age      : {result['age']} | Gender: {result['gender']}")
        print(f"🚦 Prediction : {result['result']['prediction']}")
        print(f"📊 Confidence : {result['result']['confidence']}")
        print(f"\n📄 Full JSON:")
        print(json.dumps(result, indent=2))

    print(f"\n{'=' * 60}")
    print("✅ All predictions complete!")
    print("=" * 60)