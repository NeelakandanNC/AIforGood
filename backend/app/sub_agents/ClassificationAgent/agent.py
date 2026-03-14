"""
TriageAI — Classification Agent
Location: backend/app/sub_agents/ClassificationAgent/agent.py
"""

import pickle
import logging
import pandas as pd
import json
from pathlib import Path
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)

# ============================================================
# MODEL PATHS
# ============================================================

AGENT_DIR   = Path(__file__).parent
MODEL_PATH   = (AGENT_DIR / ".." / ".." / ".." / "model" / "model.pkl").resolve()
ENCODER_PATH = (AGENT_DIR / ".." / ".." / ".." / "model" / "label_encoder.pkl").resolve()

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
GENDER_MAP       = {"Female": 0, "Male": 1, "Other": 2}
REGION_MAP       = {"Central": 0, "East": 1, "North": 2, "Northeast": 3, "South": 4, "West": 5}
URBAN_RURAL_MAP  = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}
SMOKING_MAP      = {"Current": 0, "Former": 1, "Never": 2}
ALCOHOL_MAP      = {"Heavy": 0, "Never": 1, "Occasional": 2, "Regular": 3}
DISEASE_CAT_MAP  = {
    "Genetic": 0, "Infectious": 1, "Mental Health": 2,
    "Non-Communicable": 3, "Respiratory": 4,
    "Vector-Borne": 5, "Waterborne": 6
}
SEASON_MAP       = {"Monsoon": 0, "Post-Monsoon": 1, "Summer": 2, "Winter": 3}


class ClassificationAgentImpl(BaseAgent):
    """
    Pure code agent — XGBoost classification.
    Reads:  state["raw_data"]
    Writes: state["classification_result"]
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str):
        super().__init__(name=name, sub_agents=[])
        self._model         = None
        self._label_encoder = None
        self._load_model()

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def _load_model(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                self._label_encoder = pickle.load(f)
            logger.info(f"[{self.name}] ✅ Model loaded")
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Model load failed: {e}")

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    def _validate_input(self, user_input: dict):
        REQUIRED_FIELDS = [
            "age", "gender", "bmi",
            "region", "urban_rural",
            "disease_category", "season",
            "smoking_status", "alcohol_use"
        ]
        missing = [f for f in REQUIRED_FIELDS if f not in user_input]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    # ============================================================
    # BUILD MODEL INPUT
    # ============================================================

    def _build_model_input(self, user_input: dict) -> pd.DataFrame:

        symptoms   = user_input.get("symptoms", [])
        conditions = user_input.get("conditions", [])

        row = {
            "age":              user_input["age"],
            "gender":           GENDER_MAP.get(user_input["gender"], 1),
            "region":           REGION_MAP.get(user_input["region"], 0),
            "urban_rural":      URBAN_RURAL_MAP.get(user_input["urban_rural"], 2),
            "disease_category": DISEASE_CAT_MAP.get(user_input["disease_category"], 1),
            "season":           SEASON_MAP.get(user_input["season"], 0),
            "smoking_status":   SMOKING_MAP.get(user_input["smoking_status"], 2),
            "alcohol_use":      ALCOHOL_MAP.get(user_input["alcohol_use"], 1),
            "bmi":              user_input["bmi"],
        }

        for s in ALL_SYMPTOMS:
            row[f"symptom_{s}"] = int(s in symptoms)

        row["num_symptoms"] = len(symptoms)

        for c in ALL_CONDITIONS:
            row[f"condition_{c}"] = int(c in conditions)

        row["has_pre_existing"] = int(len(conditions) > 0)
        row["num_conditions"]   = len(conditions)

        return pd.DataFrame([row])

    # ============================================================
    # PREDICTION
    # ============================================================

    def _predict(self, df: pd.DataFrame) -> dict:

        risk_code     = self._model.predict(df)[0]
        risk_label    = self._label_encoder.inverse_transform([risk_code])[0]
        probabilities = self._model.predict_proba(df)[0]

        confidence = {
            str(cls): round(float(prob) * 100, 1)
            for cls, prob in zip(self._label_encoder.classes_, probabilities)
        }

        return {
            "risk_level":      risk_label,
            "risk_code":       int(risk_code),
            "confidence":      confidence,
            "max_confidence":  round(float(max(probabilities)) * 100, 1),
        }

    # ============================================================
    # DERIVED METRICS — vitals from frontend form (unchanged)
    # ============================================================

    def _compute_vital_severity(self, user_input: dict) -> dict:

        bp_sys = user_input.get("bp_systolic", 120)
        bp_dia = user_input.get("bp_diastolic", 80)
        hr     = user_input.get("heart_rate", 75)
        temp   = user_input.get("temperature", 98.6)
        spo2   = user_input.get("spo2", 97)

        score = 0

        if bp_sys >= 180 or bp_dia >= 120:   score += 3
        elif bp_sys >= 160 or bp_dia >= 100: score += 2
        elif bp_sys >= 140 or bp_dia >= 90:  score += 1
        elif bp_sys < 90:                    score += 3

        if hr > 130 or hr < 50:    score += 3
        elif hr > 110 or hr < 55:  score += 2
        elif hr > 100:             score += 1

        if temp >= 104.0:   score += 3
        elif temp >= 102.0: score += 2
        elif temp >= 100.4: score += 1

        if spo2 < 85:   score += 4
        elif spo2 < 90: score += 3
        elif spo2 < 94: score += 2
        elif spo2 < 96: score += 1

        conditions = user_input.get("conditions", [])
        severe   = {"heart_disease", "cancer", "kidney_disease", "hiv"}
        moderate = {"diabetes", "hypertension", "copd", "liver_disease"}

        comorbidity = 0
        for c in conditions:
            if c in severe:   comorbidity += 2
            elif c in moderate: comorbidity += 1

        return {
            "vital_severity_score": score,
            "vital_severity_level": (
                "critical" if score >= 8 else
                "elevated" if score >= 4 else
                "normal"
            ),
            "comorbidity_risk_score": comorbidity,
            "comorbidity_level": (
                "high"     if comorbidity >= 3 else
                "moderate" if comorbidity >= 1 else
                "none"
            ),
        }

    # ============================================================
    # DEBUG PRINT
    # ============================================================

    def _debug_print(self, result: dict):
        print("\n" + "=" * 60)
        print("🏥 CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"Patient      : {result['patient_name']} ({result['patient_id']})")
        print(f"Age / Gender : {result['age']} / {result['gender']}")
        print(f"BMI          : {result['bmi']}")
        print(f"Region       : {result['region']} | {result['urban_rural']}")
        print(f"Season       : {result['season']}")
        print(f"Smoking      : {result['smoking_status']} | Alcohol: {result['alcohol_use']}")
        print(f"Symptoms     : {', '.join(result['symptoms'])}")
        print(f"Conditions   : {', '.join(result['conditions']) or 'None'}")

        v = result["vitals"]
        print(
            f"Vitals       : BP={v['bp_systolic']}/{v['bp_diastolic']} | "
            f"HR={v['heart_rate']} | Temp={v['temperature']}°F | SpO₂={v['spo2']}%"
        )

        p = result["prediction"]
        print(f"\n🚦 Risk Level : {p['risk_level']}")
        print(f"📊 Confidence : {p['confidence']}")

        d = result["derived_metrics"]
        print(f"⚡ Vital Severity : {d['vital_severity_level']} ({d['vital_severity_score']})")
        print(f"💊 Comorbidity    : {d['comorbidity_level']} ({d['comorbidity_risk_score']})")
        print("=" * 60)
        print("\n=== RAW classification_result JSON ===")
        print(json.dumps(result, indent=2))
        print("=" * 60 + "\n")

    # ============================================================
    # MAIN EXECUTION
    # ============================================================

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        user_input = ctx.session.state.get("raw_data")

        if not user_input:
            print("\n❌ ERROR: raw_data missing in session state\n")
            return

        if not self._model or not self._label_encoder:
            print("\n❌ ERROR: Model artifacts not loaded\n")
            return

        patient_name = user_input.get("name", "Unknown")
        patient_id   = user_input.get("patient_id", "N/A")

        yield Event(
            author=self.name,
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=f"[{self.name}] 🔄 Classifying {patient_name}...")]
            ),
        )

        try:
            self._validate_input(user_input)
            df              = self._build_model_input(user_input)
            prediction      = self._predict(df)
            derived_metrics = self._compute_vital_severity(user_input)

        except Exception as e:
            print(f"\n❌ CLASSIFICATION ERROR: {e}\n")
            return

        classification_result = {
            "patient_id":    patient_id,
            "patient_name":  patient_name,
            "age":           user_input.get("age"),
            "gender":        user_input.get("gender"),
            "bmi":           user_input.get("bmi"),
            "region":        user_input.get("region"),
            "urban_rural":   user_input.get("urban_rural"),
            "disease_category": user_input.get("disease_category"),
            "season":        user_input.get("season"),
            "smoking_status": user_input.get("smoking_status"),
            "alcohol_use":   user_input.get("alcohol_use"),
            "symptoms":      user_input.get("symptoms", []),
            "conditions":    user_input.get("conditions", []),
            "vitals": {
                "bp_systolic":  user_input.get("bp_systolic"),
                "bp_diastolic": user_input.get("bp_diastolic"),
                "heart_rate":   user_input.get("heart_rate"),
                "temperature":  user_input.get("temperature"),
                "spo2":         user_input.get("spo2"),
            },
            "prediction":      prediction,
            "derived_metrics": derived_metrics,
        }

        ctx.session.state["classification_result"] = classification_result
        self._debug_print(classification_result)

        risk_level = prediction["risk_level"]
        max_conf   = prediction["max_confidence"]

        yield Event(
            author=self.name,
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=f"🚦 RISK LEVEL: {risk_level}\nConfidence: {max_conf}%")],
            ),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=json.dumps(classification_result))],
            ),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=f"[{self.name}] ✅ Classification Complete")],
            ),
        )


# ============================================================
# EXPORT
# ============================================================

ClassificationAgent = ClassificationAgentImpl(name="ClassificationAgent")