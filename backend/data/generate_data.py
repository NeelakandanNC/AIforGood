"""
TriageAI — Synthetic Patient Data Generator (v2 - One-Hot Encoded)
Generates 1,00,000 medically coherent patient records for Indian district hospital context.
All symptoms and conditions are one-hot encoded for direct XGBoost training.
"""

import pandas as pd
import numpy as np
import random
import uuid
import os

np.random.seed(42)
random.seed(42)

NUM_PATIENTS = 100000

# ============================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================

ALL_SYMPTOMS = [
    "chest_pain", "breathlessness", "headache", "fever", "cough",
    "abdominal_pain", "nausea", "vomiting", "dizziness", "fatigue",
    "palpitations", "back_pain", "joint_pain", "diarrhea", "sore_throat",
    "body_ache", "weakness", "blurred_vision", "numbness", "confusion",
    "seizures", "blood_in_stool", "weight_loss", "sweating", "swelling",
    "burning_urination", "rash", "cold", "wheezing", "loss_of_appetite"
]

ALL_CONDITIONS = [
    "diabetes", "hypertension", "asthma", "copd", "heart_disease",
    "kidney_disease", "liver_disease", "thyroid", "tuberculosis",
    "cancer", "hiv", "anemia", "obesity"
]

SYMPTOM_CLUSTERS = {
    "cardiac": {
        "symptoms": ["chest_pain", "breathlessness", "palpitations", "sweating", "dizziness", "nausea"],
        "conditions": ["heart_disease", "hypertension", "diabetes"],
        "age_bias": "elderly",
    },
    "respiratory": {
        "symptoms": ["breathlessness", "cough", "fever", "wheezing", "fatigue", "chest_pain"],
        "conditions": ["asthma", "copd", "tuberculosis"],
        "age_bias": "any",
    },
    "neurological": {
        "symptoms": ["headache", "dizziness", "confusion", "numbness", "blurred_vision", "seizures", "weakness"],
        "conditions": ["hypertension", "diabetes"],
        "age_bias": "elderly",
    },
    "gastrointestinal": {
        "symptoms": ["abdominal_pain", "nausea", "vomiting", "diarrhea", "blood_in_stool", "loss_of_appetite"],
        "conditions": ["liver_disease", "diabetes"],
        "age_bias": "any",
    },
    "general_infection": {
        "symptoms": ["fever", "cough", "body_ache", "fatigue", "sore_throat", "cold", "headache"],
        "conditions": [],
        "age_bias": "young",
    },
    "musculoskeletal": {
        "symptoms": ["back_pain", "joint_pain", "weakness", "swelling", "fatigue"],
        "conditions": ["obesity", "diabetes"],
        "age_bias": "middle",
    },
    "urinary": {
        "symptoms": ["burning_urination", "fever", "abdominal_pain", "back_pain", "nausea"],
        "conditions": ["diabetes", "kidney_disease"],
        "age_bias": "any",
    },
    "mild_common": {
        "symptoms": ["cold", "cough", "sore_throat", "headache", "body_ache", "fatigue"],
        "conditions": [],
        "age_bias": "any",
    }
}

CLUSTER_WEIGHTS = {
    "cardiac": 0.12,
    "respiratory": 0.15,
    "neurological": 0.08,
    "gastrointestinal": 0.13,
    "general_infection": 0.25,
    "musculoskeletal": 0.10,
    "urinary": 0.07,
    "mild_common": 0.10,
}


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_age(age_bias):
    if age_bias == "elderly":
        r = random.random()
        if r < 0.6:
            return random.randint(50, 90)
        elif r < 0.9:
            return random.randint(30, 49)
        else:
            return random.randint(18, 29)
    elif age_bias == "young":
        r = random.random()
        if r < 0.5:
            return random.randint(18, 35)
        elif r < 0.8:
            return random.randint(36, 55)
        else:
            return random.randint(56, 80)
    elif age_bias == "middle":
        return random.randint(30, 65)
    else:
        return random.randint(18, 85)


def pick_symptoms(cluster):
    num = random.randint(2, 5)
    symptoms = set(random.sample(
        cluster["symptoms"],
        min(num, len(cluster["symptoms"]))
    ))
    if random.random() < 0.20:
        pool = [s for s in ALL_SYMPTOMS if s not in symptoms]
        symptoms.add(random.choice(pool))
    return symptoms


def pick_conditions(cluster):
    conditions = set()
    if random.random() < 0.35 and cluster["conditions"]:
        num = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
        available = cluster["conditions"]
        conditions = set(random.sample(available, min(num, len(available))))
        if random.random() < 0.15:
            pool = [c for c in ALL_CONDITIONS if c not in conditions]
            conditions.add(random.choice(pool))
    return conditions


def generate_vitals(age, symptoms, conditions, cluster_type):
    bp_sys = np.random.normal(120, 10)
    bp_dia = np.random.normal(80, 8)
    hr = np.random.normal(75, 8)
    temp = np.random.normal(98.6, 0.3)
    spo2 = np.random.normal(97, 1)

    if age > 60:
        bp_sys += random.uniform(5, 20)
        bp_dia += random.uniform(3, 10)
        spo2 -= random.uniform(0, 2)

    if "hypertension" in conditions:
        bp_sys += random.uniform(20, 50)
        bp_dia += random.uniform(10, 25)

    if "diabetes" in conditions:
        bp_sys += random.uniform(5, 15)
        hr += random.uniform(5, 15)

    if "heart_disease" in conditions:
        hr += random.uniform(10, 30)
        bp_sys += random.uniform(10, 30)

    if "asthma" in conditions or "copd" in conditions:
        spo2 -= random.uniform(2, 8)
        hr += random.uniform(5, 15)

    if "anemia" in conditions:
        hr += random.uniform(10, 20)
        spo2 -= random.uniform(1, 3)

    if "fever" in symptoms:
        temp += random.uniform(1.0, 4.0)
        hr += random.uniform(5, 20)

    if "breathlessness" in symptoms:
        spo2 -= random.uniform(1, 6)
        hr += random.uniform(5, 15)

    if "chest_pain" in symptoms:
        hr += random.uniform(5, 25)
        bp_sys += random.uniform(5, 20)

    if "seizures" in symptoms:
        hr += random.uniform(10, 30)
        bp_sys += random.uniform(10, 25)

    if "vomiting" in symptoms or "diarrhea" in symptoms:
        bp_sys -= random.uniform(5, 15)
        hr += random.uniform(5, 15)

    if cluster_type == "cardiac" and random.random() < 0.10:
        bp_sys += random.uniform(30, 60)
        hr += random.uniform(20, 50)
        spo2 -= random.uniform(3, 8)

    if cluster_type == "respiratory" and random.random() < 0.10:
        spo2 -= random.uniform(5, 15)
        hr += random.uniform(15, 30)

    bp_sys = int(np.clip(bp_sys, 70, 220))
    bp_dia = int(np.clip(bp_dia, 40, 140))
    hr = int(np.clip(hr, 40, 180))
    temp = round(float(np.clip(temp, 95.0, 106.0)), 1)
    spo2 = int(np.clip(spo2, 60, 100))

    if bp_sys <= bp_dia:
        bp_sys = bp_dia + random.randint(10, 30)

    return bp_sys, bp_dia, hr, temp, spo2


def determine_risk(age, symptoms, conditions, bp_sys, bp_dia, hr, temp, spo2):
    score = 0

    # Vitals
    if bp_sys >= 180 or bp_dia >= 120:
        score += 4
    elif bp_sys >= 160 or bp_dia >= 100:
        score += 2
    elif bp_sys >= 140 or bp_dia >= 90:
        score += 1
    elif bp_sys < 90:
        score += 3

    if hr > 130 or hr < 50:
        score += 3
    elif hr > 110 or hr < 55:
        score += 2
    elif hr > 100:
        score += 1

    if temp >= 104.0:
        score += 3
    elif temp >= 102.0:
        score += 2
    elif temp >= 100.4:
        score += 1
    elif temp < 96.0:
        score += 2

    if spo2 < 85:
        score += 5
    elif spo2 < 90:
        score += 3
    elif spo2 < 94:
        score += 2
    elif spo2 < 96:
        score += 1

    # Symptoms
    critical = {"chest_pain", "seizures", "confusion", "blood_in_stool"}
    moderate = {"breathlessness", "numbness", "blurred_vision", "palpitations", "vomiting"}

    for s in symptoms:
        if s in critical:
            score += 3
        elif s in moderate:
            score += 1.5

    # Conditions
    severe = {"heart_disease", "cancer", "kidney_disease", "hiv"}
    mid = {"diabetes", "hypertension", "copd", "liver_disease"}

    for c in conditions:
        if c in severe:
            score += 2
        elif c in mid:
            score += 1

    # Age
    if age > 70:
        score += 2
    elif age > 60:
        score += 1

    # Dangerous combos
    if age > 60 and "diabetes" in conditions and "chest_pain" in symptoms:
        score += 3
    if age > 60 and symptoms & {"dizziness", "confusion", "weakness"}:
        score += 2
    if "breathlessness" in symptoms and spo2 < 92:
        score += 2
    if "chest_pain" in symptoms and "sweating" in symptoms:
        score += 3

    if score >= 8:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"


def generate_patient():
    cluster_type = random.choices(
        list(CLUSTER_WEIGHTS.keys()),
        weights=list(CLUSTER_WEIGHTS.values()),
        k=1
    )[0]
    cluster = SYMPTOM_CLUSTERS[cluster_type]

    age = generate_age(cluster["age_bias"])
    gender = random.choice([0, 1])  # 0=Male, 1=Female
    symptoms = pick_symptoms(cluster)
    conditions = pick_conditions(cluster)
    bp_sys, bp_dia, hr, temp, spo2 = generate_vitals(age, symptoms, conditions, cluster_type)
    risk = determine_risk(age, symptoms, conditions, bp_sys, bp_dia, hr, temp, spo2)

    # Build row
    row = {"patient_id": str(uuid.uuid4())[:8], "age": age, "gender": gender}

    for s in ALL_SYMPTOMS:
        row[f"symptom_{s}"] = 1 if s in symptoms else 0

    row["bp_systolic"] = bp_sys
    row["bp_diastolic"] = bp_dia
    row["heart_rate"] = hr
    row["temperature"] = temp
    row["spo2"] = spo2

    for c in ALL_CONDITIONS:
        row[f"condition_{c}"] = 1 if c in conditions else 0

    row["has_pre_existing"] = 1 if len(conditions) > 0 else 0
    row["num_symptoms"] = len(symptoms)
    row["num_conditions"] = len(conditions)
    row["risk_level"] = risk

    return row


# ============================================================
# MAIN
# ============================================================

def main():
    print("🏥 TriageAI — Generating Synthetic Patient Data (v2 - One-Hot)")
    print(f"   Generating {NUM_PATIENTS} records...\n")

    patients = [generate_patient() for _ in range(NUM_PATIENTS)]
    df = pd.DataFrame(patients)

    print("=" * 55)
    print("📊 Dataset Statistics")
    print("=" * 55)

    print(f"\nTotal Records:  {len(df)}")
    print(f"Total Features: {len(df.columns) - 2}")

    symptom_cols = [c for c in df.columns if c.startswith("symptom_")]
    condition_cols = [c for c in df.columns if c.startswith("condition_")]
    vital_cols = ["bp_systolic", "bp_diastolic", "heart_rate", "temperature", "spo2"]
    other_cols = ["age", "gender", "has_pre_existing", "num_symptoms", "num_conditions"]

    print(f"\n📋 Feature Breakdown:")
    print(f"   Symptom features:   {len(symptom_cols)}")
    print(f"   Condition features: {len(condition_cols)}")
    print(f"   Vital features:     {len(vital_cols)}")
    print(f"   Other features:     {len(other_cols)}")
    print(f"   Total:              {len(symptom_cols) + len(condition_cols) + len(vital_cols) + len(other_cols)}")

    print(f"\n🚦 Risk Level Distribution:")
    risk_counts = df["risk_level"].value_counts()
    for level in ["Low", "Medium", "High"]:
        count = risk_counts.get(level, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {level:6s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n👤 Gender: Male={df['gender'].value_counts().get(0, 0)}, Female={df['gender'].value_counts().get(1, 0)}")
    print(f"\n📅 Age: Mean={df['age'].mean():.1f}, Min={df['age'].min()}, Max={df['age'].max()}")

    print(f"\n💓 Vitals (Mean):")
    print(f"   BP:   {df['bp_systolic'].mean():.0f}/{df['bp_diastolic'].mean():.0f} mmHg")
    print(f"   HR:   {df['heart_rate'].mean():.0f} bpm")
    print(f"   Temp: {df['temperature'].mean():.1f}°F")
    print(f"   SpO2: {df['spo2'].mean():.0f}%")

    print(f"\n🩺 Top 10 Symptoms:")
    symptom_freq = df[symptom_cols].sum().sort_values(ascending=False)
    for col, count in symptom_freq.head(10).items():
        name = col.replace("symptom_", "")
        print(f"   {name:20s}: {int(count):5d} ({count/len(df)*100:.1f}%)")

    print(f"\n📋 Conditions:")
    cond_freq = df[condition_cols].sum().sort_values(ascending=False)
    for col, count in cond_freq.items():
        if count > 0:
            name = col.replace("condition_", "")
            print(f"   {name:20s}: {int(count):5d} ({count/len(df)*100:.1f}%)")

    print(f"\n   Patients with conditions: {df['has_pre_existing'].sum()} ({df['has_pre_existing'].mean()*100:.1f}%)")
    print(f"   Avg symptoms/patient:     {df['num_symptoms'].mean():.1f}")
    print(f"   Avg conditions (if any):  {df[df['has_pre_existing']==1]['num_conditions'].mean():.1f}")

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "patients.csv")
    excel_path = os.path.join(script_dir, "patients.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False, sheet_name="Patient Data")

    print(f"\n✅ Saved:")
    print(f"   CSV:   {csv_path}")
    print(f"   Excel: {excel_path}")
    print(f"   Shape: {df.shape}")
    print(f"\n🚀 Ready for model training!")


if __name__ == "__main__":
    main()