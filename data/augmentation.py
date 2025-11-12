import pandas as pd
import os
import json
from openai import OpenAI
import numpy as np

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

def augment_features(df: pd.DataFrame, num_aug=None) -> pd.DataFrame:
    """
    Vectorized feature augmentation for the UCI Heart Disease dataset.
    Converts categorical columns to numeric encodings and adds derived features.
    Works efficiently on entire DataFrames.
    """

    if num_aug is None:
        df = df.copy()
    else:
        df = df.head(num_aug).copy()

    # --- Normalize text values to lowercase strings ---
    def norm(series):
        return series.astype(str).str.strip().str.lower()

    # --- Category mappings (from your schema) ---
    sex_map = {'male': 1, 'female': 0}
    dataset_map = {'cleveland': 0, 'hungary': 1, 'switzerland': 2, 'va long beach': 3}
    cp_map = {
        'typical angina': 1,
        'atypical angina': 2,
        'non-anginal pain': 3,
        'asymptomatic': 4
    }
    restecg_map = {
        'normal': 0,
        'st-t abnormality': 1,
        'lv hypertrophy': 2
    }
    slope_map = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
    thal_map = {'normal': 3, 'fixed defect': 6, 'reversable defect': 7}

    # --- Encode categorical columns ---
    df['sex_encoded'] = norm(df['sex']).map(sex_map)
    df['dataset_encoded'] = norm(df['dataset']).map(dataset_map)
    df['cp_encoded'] = norm(df['cp']).map(cp_map)
    df['restecg_encoded'] = norm(df['restecg']).map(restecg_map)
    df['slope_encoded'] = norm(df['slope']).map(slope_map)
    df['thal_encoded'] = norm(df['thal']).map(thal_map)

    # --- Boolean normalization (fbs, exang) ---
    def to_int_bool(col):
        return (
            col.astype(str)
               .str.strip()
               .str.lower()
               .replace({'true': 1, 'false': 0, 'yes': 1, 'no': 0})
               .astype(float)
        )

    df['fbs'] = to_int_bool(df['fbs'])
    df['exang'] = to_int_bool(df['exang'])

    # --- Numeric columns ---
    for col in ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Derived features (vectorized math) ---
    df['chol_age_ratio'] = df['chol'] / df['age']
    df['fbs_trestbps_product'] = df['fbs'] * df['trestbps']
    df['bp_chol_interaction'] = df['trestbps'] * df['chol']
    df['heart_rate_reserve'] = df['thalch'] / df['age']
    df['stress_slope_index'] = (df['slope_encoded'] + 1) * (df['thalch'] - df['oldpeak'])
    df['exercise_impact'] = df['thalch'] * (1 - df['exang'])

    df['is_hypertensive'] = (df['trestbps'] >= 140).astype(float)
    df['is_hypercholesterolemia'] = (df['chol'] >= 240).astype(float)

    df['risk_factor_count'] = (
        df[['is_hypertensive', 'fbs', 'exang']].fillna(0).sum(axis=1)
    )

    df['sex_cp_interaction'] = df['sex_encoded'] * df['cp_encoded']
    df['restecg_abnormal'] = (df['restecg_encoded'] != 0).astype(float)

    df['log_chol'] = np.log1p(df['chol'].clip(lower=0))
    df['log_trestbps'] = np.log1p(df['trestbps'].clip(lower=0))
    df['oldpeak_sq'] = df['oldpeak'] ** 2
    df['cardiac_stress_index'] = (df['oldpeak'] * (1 + df['exang'])) / (df['thalch'] + 1)
    df['metabolic_risk_index'] = (df['chol'] * df['fbs'] * df['trestbps']) / df['age']
    df['oxygen_efficiency_index'] = df['thalch'] / df['trestbps']
    df['cp_slope_interaction'] = df['cp_encoded'] * df['slope_encoded']

    df['risk_score_raw'] = (
        0.4 * df['is_hypertensive'].fillna(0)
        + 0.3 * df['fbs'].fillna(0)
        + 0.3 * df['exang'].fillna(0)
        + 0.2 * df['restecg_abnormal'].fillna(0)
    )

    return df
