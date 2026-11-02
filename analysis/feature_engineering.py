import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age", "mileage", "engine", "damage_record", "annual_avg_mileage",
    "mileage_x_age", "premium_brand", "age_category", "mileage_category",
]
CATEGORICAL_FEATURES = [
    "brand", "fuel_type", "transmission", "city",
    "damage_status", "owner_count", "vehicle_segment",
]


def apply_new_features(df):
    df = df.copy()

    df["annual_avg_mileage"] = df["mileage"] / df["age"].clip(lower=1)

    df["mileage_category"] = pd.cut(
        df["mileage"],
        bins=[-1, 50000, 120000, float('inf')],
        labels=['Low', 'Medium', 'High'])

    df["age_category"] = pd.cut(
        df["age"],
        bins=[-1, 3, 7, float('inf')],
        labels=['New', 'Medium', 'Old']
    )

    df["damage_status"] = np.where(df["damage_record"] == 0, "Undamaged", "Damaged")

    premium_brands = ["BMW", "Mercedes"]
    df["premium_brand"] = df['brand'].isin(premium_brands).astype(int)

    df["mileage_x_age"] = df["mileage"] * df["age"]

    # Ordinal features
    age_mapping = {'New': 0, 'Medium': 1, 'Old': 2}
    mileage_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df["age_category"] = df["age_category"].map(age_mapping).astype(int)
    df["mileage_category"] = df["mileage_category"].map(mileage_mapping).astype(int)

    return df


def build_pipeline(model, extra_numeric=()):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES + list(extra_numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(apply_new_features)),
        ("preprocess", preprocessor),
        ("model", model),
    ])
