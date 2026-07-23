import argparse
import io
import json
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "best_emissions_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
VERSION_PATH = os.path.join(MODEL_DIR, "model_version.json")


# Maps clean, API-friendly field names -> the exact column names the model
# was trained on (the original dataset uses spaces/parentheses that aren't
# valid Python identifiers, e.g. "Drained organic soils (CO2)").
FEATURE_MAP = {
    "net_forest_conversion": "Net Forest conversion",
    "fires_humid_tropical_forests": "Fires in humid tropical forests",
    "forest_fires": "Forest fires",
    "savanna_fires": "Savanna fires",
    "agrifood_systems_waste_disposal": "Agrifood Systems Waste Disposal",
    "crop_residues": "Crop Residues",
    "manure_applied_to_soils": "Manure applied to Soils",
    "manure_management": "Manure Management",
    "food_retail": "Food Retail",
    "rice_cultivation": "Rice Cultivation",
    "food_transport": "Food Transport",
    "drained_organic_soils_co2": "Drained organic soils (CO2)",
    "pesticides_manufacturing": "Pesticides Manufacturing",
    "food_processing": "Food Processing",
    "on_farm_electricity_use": "On-farm Electricity Use",
    "food_household_consumption": "Food Household Consumption",
}
API_FIELD_ORDER = list(FEATURE_MAP.keys())
TRAINING_COLUMN_ORDER = list(FEATURE_MAP.values())
TARGET_COLUMN = "total_emission"

# FAO `Area` naming (matches the raw Agrofood_co2_emission.csv, not ISO3
# codes) - 53 African countries/areas, including two historical entities
# ("Ethiopia PDR", "Sudan (former)") present in the raw file's early years.
AFRICAN_COUNTRIES = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
    "Congo", "Democratic Republic of the Congo", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Ethiopia PDR",
    "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
    "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
    "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
    "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
    "Sudan (former)", "Togo", "Tunisia", "Uganda",
    "United Republic of Tanzania", "Zambia", "Zimbabwe", "Western Sahara",
]


class EmissionFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "country": "Kenya",
                "year": 2020,
                "net_forest_conversion": 15000.0,
                "fires_humid_tropical_forests": 500.0,
                "forest_fires": 200.0,
                "savanna_fires": 1000.0,
                "agrifood_systems_waste_disposal": 300.0,
                "crop_residues": 50.0,
                "manure_applied_to_soils": 20.0,
                "manure_management": 40.0,
                "food_retail": 100.0,
                "rice_cultivation": 30.0,
                "food_transport": 150.0,
                "drained_organic_soils_co2": 10.0,
                "pesticides_manufacturing": 5.0,
                "food_processing": 200.0,
                "on_farm_electricity_use": 60.0,
                "food_household_consumption": 400.0,
            }
        }
    )

    country: str = Field(..., min_length=2, max_length=56,
                          description="Country name, for reference/logging only.")
    year: int = Field(..., ge=1990, le=2035,
                       description="Calendar year the observation refers to.")

    # Range bounds derived from the actual min/max observed in the cleaned
    # African subset of the training data (1990-2020), with a safety margin
    # added on the upper bound for legitimate future growth. Net Forest
    # conversion and Drained organic soils (CO2) can legitimately go
    # negative in FAO data (e.g. net afforestation, or net-rewetted soils).
    net_forest_conversion: float = Field(
        ..., ge=-50000, le=650000,
        description="Net forest area converted to other land use.")
    fires_humid_tropical_forests: float = Field(
        ..., ge=0, le=40000,
        description="Emissions from fires in humid tropical forests.")
    forest_fires: float = Field(
        ..., ge=0, le=40000,
        description="Emissions from forest fires (all forest types).")
    savanna_fires: float = Field(
        ..., ge=0, le=35000,
        description="Emissions from savanna/grassland fires.")
    agrifood_systems_waste_disposal: float = Field(
        ..., ge=0, le=35000,
        description="Emissions from disposal of agrifood system waste.")
    crop_residues: float = Field(
        ..., ge=0, le=2500,
        description="Emissions from burning/decomposition of crop residues.")
    manure_applied_to_soils: float = Field(
        ..., ge=0, le=1200,
        description="Emissions from manure applied as fertilizer.")
    manure_management: float = Field(
        ..., ge=0, le=3500,
        description="Emissions from livestock manure management/storage.")
    food_retail: float = Field(
        ..., ge=0, le=11000,
        description="Emissions from food retail operations.")
    rice_cultivation: float = Field(
        ..., ge=0, le=19000,
        description="Methane emissions from flooded rice cultivation.")
    food_transport: float = Field(
        ..., ge=0, le=7000,
        description="Emissions from transporting food along the supply chain.")
    drained_organic_soils_co2: float = Field(
        ..., ge=-5000, le=15000,
        description="CO2 emissions from drained organic (peat) soils.")
    pesticides_manufacturing: float = Field(
        ..., ge=0, le=1600,
        description="Emissions from manufacturing pesticides.")
    food_processing: float = Field(
        ..., ge=0, le=21000,
        description="Emissions from industrial food processing.")
    on_farm_electricity_use: float = Field(
        ..., ge=0, le=7000,
        description="Emissions from electricity used on farms.")
    food_household_consumption: float = Field(
        ..., ge=0, le=25000,
        description="Emissions attributable to household-level food consumption.")


class PredictionResponse(BaseModel):
    predicted_total_emission: float
    model_version: str
    inputs: EmissionFeatures


class RetrainResponse(BaseModel):
    message: str
    rows_used: int
    train_r2: float
    test_r2: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelNotLoadedError(RuntimeError):
    pass


def _write_version(version: str) -> None:
    with open(VERSION_PATH, "w") as f:
        json.dump({"version": version}, f)


def _read_version() -> Optional[str]:
    if os.path.exists(VERSION_PATH):
        with open(VERSION_PATH) as f:
            return json.load(f).get("version")
    return None


def load_artifacts():
    """Load model + scaler from disk. Raises ModelNotLoadedError if missing."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise ModelNotLoadedError(
            "Model artifacts not found. Run `python prediction.py` to train "
            "an initial model, or call POST /retrain with a training CSV."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def features_to_dataframe(features: dict) -> pd.DataFrame:
    """Convert a validated EmissionFeatures dict into a 1-row DataFrame with
    columns in the exact order the model/scaler expect."""
    row = {FEATURE_MAP[k]: features[k] for k in API_FIELD_ORDER}
    return pd.DataFrame([row], columns=TRAINING_COLUMN_ORDER)


def predict_one(features: dict) -> float:
    model, scaler = load_artifacts()
    X = features_to_dataframe(features)
    # Scaling is applied unconditionally here because train_and_save()
    # always fits the model on scaled data below - keeping predict/train
    # consistent avoids the classic bug of scaling at only one end.
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def train_and_save(df: pd.DataFrame, n_estimators: int = 200, random_state: int = 42) -> dict:
    if "Area" in df.columns:
        before = len(df)
        df = df[df["Area"].isin(AFRICAN_COUNTRIES)].copy()
        print(f"Filtered uploaded data to African countries: {before} -> {len(df)} rows")

    missing = [c for c in TRAINING_COLUMN_ORDER + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded data is missing required columns: {missing}")

    df = df[TRAINING_COLUMN_ORDER + [TARGET_COLUMN]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} rows remained after filtering to African countries — "
            "need at least 20 rows to retrain reliably."
        )

    X = df[TRAINING_COLUMN_ORDER]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_scaled))
    test_r2 = r2_score(y_test, model.predict(X_test_scaled))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_version(version)

    return {
        "rows_used": int(len(df)),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "model_version": version,
    }


def load_real_data(csv_path: str) -> pd.DataFrame:
    """Reproduces the Task 1 cleaning steps for the real dataset
    (Agrofood_co2_emission.csv)."""
    df = pd.read_csv(csv_path)
    df = df[df["Area"].isin(AFRICAN_COUNTRIES)].copy()
    df = df.drop(columns=["On-farm energy use"], errors="ignore")
    for col in df.columns:
        if col != "Area":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    # Drop zero-variance columns (e.g. "Fires in organic soils" is 0 for
    # every African row) - can't help predict anything.
    constant_cols = [c for c in df.columns if c != "Area" and df[c].nunique() == 1]
    df = df.drop(columns=constant_cols)
    return df


def make_synthetic_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Roughly mirrors the real feature set's shape/scale, purely so the API
    can be smoke-tested with no external dataset available. Net forest
    conversion is made the dominant driver, matching the ~0.96 correlation
    with total_emission observed in the real African data."""
    rng = np.random.default_rng(seed)

    net_forest = np.clip(rng.gamma(shape=1.5, scale=15000, size=n_rows), 0, 600000)
    fires_tropical = np.clip(net_forest * rng.uniform(0.01, 0.06, n_rows), 0, 40000)
    forest_fires = np.clip(net_forest * rng.uniform(0.01, 0.06, n_rows), 0, 40000)
    savanna_fires = np.clip(rng.gamma(1.5, 1500, n_rows), 0, 35000)
    waste_disposal = np.clip(rng.gamma(2, 500, n_rows), 0, 35000)
    crop_residues = np.clip(rng.gamma(2, 50, n_rows), 0, 2500)
    manure_soils = np.clip(rng.gamma(2, 30, n_rows), 0, 1200)
    manure_mgmt = np.clip(rng.gamma(2, 60, n_rows), 0, 3500)
    food_retail = np.clip(rng.gamma(2, 200, n_rows), 0, 11000)
    rice = np.clip(rng.gamma(1.2, 300, n_rows), 0, 19000)
    transport = np.clip(rng.gamma(2, 100, n_rows), 0, 7000)
    drained_soils = np.clip(rng.normal(500, 800, n_rows), -5000, 15000)
    pesticides = np.clip(rng.gamma(1.5, 50, n_rows), 0, 1600)
    processing = np.clip(rng.gamma(2, 400, n_rows), 0, 21000)
    on_farm_elec = np.clip(rng.gamma(2, 150, n_rows), 0, 7000)
    household = np.clip(rng.gamma(2, 500, n_rows), 0, 25000)

    target = np.clip(
        net_forest * 1.05 + fires_tropical * 0.8 + forest_fires * 0.7
        + savanna_fires * 0.4 + waste_disposal * 0.3 + rng.normal(0, 2000, n_rows),
        0, None,
    )

    return pd.DataFrame({
        TRAINING_COLUMN_ORDER[0]: net_forest,
        TRAINING_COLUMN_ORDER[1]: fires_tropical,
        TRAINING_COLUMN_ORDER[2]: forest_fires,
        TRAINING_COLUMN_ORDER[3]: savanna_fires,
        TRAINING_COLUMN_ORDER[4]: waste_disposal,
        TRAINING_COLUMN_ORDER[5]: crop_residues,
        TRAINING_COLUMN_ORDER[6]: manure_soils,
        TRAINING_COLUMN_ORDER[7]: manure_mgmt,
        TRAINING_COLUMN_ORDER[8]: food_retail,
        TRAINING_COLUMN_ORDER[9]: rice,
        TRAINING_COLUMN_ORDER[10]: transport,
        TRAINING_COLUMN_ORDER[11]: drained_soils,
        TRAINING_COLUMN_ORDER[12]: pesticides,
        TRAINING_COLUMN_ORDER[13]: processing,
        TRAINING_COLUMN_ORDER[14]: on_farm_elec,
        TRAINING_COLUMN_ORDER[15]: household,
        TARGET_COLUMN: target,
    })


app = FastAPI(
    title="African Agrifood CO2 Emissions Prediction API",
    description=(
        "Predicts total agrifood-system CO2 emissions (total_emission) for "
        "African countries from land-use, fire, and agricultural-process "
        "indicators, using a Random Forest model trained in Task 1."
    ),
    version="1.0.0",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect the bare root URL to the interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    model_loaded = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
    return HealthResponse(status="ok", model_loaded=model_loaded, model_version=_read_version())


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_total_emission(features: EmissionFeatures):
    """Predict total agrifood-system CO2 emissions for a single
    country-year observation."""
    try:
        prediction = predict_one(features.model_dump())
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        predicted_total_emission=prediction,
        model_version=_read_version() or "unknown",
        inputs=features,
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["Retraining"])
async def retrain_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        raw_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 rows to retrain reliably, got {len(df)}.",
        )

    try:
        metrics = train_and_save(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return RetrainResponse(
        message="Model retrained and saved successfully.",
        rows_used=metrics["rows_used"],
        train_r2=metrics["train_r2"],
        test_r2=metrics["test_r2"],
        model_version=metrics["model_version"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/build the total_emission model.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to Agrofood_co2_emission.csv. "
             "If omitted, synthetic demo data is used instead.",
    )
    args = parser.parse_args()

    if args.data:
        print(f"Loading real data from {args.data} ...")
        training_df = load_real_data(args.data)
    else:
        print("No --data provided; generating synthetic demo data instead.")
        print("Replace this with a real training run before deploying for real use.")
        training_df = make_synthetic_data()

    result = train_and_save(training_df)
    print("Training complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
