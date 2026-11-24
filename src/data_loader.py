"""Schema and source metadata for the IBM Telco sample."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/Telco-Customer-Churn.csv"
SOURCE = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
          "master/data/Telco-Customer-Churn.csv")
SOURCE_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
ID = "customerID"
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ("No", "Yes")
CATEGORIES = {
    "gender": ("Female", "Male"),
    "SeniorCitizen": ("0", "1"),
    "Partner": BINARY,
    "Dependents": BINARY,
    "PhoneService": BINARY,
    "MultipleLines": (*BINARY, "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **dict.fromkeys(
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
         "StreamingTV", "StreamingMovies"],
        (*BINARY, "No internet service"),
    ),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": BINARY,
    "PaymentMethod": ("Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"),
}
FEATURES = NUMERIC + list(CATEGORIES)


def validate_features(frame: pd.DataFrame, *, strict_categories: bool = False) -> pd.DataFrame:
    if frame.columns.has_duplicates:
        raise ValueError("Column names must be unique.")
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The file must contain at least one customer row.")
    clean = pd.DataFrame(index=frame.index)
    for name in NUMERIC:
        values = frame[name].astype("string").str.strip().replace("", pd.NA)
        numbers = pd.to_numeric(values, errors="coerce").astype(float)
        invalid = values.notna() & (~np.isfinite(numbers) | (numbers < 0))
        if name == "tenure":
            invalid |= numbers.notna() & (numbers % 1 != 0)
        if invalid.any():
            rows = ", ".join(map(str, np.flatnonzero(invalid.to_numpy())[:5] + 2))
            raise ValueError(f"{name} must contain nonnegative finite "
                             f"{'whole numbers' if name == 'tenure' else 'numbers'} or blanks (CSV lines {rows}).")
        clean[name] = numbers
    for name, categories in CATEGORIES.items():
        values = frame[name].astype("string").str.strip().replace("", pd.NA)
        if strict_categories and (values.notna() & ~values.isin(categories)).any():
            raise ValueError(f"{name} contains an unsupported source category.")
        clean[name] = values.astype(object).where(values.notna(), np.nan)
    return clean


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    if set(raw.columns) != {*FEATURES, ID, TARGET}:
        raise ValueError("Source CSV columns differ from the IBM Telco schema.")
    clean = validate_features(raw, strict_categories=True)
    ids = raw[ID].str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Source customer IDs must be present and unique.")
    if not raw[TARGET].isin(BINARY).all() or raw[TARGET].nunique() != 2:
        raise ValueError("Source labels must contain both Yes and No.")
    return clean.assign(**{ID: ids, TARGET: raw[TARGET]})


def download_bytes() -> bytes:
    from hashlib import sha256
    from urllib.request import urlopen

    with urlopen(SOURCE, timeout=25) as response:
        payload = response.read(2_000_000)
    if sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("IBM source checksum differs from the documented snapshot.")
    return payload
