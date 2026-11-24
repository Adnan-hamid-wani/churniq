"""Trusted artifact loading and shared inference for both prediction screens."""
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data_loader import CATEGORIES, ID, ROOT, validate_features

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 20_000


def load_bundle(path: Path = ROOT / "models"):
    metrics = json.loads((path / "metrics.json").read_text())
    if metrics["sklearn_version"] != sklearn.__version__:
        raise ValueError("Install the pinned scikit-learn version or rebuild both model artifacts.")
    payload = (path / "churn_pipeline.joblib").read_bytes()
    if sha256(payload).hexdigest() != metrics["pipeline_sha256"]:
        raise ValueError("Model and metrics do not match. Restore or regenerate both artifacts.")
    model = joblib.load(BytesIO(payload))
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected model classes.")
    return model, metrics


def read_batch(payload: bytes) -> pd.DataFrame:
    if len(payload) > MAX_BYTES:
        raise ValueError("Upload a CSV of at most 5 MiB.")
    text = payload.decode("utf-8-sig")
    reader = csv.reader(StringIO(text), strict=True)
    header = next(reader, [])
    if not header or len(set(header)) != len(header):
        raise ValueError("Provide a CSV header with unique column names.")
    rows = []
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"CSV record ending at line {reader.line_num} has the wrong number of fields.")
        rows.append(row)
        if len(rows) > MAX_ROWS:
            raise ValueError(f"Upload at most {MAX_ROWS:,} records.")
    return pd.DataFrame(rows, columns=header)


def predict_batch(model, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    clean = validate_features(frame)
    if ID not in frame:
        raise ValueError(f"Missing required column: {ID}")
    ids = frame[ID].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Provide a unique, nonempty customerID for each row.")
    if ids.str.match(r"^[=+@\-\t\r\n]").any():
        raise ValueError("A customerID starts with a character that is unsafe in spreadsheet exports.")
    probabilities = model.predict_proba(clean)[:, 1]
    return pd.DataFrame({
        ID: ids,
        "predicted_class": np.where(probabilities >= threshold, "Churn", "Stay"),
        "predicted_probability": probabilities,
    })


def unseen_fields(clean: pd.DataFrame) -> list[str]:
    return [field for field, values in CATEGORIES.items()
            if (clean[field].notna() & ~clean[field].isin(values)).any()]
