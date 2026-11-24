from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import CATEGORIES, NUMERIC


def make_pipeline(model, *, scale: bool = False) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, list(CATEGORIES)),
    ])
    return Pipeline([("features", features), ("model", model)])
