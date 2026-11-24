
import pandas as pd
import pytest

from src.data_loader import CATEGORIES, ID, NUMERIC, TARGET, load_dataset
from src.predict import load_bundle


@pytest.fixture(scope="session")
def sample():
    row = {field: options[0] for field, options in CATEGORIES.items()}
    row.update(dict(zip(NUMERIC, [12, 65.0, 780.0])))
    return row


@pytest.fixture(scope="session")
def trained():
    return load_bundle()


@pytest.fixture
def batch(sample):
    return pd.DataFrame([{ID: "001", **sample}, {ID: "002", **sample}])


@pytest.fixture
def source_csv(tmp_path, batch):
    path = tmp_path / "source.csv"
    batch.assign(**{TARGET: ["Yes", "No"]}).to_csv(path, index=False)
    return path
