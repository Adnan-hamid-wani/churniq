import pandas as pd
import pytest

from src.data_loader import ID, load_dataset, validate_features
from src.predict import read_batch


def test_schema_and_identifiers(source_csv):
    assert load_dataset(source_csv)[ID].tolist() == ["001", "002"]
    original = pd.read_csv(source_csv)
    original.drop(columns="Contract").to_csv(source_csv, index=False)
    with pytest.raises(ValueError, match="schema"):
        load_dataset(source_csv)


def test_duplicate_source_ids(source_csv):
    frame = pd.read_csv(source_csv).assign(customerID="same")
    frame.to_csv(source_csv, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(source_csv)


@pytest.mark.parametrize("value", ["garbage", "-1", "inf", "1.5"])
def test_invalid_tenure(batch, value):
    batch["tenure"] = value
    with pytest.raises(ValueError, match="tenure"):
        validate_features(batch)


@pytest.mark.parametrize("payload, message", [(b"id,id\n1,2", "unique"),
                                               (b"id,tenure\n1", "wrong number")])
def test_bad_csv(payload, message):
    with pytest.raises(ValueError, match=message):
        read_batch(payload)


def test_missing_values(batch):
    batch["TotalCharges"] = "  "
    batch["Contract"] = ""
    clean = validate_features(batch)
    assert clean[["TotalCharges", "Contract"]].isna().all().all()
