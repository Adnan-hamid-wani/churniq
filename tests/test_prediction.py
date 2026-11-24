import json
import shutil

import numpy as np
import pytest

from src.data_loader import FEATURES, ROOT
from src.explain import local_sensitivity
from src.predict import load_bundle, predict_batch, read_batch


def test_saved_model_and_batch(trained, batch):
    model, metrics = trained
    batch["Contract"] = ["", "Unknown contract"]
    batch["TotalCharges"] = ["", "780"]
    batch["extra"] = "excluded"
    uploaded = read_batch(batch.to_csv(index=False).encode())
    results = predict_batch(model, uploaded, metrics["classification_threshold"])
    assert list(model.feature_names_in_) == FEATURES
    assert list(results) == ["customerID", "predicted_class", "predicted_probability"]
    assert list(results.customerID) == ["001", "002"]
    assert results.predicted_probability.between(0, 1).all()
    expected = np.where(results.predicted_probability >= .5, "Churn", "Stay")
    assert (results.predicted_class == expected).all()


def test_batch_requires_ids_and_features(trained, batch):
    model, _ = trained
    for name in ["customerID", "Contract"]:
        with pytest.raises(ValueError, match=name):
            predict_batch(model, batch.drop(columns=name), .5)
    with pytest.raises(ValueError, match="unique"):
        predict_batch(model, batch.assign(customerID="same"), .5)
    with pytest.raises(ValueError, match="unsafe"):
        predict_batch(model, batch.assign(customerID=["=1+1", "safe"]), .5)


def test_mismatched_bundle_fails_before_load(tmp_path):
    shutil.copy(ROOT / "models/churn_pipeline.joblib", tmp_path)
    metrics = json.loads((ROOT / "models/metrics.json").read_text())
    metrics["pipeline_sha256"] = "invalid"
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="do not match"):
        load_bundle(tmp_path)


def test_local_sensitivity_recomputes_actual_probability(trained, batch):
    from src.data_loader import validate_features

    model, metrics = trained
    clean = validate_features(batch.iloc[[0]])
    reference = metrics["training_reference"]
    result = local_sensitivity(model, clean, reference).iloc[0]
    alternate = clean.assign(**{result["Feature"]: reference[result["Feature"]]})
    delta = model.predict_proba(clean)[0, 1] - model.predict_proba(alternate)[0, 1]
    assert result["Difference (pp)"] == pytest.approx(delta * 100)
