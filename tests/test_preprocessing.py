import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import FEATURES, TARGET, validate_features
from src.preprocessing import make_pipeline
from src.train import split_data


def test_fold_preprocessing_and_unseen_category(batch):
    batch["tenure"] = [0, 12]
    features = validate_features(batch)
    pipeline = make_pipeline(LogisticRegression(), scale=True).fit(features, [0, 1])
    numeric = pipeline["features"].named_transformers_["numeric"]
    assert numeric["impute"].statistics_[0] == 6
    assert numeric["scale"].mean_[0] == 6
    sample = batch.assign(Contract="New contract", TotalCharges="")
    probabilities = pipeline.predict_proba(validate_features(sample))
    assert probabilities.shape == (2, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert list(pipeline.feature_names_in_) == FEATURES


def test_reproducible_stratified_split(batch):
    frame = pd.concat([batch] * 10, ignore_index=True)
    frame[TARGET] = ["Yes", "No"] * 10
    a, b, y_train, y_test = split_data(frame)
    assert a.index.equals(split_data(frame)[0].index)
    assert not a.index.intersection(b.index).size
    assert y_train.mean() == y_test.mean() == .5
