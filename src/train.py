"""Rebuild the deployment artifacts with python -m src.train."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import DATA_PATH, FEATURES, NUMERIC, ROOT, TARGET, load_dataset
from src.evaluate import HIGH_RISK, LOW_RISK, THRESHOLD, evaluate
from src.preprocessing import make_pipeline

SEED = 42


def split_data(data):
    return train_test_split(
        data[FEATURES], data[TARGET].eq("Yes").astype(int),
        test_size=0.2, random_state=SEED, stratify=data[TARGET],
    )


def train(data_path: Path, output: Path) -> dict:
    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_data(data)
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000),
         {"model__C": [0.05, 0.2, 1, 5]}, True),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED),
         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [10, 30]}, False),
        ("Random Forest", RandomForestClassifier(n_estimators=120, random_state=SEED),
         {"model__max_depth": [6, 12, None], "model__min_samples_leaf": [3, 12]}, False),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    searches = {}
    for name, estimator, grid, scale in candidates:
        search = GridSearchCV(make_pipeline(estimator, scale=scale), grid,
                              cv=cv, scoring="roc_auc", n_jobs=2, error_score="raise")
        searches[name] = search.fit(x_train, y_train)
    selected = max(searches, key=lambda name: searches[name].best_score_)
    model = searches[selected].best_estimator_
    comparison = []
    for name, search in searches.items():
        scores = evaluate(y_test, search.predict_proba(x_test)[:, 1])
        comparison.append({
            "model": name, "cv_roc_auc": search.best_score_,
            "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "parameters": {key.removeprefix("model__"): value
                           for key, value in search.best_params_.items()},
            "test": {key: value for key, value in scores.items() if isinstance(value, float)},
        })
    importance = permutation_importance(model, x_test, y_test, scoring="roc_auc",
                                        n_repeats=8, random_state=SEED, n_jobs=2)
    reference = {
        field: float(x_train[field].median()) if field in NUMERIC else x_train[field].mode()[0]
        for field in FEATURES
    }
    results = {
        "selected_model": selected, "selection_metric": "mean cross-validated ROC-AUC",
        "sklearn_version": sklearn.__version__, "seed": SEED, "folds": 5,
        "data_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "train_rows": len(x_train), "test_rows": len(x_test),
        "test_prevalence": float(y_test.mean()), "classification_threshold": THRESHOLD,
        "risk_cutoffs": {"medium": LOW_RISK, "high": HIGH_RISK},
        "comparison": comparison, "selected_test": evaluate(y_test, model.predict_proba(x_test)[:, 1]),
        "permutation_importance": sorted(
            [{"feature": f, "mean": float(m), "std": float(s)}
             for f, m, s in zip(FEATURES, importance.importances_mean, importance.importances_std)],
            key=lambda item: item["mean"], reverse=True),
        "training_reference": reference,
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "churn_pipeline.joblib", compress=3)
    results["pipeline_sha256"] = sha256((output / "churn_pipeline.joblib").read_bytes()).hexdigest()
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    result = train(args.data, args.output)
    print("Selected", result["selected_model"])
    for candidate in result["comparison"]:
        print(candidate["model"], candidate["cv_roc_auc"], candidate["test"])
