"""Model-agnostic, one-feature-at-a-time prediction sensitivity."""
import numpy as np
import pandas as pd


def local_sensitivity(model, row: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Compare a single prediction with each field set to its training baseline."""
    if len(row) != 1:
        raise ValueError("Local sensitivity requires exactly one customer.")
    fields = list(reference)
    alternatives = pd.concat([row] * len(fields), ignore_index=True)
    for index, field in enumerate(fields):
        alternatives.at[index, field] = reference[field]
    original = model.predict_proba(row)[0, 1]
    changes = original - model.predict_proba(alternatives)[:, 1]
    return (
        pd.DataFrame({"Feature": fields, "Training reference": [reference[f] for f in fields],
                      "Difference (pp)": changes * 100})
        .iloc[np.argsort(-np.abs(changes), kind="stable")]
        .reset_index(drop=True)
    )
