"""Positive class is churn (1)."""
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

THRESHOLD = 0.50
LOW_RISK = 0.30
HIGH_RISK = 0.70


def evaluate(y, probabilities) -> dict:
    labels = probabilities >= THRESHOLD
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return {
        "accuracy": accuracy_score(y, labels),
        "precision": precision_score(y, labels, zero_division=0),
        "recall": recall_score(y, labels, zero_division=0),
        "f1": f1_score(y, labels, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "confusion_matrix": confusion_matrix(y, labels, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"recall": recall.tolist(), "precision": precision.tolist()},
    }
