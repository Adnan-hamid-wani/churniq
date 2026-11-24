import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import get_bundle

st.title("Model Performance")
_, metrics = get_bundle()
score = metrics["selected_test"]
st.subheader(metrics["selected_model"])
st.caption(f"Selected by {metrics['selection_metric']} across {metrics['folds']} stratified folds. "
           f"Held-out test set of {metrics['test_rows']:,} customers, "
           f"with a {metrics['classification_threshold']:.0%} classification cutoff.")
labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "f1": "F1", "roc_auc": "ROC-AUC", "average_precision": "Avg. precision"}
for column, key in zip(st.columns(5), list(labels)[:5]):
    column.metric(labels[key], f"{score[key]:.3f}")
comparison = pd.DataFrame([
    {"Model": entry["model"], "CV ROC-AUC": entry["cv_roc_auc"], "CV fold SD": entry["cv_std"],
     **{label: entry["test"][key] for key, label in labels.items()}}
    for entry in metrics["comparison"]
]).sort_values("CV ROC-AUC", ascending=False)
st.subheader("Model comparison")
st.dataframe(comparison.style.format(precision=3), hide_index=True, width="stretch")
with st.expander("Metrics and tuning"):
    st.markdown(
        "- **Accuracy** measures the fraction of all predictions that are correct.\n"
        "- **Precision** measures how many churn flags were correct.\n"
        "- **Recall** measures how many actual churners were flagged.\n"
        "- **F1** is the harmonic mean of precision and recall.\n"
        "- **ROC-AUC** measures how well scores rank churners above non-churners.\n"
        "- **Average precision** summarizes precision across increases in recall.\n"
        "- **CV fold SD** describes the spread of validation ROC-AUC across folds."
    )
    st.json({entry["model"]: entry["parameters"] for entry in metrics["comparison"]})
roc, pr, matrix = st.columns(3)
with roc:
    st.subheader("ROC curve")
    chart = px.line(score["roc"], x="fpr", y="tpr",
                    labels={"fpr": "False positive rate", "tpr": "True positive rate"})
    chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot"})
    st.plotly_chart(chart, width="stretch")
with pr:
    st.subheader("Precision–recall")
    chart = px.line(score["pr"], x="recall", y="precision",
                    labels={"recall": "Recall", "precision": "Precision"})
    chart.add_hline(y=metrics["test_prevalence"], line_dash="dot")
    st.plotly_chart(chart, width="stretch")
with matrix:
    st.subheader("Confusion matrix")
    chart = px.imshow(score["confusion_matrix"], x=["Stay", "Churn"], y=["Stay", "Churn"],
                      labels={"x": "Predicted", "y": "Observed", "color": "Customers"},
                      color_continuous_scale="Blues", text_auto=True, aspect="auto")
    st.plotly_chart(chart, width="stretch")

st.subheader("Global feature importance")
importance = pd.DataFrame(metrics["permutation_importance"])
chart = px.bar(importance.sort_values("mean"), y="feature", x="mean", error_x="std",
               orientation="h", labels={"feature": "Feature", "mean": "Mean ROC-AUC decrease"},
               height=580)
st.plotly_chart(chart, width="stretch")
st.caption("Each feature was shuffled eight times on the held-out set. Bars show the mean drop "
           "in ROC-AUC and whiskers show repeat standard deviation. Correlated inputs can share "
           "importance, and noisy estimates can be negative. These results describe model reliance "
           "across the test set and do not establish causality.")
