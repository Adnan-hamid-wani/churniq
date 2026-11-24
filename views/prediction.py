import pandas as pd
import streamlit as st

from src.data_loader import CATEGORIES, validate_features
from src.explain import local_sensitivity
from src.ui import get_bundle

st.title("Predict Churn")
st.caption("Estimate churn risk for one telecom customer.")
model, metrics = get_bundle()
threshold = metrics["classification_threshold"]
low = metrics["risk_cutoffs"]["medium"]
high = metrics["risk_cutoffs"]["high"]
with st.form("customer"):
    billing, services, profile = st.columns(3)
    row = {}
    for column, title, fields in [
        (billing, "Account and billing", ["tenure", "MonthlyCharges", "TotalCharges",
                                          "Contract", "PaperlessBilling", "PaymentMethod"]),
        (services, "Phone and internet", ["PhoneService", "MultipleLines", "InternetService",
                                          "OnlineSecurity", "OnlineBackup", "TechSupport"]),
        (profile, "Customer and add-ons", ["gender", "SeniorCitizen", "Partner", "Dependents",
                                          "DeviceProtection", "StreamingTV", "StreamingMovies"]),
    ]:
        with column:
            st.subheader(title)
            for field in fields:
                label = {
                    "tenure": "Tenure in months", "MonthlyCharges": "Monthly charges",
                    "TotalCharges": "Total charges", "gender": "Gender",
                    "SeniorCitizen": "Senior citizen", "PhoneService": "Phone service",
                    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
                    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
                    "TechSupport": "Tech support", "DeviceProtection": "Device protection",
                    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
                    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
                }.get(field, field)
                if field in CATEGORIES:
                    options = CATEGORIES[field]
                    shown = {"0": "No", "1": "Yes"} if field == "SeniorCitizen" else {}
                    row[field] = st.selectbox(label, options, index=int(field == "PhoneService"), format_func=lambda v, names=shown: names.get(v, v))
                elif field == "tenure":
                    row[field] = st.number_input(label, min_value=0, value=12, step=1)
                else:
                    row[field] = st.number_input(label, min_value=0.0,
                                                 value=65.0 if field == "MonthlyCharges" else 780.0)
    submitted = st.form_submit_button("Estimate churn risk", type="primary")

if submitted:
    internet, phone = row["InternetService"], row["PhoneService"]
    internet_addons = ["OnlineSecurity", "OnlineBackup", "TechSupport", "DeviceProtection",
                       "StreamingTV", "StreamingMovies"]
    invalid_internet = any((row[field] == "No internet service") != (internet == "No")
                           for field in internet_addons)
    if invalid_internet or ((row["MultipleLines"] == "No phone service") != (phone == "No")):
        st.session_state.pop("scored_customer", None)
        st.error("Match add-on selections to the selected internet and phone services.")
    else:
        st.session_state["scored_customer"] = row
if "scored_customer" in st.session_state:
    clean = validate_features(pd.DataFrame([st.session_state["scored_customer"]]))
    probability = model.predict_proba(clean)[0, 1]
    risk = "High" if probability >= high else "Medium" if probability >= low else "Low"
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{probability:.1%}")
    b.metric("Predicted class", "Churn" if probability >= threshold else "Stay")
    c.metric("Risk category", risk)
    st.caption(f"Class cutoff {threshold:.0%}. Low below {low:.0%}, Medium {low:.0%} to below "
               f"{high:.0%}, High {high:.0%} and above. This statistical estimate carries uncertainty.")
    st.subheader("Which inputs moved this estimate?")
    factors = local_sensitivity(model, clean, metrics["training_reference"])
    factors = factors[factors["Difference (pp)"].abs() >= 0.005].head(5).copy()
    if factors.empty:
        st.info("No input moved the estimate by a measurable amount in this comparison.")
    else:
        top = factors.iloc[0]
        st.write(f"Compared with the training reference for **{top['Feature']}**, the entered value "
                 f"{'raised' if top['Difference (pp)'] > 0 else 'lowered'} the estimate by "
                 f"**{abs(top['Difference (pp)']):.1f} percentage points** when all other values stayed fixed.")
        factors["Training reference"] = factors["Training reference"].astype(str)
        st.dataframe(factors, hide_index=True, width="stretch",
                     column_config={"Difference (pp)": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Each comparison replaces one field with its training median or mode. "
               "Correlated fields may yield unrealistic configurations. Differences are model sensitivities, "
               "can overlap, and do not establish cause and effect.")
