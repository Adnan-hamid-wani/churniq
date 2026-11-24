import csv
from hashlib import sha256

import pandas as pd
import streamlit as st

from src.data_loader import FEATURES, ID, validate_features
from src.predict import predict_batch, read_batch, unseen_fields
from src.ui import get_bundle

st.title("Batch Predictions")
st.caption("Score up to 20,000 customer records from a UTF-8 CSV, with a 5 MiB file limit.")
model, metrics = get_bundle()
st.download_button("Download CSV header template",
                   pd.DataFrame(columns=[ID, *FEATURES]).to_csv(index=False),
                   "churniq_template.csv", mime="text/csv")
st.caption("Provide a unique customerID per row. Blank feature cells are imputed using training statistics. "
           "Unrecognized categories are encoded as all zeroes for that field.")
uploaded = st.file_uploader("Customer records", type="csv")
if uploaded is None:
    st.session_state.pop("batch_result", None)
    st.stop()
payload = uploaded.getvalue()
fingerprint = sha256(payload).hexdigest()
cached = st.session_state.get("batch_result")
if cached and cached[0] != fingerprint:
    st.session_state.pop("batch_result")
if st.button("Generate predictions", type="primary"):
    st.session_state.pop("batch_result", None)
    try:
        frame = read_batch(payload)
        with st.spinner("Scoring customers…"):
            result = predict_batch(model, frame, metrics["classification_threshold"])
        st.session_state["batch_result"] = (fingerprint, result, unseen_fields(validate_features(frame)))
    except (ValueError, csv.Error) as error:
        st.error(str(error))
if "batch_result" in st.session_state:
    _, result, unseen = st.session_state["batch_result"]
    if unseen:
        st.warning("Check unfamiliar categories in " + ", ".join(unseen) + ".")
    st.success(f"Scored {len(result):,} customers.")
    st.dataframe(result.head(50), hide_index=True, width="stretch",
                 column_config={"predicted_probability": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download predictions", result.to_csv(index=False), "churniq_predictions.csv",
                       mime="text/csv", on_click="ignore")
