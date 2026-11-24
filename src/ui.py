from io import BytesIO

import plotly.express as px
import streamlit as st

from src.data_loader import DATA_PATH, ID, download_bytes, load_dataset
from src.predict import load_bundle

COLORS = {"No": "#2057A0", "Yes": "#A34823"}


@st.cache_data(show_spinner="Loading the IBM sample…")
def dashboard_data():
    source = DATA_PATH if DATA_PATH.is_file() else BytesIO(download_bytes())
    return load_dataset(source).drop(columns=ID)


@st.cache_resource
def artifacts():
    return load_bundle()


def get_data():
    try:
        return dashboard_data()
    except (OSError, ValueError) as error:
        st.error(f"Cannot load the IBM sample. Run python -m scripts.download_data and retry. {error}")
        st.stop()


def get_bundle():
    try:
        return artifacts()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Cannot load the trained model. Restore the release artifacts or run python -m src.train. {error}")
        st.stop()


def group_churn(frame, field):
    rates = (frame.assign(churned=frame["Churn"].eq("Yes"))
             .groupby(field, dropna=False)["churned"]
             .agg(rate="mean", customers="size").reset_index())
    rates[field] = rates[field].fillna("Missing")
    return rates


def plot_rates(rates, field):
    chart = px.bar(rates, x=field, y="rate", hover_data=["customers"],
                   labels={"rate": "Churn rate"}, color_discrete_sequence=[COLORS["No"]])
    chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(chart, width="stretch")
