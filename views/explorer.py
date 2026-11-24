import plotly.express as px
import streamlit as st

from src.data_loader import CATEGORIES, NUMERIC
from src.ui import COLORS, get_data, group_churn, plot_rates

st.title("Data Explorer")
data = get_data()
a, b, c = st.columns(3)
contracts = a.multiselect("Contract types", CATEGORIES["Contract"], default=CATEGORIES["Contract"])
services = b.multiselect("Internet services", CATEGORIES["InternetService"], default=CATEGORIES["InternetService"])
bounds = (int(data.tenure.min()), int(data.tenure.max()))
tenure = c.slider("Tenure in months", *bounds, value=bounds)
filtered = data[data.Contract.isin(contracts) & data.InternetService.isin(services)
                & data.tenure.between(*tenure)]
if filtered.empty:
    st.info("No customers match these filters.")
    st.stop()
st.caption(f"{len(filtered):,} customers · {filtered.Churn.eq('Yes').mean():.1%} observed churn")
left, right = st.columns(2)
with left:
    numerical = st.selectbox("Numerical feature", NUMERIC)
    histogram = px.histogram(filtered, x=numerical, color="Churn", barmode="overlay",
                             opacity=0.64, histnorm="percent", color_discrete_map=COLORS)
    histogram.update_yaxes(title="Share within churn group (%)")
    st.plotly_chart(histogram, width="stretch")
with right:
    category = st.selectbox("Categorical feature", list(CATEGORIES), index=list(CATEGORIES).index("Contract"))
    plot_rates(group_churn(filtered, category), category)
st.subheader("Customer sample")
st.caption("First 50 matching records, with customer identifiers omitted.")
st.dataframe(filtered.head(50), hide_index=True, width="stretch")
