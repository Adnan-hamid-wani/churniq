import plotly.express as px
import streamlit as st

from src.ui import COLORS, get_data, group_churn, plot_rates

st.title("Customer churn overview")
st.caption("IBM Telco Customer Churn sample")
data = get_data()
customers, churn, charges, tenure = st.columns(4)
customers.metric("Total customers", f"{len(data):,}")
churn.metric("Overall churn rate", f"{data.Churn.eq('Yes').mean():.1%}")
charges.metric("Avg. monthly charges", f"${data.MonthlyCharges.mean():,.2f}")
tenure.metric("Avg. tenure", f"{data.tenure.mean():.1f} months")

distribution, contracts = st.columns(2)
with distribution:
    st.subheader("Churn distribution")
    counts = data.Churn.value_counts().rename_axis("Churn").reset_index(name="Customers")
    chart = px.pie(counts, values="Customers", names="Churn", hole=0.55,
                   color="Churn", color_discrete_map=COLORS)
    st.plotly_chart(chart, width="stretch")
with contracts:
    st.subheader("Churn by contract")
    rates = group_churn(data, "Contract")
    plot_rates(rates, "Contract")
highest = rates.loc[rates.rate.idxmax()]
lowest = rates.loc[rates.rate.idxmin()]
st.subheader("Key observations")
st.markdown(
    f"- **{highest.Contract}** customers have the highest observed churn rate ({highest.rate:.1%}).\n"
    f"- **{lowest.Contract}** customers have the lowest observed churn rate ({lowest.rate:.1%}).\n"
    f"- Median tenure is **{data.loc[data.Churn.eq('Yes'), 'tenure'].median():.0f} months** among "
    f"customers who churned and **{data.loc[data.Churn.eq('No'), 'tenure'].median():.0f} months** "
    "among customers who stayed."
)
