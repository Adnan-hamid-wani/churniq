import streamlit as st

st.set_page_config(page_title="ChurnIQ", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2.8rem;}
[data-testid="stMetric"] {background: white; border: 1px solid #dbe4ef;
    border-radius: 14px; padding: 1rem; min-width: 0;}
</style>
""", unsafe_allow_html=True)
st.sidebar.title("ChurnIQ")
st.sidebar.caption("Telecom churn analytics")
pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/explorer.py", title="Data Explorer", url_path="explorer"),
    st.Page("views/prediction.py", title="Predict Churn", url_path="prediction"),
    st.Page("views/performance.py", title="Model Performance", url_path="performance"),
    st.Page("views/batch.py", title="Batch Predictions", url_path="batch"),
]
st.navigation(pages).run()
