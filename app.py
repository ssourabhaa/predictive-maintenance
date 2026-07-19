import streamlit as st
import duckdb

st.set_page_config(page_title="SCANIA PdM", layout="wide")

@st.cache_resource
def get_con():
    return duckdb.connect("scania.duckdb", read_only=True)

st.title("SCANIA Predictive Maintenance")
st.markdown("Physics-informed ML with conformal uncertainty and cost-optimal maintenance decisions.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Page 1 — Model Performance")
    st.write("Training and validation metrics, confusion matrix, cost analysis, and retrospective RUL analysis.")
    st.page_link("pages/1_Model_Performance.py", label="Open Page 1", icon="📊")

with col2:
    st.subheader("📈 Page 2 — Validation Forecasts")
    st.write("Per-vehicle risk trajectories with conformal intervals and Bayes cost-based decisions.")
    st.page_link("pages/2_Validation_Forecasts.py", label="Open Page 2", icon="📈")

st.divider()
con = get_con()
n_val = con.execute("SELECT n_validation FROM metrics").fetchone()[0]
n_train = con.execute("SELECT count(*) FROM predictions").fetchone()[0] - n_val
st.caption(f"Training vehicles: {n_train:,} · Validation vehicles: {n_val:,}")