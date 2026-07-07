import streamlit as st
import duckdb

st.set_page_config(page_title="Fleet PdM", layout="wide")

@st.cache_resource
def get_con():
    return duckdb.connect("scania.duckdb", read_only=True)

@st.cache_data
def fleet_table():
    return get_con().execute("""
        SELECT vehicle_id, pred_class, risk_score, rul_estimate,
               conf_low, conf_high, recommended_action
        FROM predictions ORDER BY risk_score DESC
    """).df()

st.title("Fleet Predictive Maintenance")

df = fleet_table()

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trucks Monitored", len(df))
c2.metric("Flagged (Failure Predicted)", int(df["pred_class"].sum()))
c3.metric("Highest Risk Score", f"{df['risk_score'].max():.4f}")
c4.metric("Avg RUL (days)", int(df["rul_estimate"].mean()))

# Risk filter
st.subheader("Fleet Overview")
min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)
filtered = df[df["risk_score"] >= min_risk]
st.write(f"Showing {len(filtered)} of {len(df)} vehicles")
st.dataframe(filtered, use_container_width=True, height=400)