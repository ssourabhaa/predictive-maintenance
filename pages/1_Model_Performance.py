import streamlit as st
import duckdb
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Model Performance", layout="wide")

@st.cache_resource
def get_con():
    return duckdb.connect("scania.duckdb", read_only=True)

@st.cache_data
def load_metrics():
    return get_con().execute("SELECT * FROM metrics").df().iloc[0]

@st.cache_data
def load_cm():
    return get_con().execute("SELECT * FROM confusion_matrix").df()

@st.cache_data
def load_retro():
    return get_con().execute("SELECT * FROM retrospective").df()

@st.cache_data
def load_importances():
    return get_con().execute("SELECT * FROM feature_importances ORDER BY importance DESC LIMIT 20").df()

st.title("Model Performance & Retrospective Analysis")
st.caption("How well does the LightGBM model predict failures on the SCANIA validation set?")

m = load_metrics()

# ─── Classification metrics ───
st.subheader("Classification Metrics (Validation Set)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("Precision", f"{m['precision']:.3f}")
c3.metric("Recall", f"{m['recall']:.3f}")
c4.metric("F1 Score", f"{m['f1']:.3f}")
c5.metric("AUC", f"{m['auc']:.3f}")

st.caption(
    f"Threshold {m['decision_threshold']:.4f} · "
    f"Conformal band width {m['conformal_band_width']:.3f} · "
    f"Precision = of the trucks flagged, how many actually failed · "
    f"Recall = of the actual failures, how many did we catch"
)

# ─── Cost analysis ───
st.subheader("SCANIA Cost Analysis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost", f"${int(m['total_cost']):,}")
c2.metric("Cost / Vehicle", f"${m['cost_per_vehicle']:.2f}")
c3.metric("False Positives", int(m['n_false_positive']))
c4.metric("False Negatives (missed)", int(m['n_false_negative']))

st.info(
    f"Cost matrix: **false positive = $10** (unneeded maintenance), "
    f"**false negative = $500** (missed real failure). "
    f"Model achieved **${int(m['total_cost']):,}** total cost on {int(m['n_validation']):,} validation vehicles."
)

# ─── Confusion matrix ───
st.subheader("Confusion Matrix (5 Urgency Classes)")
cm = load_cm()
cm_matrix = cm[[f"pred_{i}" for i in range(5)]].values
fig_cm = px.imshow(
    cm_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=[f"Class {i}" for i in range(5)],
    y=[f"Class {i}" for i in range(5)],
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="equal"
)
fig_cm.update_layout(height=450, template="plotly_dark")
st.plotly_chart(fig_cm, use_container_width=True)
st.caption("Class 0 = no failure, Class 4 = failure imminent (within ~6 steps).")

# ─── Retrospective RUL analysis ───
st.subheader("Retrospective: Predicted RUL vs Actual Urgency Class")
retro = load_retro()
fig_retro = px.box(retro, x="actual_class", y="predicted_rul",
                    color="actual_class", points="outliers")
fig_retro.update_layout(
    xaxis_title="Actual class (0 = healthy, 4 = failing soon)",
    yaxis_title="Predicted RUL (days)",
    height=400,
    template="plotly_dark",
    showlegend=False
)
st.plotly_chart(fig_retro, use_container_width=True)
st.caption("If the model is well calibrated, predicted RUL should decrease as actual urgency class increases.")

# ─── Feature importances ───
st.subheader("Top 20 Features")
fi = load_importances()
fig_fi = go.Figure(go.Bar(
    x=fi["importance"].values[::-1],
    y=fi["feature"].values[::-1],
    orientation="h",
    marker_color="#2a78d6"
))
fig_fi.update_layout(
    xaxis_title="LightGBM importance",
    height=550,
    template="plotly_dark",
    margin=dict(l=100)
)
st.plotly_chart(fig_fi, use_container_width=True)
st.caption("Features prefixed with `damage_` are the physics-based Miner damage indices.")