import streamlit as st
import duckdb
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Validation Forecasts", layout="wide")

# ─── Cost matrix (must match pipeline.py) ───
COST_FP = 10
COST_MISS = 500

@st.cache_resource
def get_con():
    return duckdb.connect("scania.duckdb", read_only=True)

@st.cache_data
def load_val_predictions():
    return get_con().execute("""
        SELECT p.*
        FROM predictions p
        INNER JOIN validation_labels v ON p.vehicle_id = v.vehicle_id
        ORDER BY p.risk_score DESC
    """).df()

@st.cache_data
def load_val_labels():
    return get_con().execute("SELECT * FROM validation_labels").df()

@st.cache_data
def get_readouts(vehicle_id):
    return get_con().execute(f"""
        SELECT * FROM validation_operational_readouts
        WHERE vehicle_id = {vehicle_id}
        ORDER BY time_step
    """).df()

@st.cache_data
def load_metrics():
    return get_con().execute("SELECT * FROM metrics").df().iloc[0]

# ─── Synthetic health index from counter columns ───
def compute_health(readouts):
    counters = ["171_0","666_0","427_0","837_0","309_0","835_0","370_0","100_0"]
    present = [c for c in counters if c in readouts.columns]
    if not present or len(readouts) < 2:
        return readouts["time_step"].values, np.ones(len(readouts))
    vals = readouts[present].values
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (vals - mins) / ranges
    health = 1 - normed.mean(axis=1)
    return readouts["time_step"].values, health

st.title("Validation Forecasts")
st.caption("Per-vehicle risk trajectory, conformal interval, and cost-based Bayes decision on the SCANIA validation set.")

preds = load_val_predictions()
labels = load_val_labels()
m = load_metrics()

# ─── Vehicle selector ───
merged = preds.merge(labels, on="vehicle_id")
vehicle_ids = merged["vehicle_id"].tolist()

selected = st.selectbox(
    "Select a validation vehicle",
    vehicle_ids,
    format_func=lambda x: (
        f"Vehicle {x} — risk {merged[merged['vehicle_id']==x]['risk_score'].values[0]:.4f}"
        f" · actual class {merged[merged['vehicle_id']==x]['class_label'].values[0]}"
    )
)

truck = merged[merged["vehicle_id"] == selected].iloc[0]
readouts = get_readouts(selected)

# ─── KPI row ───
c1, c2, c3, c4 = st.columns(4)
risk_icon = "🔴" if truck["risk_score"] > 0.5 else "🟡" if truck["risk_score"] > 0.1 else "🟢"
c1.metric("Risk Score", f"{risk_icon} {truck['risk_score']:.4f}")
c2.metric("Predicted RUL", f"{truck['rul_estimate']} days")
c3.metric("Actual Class", int(truck["class_label"]))
c4.metric("Decision", "🚨 Flagged" if truck["pred_class"] == 1 else "✅ OK")

# ─── Risk trajectory line chart ───
st.subheader("Risk Trajectory with Conformal Interval")

if len(readouts) > 1:
    time_steps, health = compute_health(readouts)
    # Convert health to a rough risk score over time (1 - health)
    risk_traj = 1 - health
    # Anchor the last point to the model's actual risk score
    risk_traj = risk_traj * (truck["risk_score"] / max(risk_traj[-1], 1e-6))
    risk_traj = np.clip(risk_traj, 0, 1)

    # Conformal band around the trajectory
    band = (truck["conf_high"] - truck["conf_low"]) / 2
    upper = np.clip(risk_traj + band, 0, 1)
    lower = np.clip(risk_traj - band, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=upper, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time_steps, y=lower, fill="tonexty",
                             line=dict(width=0), name="90% conformal band",
                             fillcolor="rgba(42,120,214,0.15)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time_steps, y=risk_traj,
                             name="Risk score",
                             line=dict(color="#2a78d6", width=2.5)))
    fig.add_hline(y=m["decision_threshold"], line_dash="dash",
                  line_color="#e34948",
                  annotation_text=f"Decision threshold ({m['decision_threshold']:.4f})")

    fig.update_layout(
        xaxis_title="Operating time step",
        yaxis_title="Risk score (probability of failure)",
        yaxis=dict(range=[0, 1.05]),
        height=430,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough readout data to plot a trajectory.")

# ─── Bayes decision explained ───
st.subheader("Bayes Cost-Based Decision")

expected_cost_flag = COST_FP * (1 - truck["risk_score"])
expected_cost_no_flag = COST_MISS * truck["risk_score"]
optimal = "Flag for maintenance" if expected_cost_flag < expected_cost_no_flag else "No action"

st.markdown(f"""
The Bayes decision minimizes expected cost. For this vehicle:

- **P(failure)** = `{truck['risk_score']:.4f}` (conformal band `{truck['conf_low']:.3f} – {truck['conf_high']:.3f}`)
- **Cost of flagging** = false positive × P(no failure) = **${COST_FP}** × `{1-truck['risk_score']:.4f}` = **${expected_cost_flag:.2f}**
- **Cost of not flagging** = missed failure × P(failure) = **${COST_MISS}** × `{truck['risk_score']:.4f}` = **${expected_cost_no_flag:.2f}**
- **Optimal decision:** {optimal}
- **Recommended action:** {truck["recommended_action"]}
""")

if truck["pred_class"] == 1:
    st.success(f"✅ Model flagged this truck. Expected savings vs. no action: **${expected_cost_no_flag - expected_cost_flag:.2f}**")
else:
    st.info(f"🟢 Model chose not to flag. Doing nothing is cheaper by **${expected_cost_flag - expected_cost_no_flag:.2f}** in expectation.")

# ─── Cost optimizer curve ───
st.subheader("Cost Optimizer: When to Schedule Maintenance")

rul = int(truck["rul_estimate"])
days = np.arange(1, min(rul + 30, 120))
# Waste of maintaining too early (component life left)
early_waste = COST_FP * np.maximum(0, rul - days) / max(rul, 1) * 1.5
# Failure risk climbs sigmoidally past the RUL point
sigma = max((truck["conf_high"] - truck["conf_low"]) * 20, 3)
failure_prob = 1 / (1 + np.exp(-(days - rul) / sigma))
expected_cost = COST_FP + early_waste + COST_MISS * failure_prob
optimal_day = int(days[np.argmin(expected_cost)])

fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(x=days, y=expected_cost, mode="lines",
                              line=dict(color="#2a78d6", width=2.5),
                              name="Expected cost"))
fig_cost.add_trace(go.Scatter(x=[optimal_day], y=[expected_cost.min()],
                              mode="markers",
                              marker=dict(size=12, color="#eda100",
                                          line=dict(color="#854f0b", width=1.5)),
                              name=f"Optimal: day {optimal_day}"))
fig_cost.update_layout(
    xaxis_title="Maintain in N days",
    yaxis_title="Expected total cost ($)",
    height=380,
    template="plotly_dark",
    legend=dict(orientation="h", y=-0.15)
)
st.plotly_chart(fig_cost, use_container_width=True)
st.caption(f"Cost-optimal maintenance window: **day {optimal_day}** (expected cost **${expected_cost.min():.2f}**). "
           "Too early wastes component life; too late risks the failure cost.")