# pipeline.py
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import pickle, os

# ─── Config ───
DB_PATH = "scania.duckdb"
COST_FN = 10    # false positive cost
COST_FP = COST_FN
COST_TP = COST_FN
COST_TN = 0
COST_MISS = 500  # false negative cost (missing a real failure)

# Histogram groups (these are the load-collective bins → Miner damage)
HIST_GROUPS = {
    "h167": [f"167_{i}" for i in range(10)],
    "h272": [f"272_{i}" for i in range(10)],
    "h291": [f"291_{i}" for i in range(11)],
    "h158": [f"158_{i}" for i in range(10)],
    "h459": [f"459_{i}" for i in range(20)],
    "h397": [f"397_{i}" for i in range(36)],
}
# Counter columns (cumulative → need delta cleaning)
COUNTERS = ["171_0","666_0","427_0","837_0","309_0","835_0","370_0","100_0"]

# TTE → class label buckets (in time_steps)
CLASS_BINS = [0, 6, 12, 24, 48, np.inf]  # class 4,3,2,1,0
CLASS_LABELS_REV = [4, 3, 2, 1, 0]

# ─── Step 1: Load data ───
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    readouts = con.execute("SELECT * FROM train_operational_readouts").df()
    tte = con.execute("SELECT * FROM train_tte").df()
    specs = con.execute("SELECT * FROM train_specifications").df()
    val_readouts = con.execute("SELECT * FROM validation_operational_readouts").df()
    val_labels = con.execute("SELECT * FROM validation_labels").df()
    val_specs = con.execute("SELECT * FROM validation_specifications").df()
    con.close()
    print(f"Loaded: {len(readouts):,} train readouts, {len(tte):,} vehicles")
    return readouts, tte, specs, val_readouts, val_labels, val_specs

# ─── Step 2: Clean counter resets ───
def clean_counters(df):
    """ECU resets can cause negative deltas in counters — clip them."""
    df = df.sort_values(["vehicle_id", "time_step"]).copy()
    for c in COUNTERS:
        if c in df.columns:
            deltas = df.groupby("vehicle_id")[c].diff()
            # Where delta is negative, it's a reset — carry forward previous value
            mask = deltas < 0
            if mask.any():
                df.loc[mask, c] = np.nan
                df[c] = df.groupby("vehicle_id")[c].ffill()
    print(f"Cleaned counter resets in {len(COUNTERS)} columns")
    return df

# ─── Step 3: Miner damage indices ───
def miner_damage(df, beta=2.0):
    """
    Physics-inspired damage index per histogram group.
    D = sum(bin_count * (bin_index + 1)^beta) for each group.
    Beta acts like a Basquin exponent — higher bins = more severe loads.
    """
    damage_cols = {}
    for name, cols in HIST_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            weights = np.array([(i + 1) ** beta for i in range(len(present))])
            damage_cols[f"damage_{name}"] = df[present].values @ weights
    damage_df = pd.DataFrame(damage_cols, index=df.index)
    print(f"Created {len(damage_cols)} Miner damage features")
    return pd.concat([df, damage_df], axis=1)

# ─── Step 4: Latest snapshot per vehicle ───
def latest_snapshot(df):
    """Take the last readout per vehicle — that's what we predict on."""
    idx = df.groupby("vehicle_id")["time_step"].idxmax()
    snap = df.loc[idx].reset_index(drop=True)
    print(f"Snapshot: {len(snap)} vehicles")
    return snap

# ─── Step 5: Build labels ───
def build_labels(snap, tte):
    """
    For vehicles with in_study_repair=1: TTE = length_of_study - last_time_step
    For censored (in_study_repair=0): class 0 (no failure in window)
    """
    merged = snap.merge(tte, on="vehicle_id", how="inner")
    merged["tte"] = merged["length_of_study_time_step"] - merged["time_step"]
    merged["tte"] = merged["tte"].clip(lower=0)

    # Assign class labels
    merged["class_label"] = 0  # default: no failure
    failed = merged["in_study_repair"] == 1
    merged.loc[failed, "class_label"] = pd.cut(
        merged.loc[failed, "tte"],
        bins=CLASS_BINS,
        labels=CLASS_LABELS_REV,
        right=True
    ).astype(int)

    print(f"Label distribution:\n{merged['class_label'].value_counts().sort_index()}")
    return merged

# ─── Step 6: Prepare features ───
def get_feature_cols(df):
    """All numeric columns except identifiers, labels, and metadata."""
    exclude = {"vehicle_id", "time_step", "length_of_study_time_step",
               "in_study_repair", "tte", "class_label", "risk_score",
               "pred_class", "conf_low", "conf_high"}
    spec_cols = [c for c in df.columns if c.startswith("Spec_")]
    return [c for c in df.columns if c not in exclude and c not in spec_cols
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

# ─── Step 7: Train LightGBM ───
def train_model(train_df, feature_cols):
    X = train_df[feature_cols].values
    y = train_df["class_label"].values

    # Binary: class >= 1 means failure upcoming
    y_bin = (y >= 1).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    scale = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": scale,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y_bin)
    print(f"Trained LightGBM: {n_pos} failures, {n_neg} non-failures, scale_pos_weight={scale:.1f}")
    return model

# ─── Step 8: Bayes-optimal decision ───
def bayes_decision(proba, threshold=None):
    """Cost-optimal threshold: flag if P(fail) > cost_fp / (cost_fp + cost_miss)"""
    if threshold is None:
        threshold = COST_FP / (COST_FP + COST_MISS)
    return (proba >= threshold).astype(int), threshold

# ─── Step 9: Conformal prediction ───
def conformal_intervals(cal_proba, cal_y, test_proba, alpha=0.10):
    """
    Split conformal: use calibration scores to build prediction intervals.
    Score = |y - proba|. Quantile gives the conformal width.
    """
    cal_scores = np.abs(cal_y - cal_proba)
    q = np.quantile(cal_scores, 1 - alpha, method="higher")
    lo = np.clip(test_proba - q, 0, 1)
    hi = np.clip(test_proba + q, 0, 1)
    print(f"Conformal band width (q): {q:.3f}, coverage target: {1-alpha:.0%}")
    return lo, hi, q

# ─── Step 10: Risk score → RUL estimate ───
def risk_to_rul(proba, max_rul=300):
    """Simple mapping: higher failure probability → lower RUL."""
    return np.round((1 - proba) * max_rul).astype(int)

# ─── Step 11: Recommended actions ───
def recommend_action(pred_class, rul):
    actions = []
    for p, r in zip(pred_class, rul):
        if p == 1 and r < 30:
            actions.append("Book workshop slot in 5-10 days; order parts now.")
        elif p == 1 and r < 60:
            actions.append("Order parts and schedule maintenance this month.")
        elif p == 1:
            actions.append("Re-evaluate at next monthly readout.")
        else:
            actions.append("No action — routine monitoring.")
    return actions

# ═══════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("SCANIA PdM Pipeline — Day 2")
    print("=" * 50)

    # Load
    readouts, tte, specs, val_readouts, val_labels, val_specs = load_data()

    # Clean
    readouts = clean_counters(readouts)
    val_readouts = clean_counters(val_readouts)

    # Miner damage features
    readouts = miner_damage(readouts)
    val_readouts = miner_damage(val_readouts)

    # Snapshot
    train_snap = latest_snapshot(readouts)
    val_snap = latest_snapshot(val_readouts)

    # Labels
    train_labeled = build_labels(train_snap, tte)
    val_labeled = val_snap.merge(val_labels, on="vehicle_id", how="inner")

    # Features
    feature_cols = get_feature_cols(train_labeled)
    print(f"Using {len(feature_cols)} features")

    # Split train into train/calibration (80/20 by vehicle)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, cal_idx = next(splitter.split(
        train_labeled, groups=train_labeled["vehicle_id"]
    ))
    train_set = train_labeled.iloc[train_idx]
    cal_set = train_labeled.iloc[cal_idx]
    print(f"Train: {len(train_set)}, Calibration: {len(cal_set)}")

    # Train
    model = train_model(train_set, feature_cols)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/lgbm_baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/lgbm_baseline.pkl")

    # Save feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv("models/feature_importances.csv", index=False)

    # Predict on calibration set
    cal_proba = model.predict_proba(cal_set[feature_cols].values)[:, 1]
    cal_y = (cal_set["class_label"].values >= 1).astype(int)

    # Predict on validation set
    val_X = val_labeled[feature_cols].values
    val_proba = model.predict_proba(val_X)[:, 1]

    # Conformal intervals
    val_lo, val_hi, q = conformal_intervals(cal_proba, cal_y, val_proba)

    # Bayes decision
    val_pred, thresh = bayes_decision(val_proba)
    print(f"Decision threshold: {thresh:.4f}")

    # RUL
    val_rul = risk_to_rul(val_proba)

    # Actions
    val_actions = recommend_action(val_pred, val_rul)

    # Build predictions table
    predictions = pd.DataFrame({
        "vehicle_id": val_labeled["vehicle_id"].values,
        "pred_class": val_pred,
        "risk_score": np.round(val_proba, 4),
        "rul_estimate": val_rul,
        "conf_low": np.round(val_lo, 4),
        "conf_high": np.round(val_hi, 4),
        "recommended_action": val_actions,
    })

    # Also build predictions for ALL train vehicles (for dashboard)
    all_proba = model.predict_proba(train_labeled[feature_cols].values)[:, 1]
    all_lo, all_hi, _ = conformal_intervals(cal_proba, cal_y, all_proba)
    all_pred, _ = bayes_decision(all_proba)
    all_rul = risk_to_rul(all_proba)
    all_actions = recommend_action(all_pred, all_rul)

    all_predictions = pd.DataFrame({
        "vehicle_id": train_labeled["vehicle_id"].values,
        "pred_class": all_pred,
        "risk_score": np.round(all_proba, 4),
        "rul_estimate": all_rul,
        "conf_low": np.round(all_lo, 4),
        "conf_high": np.round(all_hi, 4),
        "recommended_action": all_actions,
    })

    # Combine train + validation predictions
    full_predictions = pd.concat([all_predictions, predictions], ignore_index=True)

    # ─── Write to DuckDB ───
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM full_predictions")
    con.execute("""
        CREATE OR REPLACE TABLE feature_importances AS
        SELECT * FROM read_csv_auto('models/feature_importances.csv')
    """)
    con.close()

    print(f"\n{'=' * 50}")
    print(f"Predictions written: {len(full_predictions):,} vehicles")
    print(f"Flagged: {full_predictions['pred_class'].sum():,}")
    print(f"Top 5 riskiest:")
    print(full_predictions.nlargest(5, "risk_score")[
        ["vehicle_id","risk_score","rul_estimate","recommended_action"]
    ].to_string(index=False))
    print(f"{'=' * 50}")
    print("Day 2 done!")