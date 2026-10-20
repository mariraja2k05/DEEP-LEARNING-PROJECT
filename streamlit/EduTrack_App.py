"""
EduTrack — OULAD Student Engagement Prediction
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
DATA_DIR = ROOT / "data" / "oulad"

ACCENT = "#00E87A"
AMBER = "#F59E0B"
RED = "#EF4444"
BG = "#000000"
CARD = "#0A0A0A"
TEXT = "#F0F0F0"
DIM = "#4A4A4A"
BORDER = "#1E1E1E"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


def hex_rgba(hex_value: str, alpha: float = 0.15) -> str:
    hex_value = hex_value.lstrip("#")
    r, g, b = [int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]
    return f"rgba({r},{g},{b},{alpha})"


PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,10,0.9)",
    font=dict(color=TEXT, size=11, family="Inter, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=12, r=12, t=40, b=12),
)


@st.cache_data
def load_data() -> dict:
    from src.data_processing import build_student_summary, build_weekly_clicks, load_oulad_tables

    tables = load_oulad_tables(DATA_DIR)
    weekly = build_weekly_clicks(DATA_DIR)
    summary = build_student_summary(DATA_DIR)

    summary["engagement_label"] = summary["final_result"].replace({
        "Pass": "High",
        "Distinction": "High",
        "Fail": "Low",
        "Withdrawn": "Low",
        "Continue": "Medium",
    })

    summary = summary.sort_values(["code_module", "code_presentation", "id_student"]).reset_index(drop=True)
    summary["total_clicks"] = pd.to_numeric(summary["total_clicks"], errors="coerce").fillna(0.0)
    summary["mean_week_clicks"] = pd.to_numeric(summary["mean_week_clicks"], errors="coerce").fillna(0.0)
    summary["active_weeks"] = pd.to_numeric(summary["active_weeks"], errors="coerce").fillna(0)
    summary["max_week_clicks"] = pd.to_numeric(summary["max_week_clicks"], errors="coerce").fillna(0.0)

    return {
        "tables": tables,
        "weekly": weekly,
        "summary": summary,
    }


@st.cache_data
def load_metric_data() -> dict:
    metrics = {}
    for name in ["ann", "lstm", "gru"]:
        path = METRICS_DIR / f"{name}_metrics.json"
        if not path.exists():
            continue
        metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


@st.cache_resource
def load_models() -> dict | None:
    required = {
        "ann": ["ann_engagement.keras", "ann_scaler.joblib", "ann_feature_columns.json", "ann_label_order.json"],
        "lstm": ["lstm_engagement.keras", "lstm_label_order.json"],
        "gru": ["gru_engagement.keras", "gru_label_order.json"],
    }
    models = {}
    missing = []
    for model_name, files in required.items():
        for file_name in files:
            file_path = MODEL_DIR / file_name
            if not file_path.exists():
                missing.append(f"{model_name}: {file_name}")
        if missing and any(item.startswith(f"{model_name}:") for item in missing):
            continue
        if model_name == "ann":
            models[model_name] = {
                "model": tf.keras.models.load_model(MODEL_DIR / "ann_engagement.keras"),
                "scaler": joblib.load(MODEL_DIR / "ann_scaler.joblib"),
                "features": json.loads((MODEL_DIR / "ann_feature_columns.json").read_text(encoding="utf-8")),
                "labels": json.loads((MODEL_DIR / "ann_label_order.json").read_text(encoding="utf-8")),
            }
        else:
            models[model_name] = {
                "model": tf.keras.models.load_model(MODEL_DIR / f"{model_name}_engagement.keras"),
                "labels": json.loads((MODEL_DIR / f"{model_name}_label_order.json").read_text(encoding="utf-8")),
            }
    if missing:
        st.error("Missing model artifacts: " + ", ".join(missing))
        return None
    return models


def derive_student_inputs(student_row: pd.Series, weekly_df: pd.DataFrame) -> dict:
    student_id = int(student_row["id_student"])
    sequence = []
    for week in range(4):
        val = weekly_df[(weekly_df["id_student"] == student_id) & (weekly_df["week"] == week)]["total_clicks"].sum()
        sequence.append(float(val))
    feature_values = {
        "total_clicks_input": float(sum(sequence)),
        "mean_clicks_input": float(np.mean(sequence)) if sequence else 0.0,
        "active_weeks_input": int(np.count_nonzero(np.array(sequence) > 0)),
        "max_week_clicks": float(max(sequence)) if sequence else 0.0,
    }
    return feature_values, np.array(sequence, dtype=float).reshape(4, 1)


def predict_student(student_row: pd.Series, weekly_df: pd.DataFrame, models: dict) -> dict:
    feature_values, sequence = derive_student_inputs(student_row, weekly_df)
    result = {}

    ann_cfg = models["ann"]
    ann_features = list(ann_cfg["features"])
    ann_vector = np.array([[feature_values.get(col, 0.0) for col in ann_features]], dtype=float)
    ann_scaled = ann_cfg["scaler"].transform(ann_vector)
    ann_probs = ann_cfg["model"].predict(ann_scaled, verbose=0)[0]
    ann_idx = int(np.argmax(ann_probs))
    result["ann"] = {
        "label": ann_cfg["labels"][ann_idx],
        "confidence": float(ann_probs[ann_idx] * 100.0),
        "probabilities": {label: float(val) for label, val in zip(ann_cfg["labels"], ann_probs)},
        "features": feature_values,
    }

    for model_name in ["lstm", "gru"]:
        model_cfg = models[model_name]
        model_probs = model_cfg["model"].predict(sequence.reshape(1, 4, 1), verbose=0)[0]
        idx = int(np.argmax(model_probs))
        result[model_name] = {
            "label": model_cfg["labels"][idx],
            "confidence": float(model_probs[idx] * 100.0),
            "probabilities": {label: float(val) for label, val in zip(model_cfg["labels"], model_probs)},
        }
    return result


st.set_page_config(page_title="EduTrack", page_icon="[E]", layout="wide", initial_sidebar_state="collapsed")

# Navigation
NAV_LABELS = ["Dashboard", "Dataset", "Prediction", "Comparison", "Methods"]
NAV_KEYS = ["dashboard", "dataset", "prediction", "comparison", "methods"]
nav_sel = st.radio("nav", NAV_LABELS, horizontal=True, label_visibility="collapsed", key="nav_radio")
active_page = NAV_KEYS[NAV_LABELS.index(nav_sel)]
active_idx = NAV_LABELS.index(nav_sel) + 1

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=League+Spartan:wght@400;500;600;700;800;900&display=swap');
    [data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu {{ visibility:hidden !important; height:0 !important; overflow:hidden !important; }}
    html,body,[data-testid="stAppViewContainer"],.main {{ background:{BG} !important; font-family:'Inter',sans-serif !important; }}
    [data-testid="stMainBlockContainer"],.block-container {{ padding-top:90px !important; padding-left:2rem !important; padding-right:2rem !important; max-width:1480px !important; }}
    .st-key-nav_radio {{ position:fixed !important; top:0 !important; left:0 !important; right:0 !important; z-index:9999 !important; background:transparent; padding:10px 6% 0 !important; }}
    .st-key-nav_radio [data-testid="stRadio"] {{ background:rgba(8,10,18,0.92); border:1px solid rgba(255,255,255,.08); border-radius:16px; width:100%; height:54px; display:flex; align-items:center; }}
    .st-key-nav_radio [data-testid="stRadioGroup"] {{ display:flex; align-items:center; width:100%; gap:0; padding:0 1.2rem; }}
    .st-key-nav_radio [data-testid="stRadioGroup"]::before {{ content:'EduTrack.'; font-size:17px; font-weight:900; letter-spacing:-.04em; background:linear-gradient(135deg,{TEXT} 60%,{ACCENT}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; flex-shrink:0; margin-right:2.2rem; font-family:'League Spartan',sans-serif; }}
    .st-key-nav_radio [data-testid="stRadioGroup"]::after {{ content:'● live OULAD'; margin-left:auto; font-size:10px; color:{DIM}; letter-spacing:.04em; font-family:'League Spartan',sans-serif; }}
    .st-key-nav_radio [data-testid="stRadioGroup"] label {{ display:flex; align-items:center; padding:0 13px; height:54px; cursor:pointer; color:{DIM}; font-size:13px; font-weight:600; font-family:'League Spartan',sans-serif; }}
    .st-key-nav_radio [data-testid="stRadioGroup"] label > div:first-child {{ display:none !important; }}
    .st-key-nav_radio [data-testid="stRadioGroup"] label:nth-child({active_idx}) {{ color:{ACCENT} !important; background:rgba(0,232,122,.06); border-radius:10px; }}
    .st-key-nav_radio [data-testid="stRadioGroup"] label:hover {{ color:{TEXT} !important; }}
    [data-testid="stMetric"] {{ background:{CARD} !important; border:1px solid {BORDER} !important; border-radius:12px; padding:18px 18px; }}
    [data-testid="stMetricLabel"] {{ color:{DIM} !important; font-size:10px !important; text-transform:uppercase; letter-spacing:.12em; font-weight:600; }}
    [data-testid="stMetricValue"] {{ color:{TEXT} !important; font-size:28px !important; font-weight:800; }}
    .section-tag {{ font-size:10px; letter-spacing:.16em; color:{DIM}; text-transform:uppercase; font-weight:700; margin-bottom:10px; }}
    .card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:18px; }}
    .stat-box {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; height:100%; }}
    .stat-label {{ color:{DIM}; text-transform:uppercase; letter-spacing:.12em; font-size:10px; }}
    .stat-value {{ font-size:24px; font-weight:800; color:{TEXT}; margin-top:6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dashboard_summary() -> dict:
    data = load_data()
    summary = data["summary"].copy()
    metrics = load_metric_data()

    model_df = []
    for model_name, payload in metrics.items():
        model_df.append({
            "model": model_name.upper(),
            "accuracy": float(payload.get("accuracy", 0.0)),
            "f1_macro": float(payload.get("f1_macro", 0.0)),
            "precision_macro": float(payload.get("precision_macro", 0.0)),
            "recall_macro": float(payload.get("recall_macro", 0.0)),
        })

    summary_stats = {
        "n_students": int(summary["id_student"].nunique()),
        "n_modules": int(summary["code_module"].nunique()),
        "avg_active_weeks": float(summary["active_weeks"].mean()),
        "avg_total_clicks": float(summary["total_clicks"].mean()),
    }

    return {
        "data": data,
        "metrics": metrics,
        "model_df": pd.DataFrame(model_df),
        "summary_stats": summary_stats,
    }


if active_page == "dashboard":
    dash = get_dashboard_summary()
    data = dash["data"]
    summary = data["summary"]
    metrics = dash["metrics"]
    cfg = load_models()

    st.markdown('<div class="section-tag">OULAD deep learning project</div>', unsafe_allow_html=True)
    st.title("Student Engagement Prediction Dashboard")
    st.caption("Using the Open University Learning Analytics Dataset (OULAD) and deep learning models: ANN, LSTM, and GRU.")

    cols = st.columns(4)
    cols[0].metric("Students", f"{int(summary['id_student'].nunique()):,}")
    cols[1].metric("Modules", f"{int(summary['code_module'].nunique()):,}")
    cols[2].metric("Mean active weeks", f"{summary['active_weeks'].mean():.2f}")
    cols[3].metric("Mean total clicks", f"{summary['total_clicks'].mean():.0f}")

    st.markdown("<div class='section-tag'>Project overview</div>", unsafe_allow_html=True)
    st.markdown(
        """This dashboard reads the verified OULAD tables in the project data directory, computes engagement features from the first four study weeks, and evaluates ANN, LSTM, and GRU classifiers on a held-out test split. The target label is designed as a future-activity engagement label so it is predictive rather than a direct copy of the same input signal."""
    )

    if cfg is None:
        st.warning("At least one trained model file is missing. Model inference is disabled until the artifacts are restored.")

    st.subheader("Model comparison")
    comparison = dash["model_df"]
    if not comparison.empty:
        fig = px.bar(
            comparison,
            x="model",
            y=["accuracy", "f1_macro"],
            barmode="group",
            title="Accuracy and macro F1 by model",
            color_discrete_sequence=[ACCENT, AMBER],
        )
        fig.update_layout(**PLOTLY_BASE, legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-tag'>Verified metrics</div>", unsafe_allow_html=True)
    metric_rows = []
    for model_name, payload in metrics.items():
        metric_rows.append({
            "Model": model_name.upper(),
            "Accuracy": round(float(payload.get("accuracy", 0.0)), 4),
            "Macro F1": round(float(payload.get("f1_macro", 0.0)), 4),
            "Precision": round(float(payload.get("precision_macro", 0.0)), 4),
            "Recall": round(float(payload.get("recall_macro", 0.0)), 4),
        })
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

elif active_page == "dataset":
    data = load_data()
    summary = data["summary"]
    tables = data["tables"]
    weekly = data["weekly"]

    st.title("OULAD Dataset Explorer")
    st.caption("Verified OULAD tables: studentInfo, studentRegistration, studentAssessment, studentVle, assessments, courses, and vle.")

    stat_cols = st.columns(4)
    stat_cols[0].metric("Rows in studentVle", f"{len(tables['studentVle']):,}")
    stat_cols[1].metric("Students", f"{summary['id_student'].nunique():,}")
    stat_cols[2].metric("Assessments", f"{len(tables['assessments']):,}")
    stat_cols[3].metric("Courses", f"{len(tables['courses']):,}")

    module_summary = (
        summary.groupby("code_module")
        .agg(students=("id_student", "nunique"), avg_clicks=("total_clicks", "mean"), avg_active_weeks=("active_weeks", "mean"))
        .reset_index()
    )
    st.subheader("Student activity by module")
    st.dataframe(module_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        module_summary,
        x="code_module",
        y="avg_clicks",
        color="code_module",
        title="Average total clicks by module",
    )
    fig.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekly activity distribution")
    hist = weekly.groupby("week")["total_clicks"].sum().reset_index()
    hist.columns = ["week", "total_clicks"]
    fig2 = px.line(hist, x="week", y="total_clicks", markers=True, title="Aggregate weekly clicks")
    fig2.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig2, use_container_width=True)

elif active_page == "prediction":
    data = load_data()
    summary = data["summary"]
    weekly = data["weekly"]
    models = load_models()

    st.title("Engagement Prediction")
    if models is None:
        st.warning("Model artifacts are missing; please restore the saved Keras files before running predictions.")
        st.stop()

    sample_students = summary[["id_student", "code_module", "code_presentation", "active_weeks", "total_clicks", "mean_week_clicks", "max_week_clicks"]].drop_duplicates().sort_values("total_clicks", ascending=False)
    selected_id = st.selectbox("Select a student", options=sorted(sample_students["id_student"].unique().tolist()))
    student_row = summary[summary["id_student"] == selected_id].iloc[0]

    feature_values, sequence = derive_student_inputs(student_row, weekly)
    result = predict_student(student_row, weekly, models)

    st.subheader(f"Student {selected_id} — feature view")
    st.dataframe(pd.DataFrame([feature_values]), use_container_width=True, hide_index=True)

    st.subheader("Weekly click sequence used by LSTM/GRU")
    sfig = go.Figure()
    sfig.add_trace(go.Scatter(x=list(range(1, 5)), y=sequence.ravel(), mode="lines+markers", line=dict(color=ACCENT, width=3), marker=dict(size=8)))
    sfig.update_layout(**PLOTLY_BASE, xaxis_title="Week", yaxis_title="Clicks", title="Four-week input sequence")
    st.plotly_chart(sfig, use_container_width=True)

    st.subheader("Predicted engagement from ANN, LSTM, and GRU")
    pred_cols = st.columns(3)
    for idx, model_name in enumerate(["ann", "lstm", "gru"]):
        with pred_cols[idx]:
            payload = result[model_name]
            st.markdown(f"### {model_name.upper()}")
            st.metric("Prediction", payload["label"])
            st.metric("Confidence", f"{payload['confidence']:.2f}%")
            st.bar_chart(pd.DataFrame(payload["probabilities"], index=[0]).T.rename_axis("Class").rename(columns={0: "Probability"}))

elif active_page == "comparison":
    metrics = load_metric_data()
    st.title("ANN vs LSTM vs GRU Comparison")
    model_rows = []
    for name, payload in metrics.items():
        model_rows.append({
            "Model": name.upper(),
            "Accuracy": float(payload.get("accuracy", 0.0)),
            "Macro F1": float(payload.get("f1_macro", 0.0)),
            "Precision": float(payload.get("precision_macro", 0.0)),
            "Recall": float(payload.get("recall_macro", 0.0)),
        })
    comparison_df = pd.DataFrame(model_rows)
    if comparison_df.empty:
        st.warning("No model metrics were found in outputs/metrics.")
        st.stop()

    fig = px.bar(
        comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Accuracy and macro F1 across models",
        color_discrete_map={"Accuracy": ACCENT, "Macro F1": AMBER},
    )
    fig.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    for model_name, payload in metrics.items():
        cm = np.array(payload.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=int)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", labels=dict(x="Predicted label", y="Actual label", color="Count"), x=LABELS, y=LABELS)
        fig_cm.update_layout(**PLOTLY_BASE, title=f"{model_name.upper()} confusion matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

elif active_page == "methods":
    st.title("Methodology and limitations")
    st.markdown(
        """
        - The project uses the OULAD tables stored under the project data folder and validates the required files before training.
        - Weekly click activity is aggregated by `id_student`, `code_module`, `code_presentation`, and `week`.
        - The feature window uses the first four study weeks to model early engagement indicators.
        - The target label is constructed from a later activity window using quantile boundaries across the target total clicks. This is a forecasting signal and not the same as the input signal.
        - The held-out evaluation uses a stratified 80/20 train-test split to preserve class balance.
        - The tested models are ANN, LSTM, and GRU, all trained on the same class labels and evaluated on the same held-out data.
        - Limitations: the dataset is observational and the derived labels still reflect a pattern of engagement intensity, not independently observed ground truth; class boundaries are based on quantiles and may not align with every educational domain definition.
        """
    )

    st.subheader("Early disengagement indicators")
    summary = load_data()["summary"]
    early = summary[(summary["active_weeks"] <= 2) | (summary["total_clicks"] <= summary["total_clicks"].quantile(0.25))].sort_values("total_clicks").head(10)
    st.dataframe(early[["id_student", "code_module", "active_weeks", "total_clicks", "mean_week_clicks", "max_week_clicks"]], use_container_width=True, hide_index=True)

    st.subheader("Leakage review")
    st.markdown(
        """
        - The target used for training is based on later-window total clicks, observed after the input window, so it is a forecasting target rather than direct leakage from the same observed inputs.
        - A strict same-student leakage check is not present because each student-course record is used as one example with a future target defined over a different period. This is valid for a temporal prediction task when the label is built from later activity.
        - The split is stratified and does not reuse the same row in both train and test partitions, which is the expected hold-out evaluation setup for this research pipeline.
        """
    )


# minimal fallback to keep app stable if page value is not recognized
if "active_page" not in locals():
    pass
