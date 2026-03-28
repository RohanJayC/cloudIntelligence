import streamlit as st
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CloudSentinel",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }

/* Header */
.sentinel-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0.5rem;
}
.sentinel-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
}
.sentinel-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.badge {
    display: inline-block;
    background: #0f1f0f;
    border: 1px solid #16a34a;
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 12px;
    vertical-align: middle;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.metric-card.danger::before {
    background: linear-gradient(90deg, #ef4444, #f97316);
}
.metric-card.success::before {
    background: linear-gradient(90deg, #22c55e, #16a34a);
}
.metric-card.warning::before {
    background: linear-gradient(90deg, #f59e0b, #eab308);
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.4rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* Anomaly table */
.anomaly-row {
    background: #1a0a0a;
    border-left: 3px solid #ef4444;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #fca5a5;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.status-anomaly {
    background: #1f0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
}
.status-normal {
    background: #0a1f0a;
    color: #4ade80;
    border: 1px solid #14532d;
}

/* Info box */
.info-box {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
REGION      = "us-east-2"
INSTANCE_ID = "i-0372640781068e64b"

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="sentinel-header">
    <div class="sentinel-title">⚡ CloudSentinel <span class="badge">LIVE</span></div>
</div>
<div class="sentinel-subtitle">
    Autonomous Cloud Cost Intelligence · AWS EC2 · Logistic Regression Anomaly Detection
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA PIPELINE (cached for performance)
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def run_pipeline():
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)

    def get_metric(metric_name):
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=metric_name,
            Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
            StartTime=datetime.now(timezone.utc) - timedelta(hours=24),
            EndTime=datetime.now(timezone.utc),
            Period=300,
            Statistics=["Average"]
        )
        return response.get("Datapoints", [])

    def to_df(data, col_name):
        if not data:
            return pd.DataFrame(columns=["Timestamp", col_name])
        df = pd.DataFrame(data)
        if "Timestamp" not in df.columns or "Average" not in df.columns:
            return pd.DataFrame(columns=["Timestamp", col_name])
        df = df.sort_values("Timestamp")[["Timestamp", "Average"]]
        df.rename(columns={"Average": col_name}, inplace=True)
        return df

    cpu_data     = get_metric("CPUUtilization")
    net_in_data  = get_metric("NetworkIn")
    net_out_data = get_metric("NetworkOut")

    df_cpu = to_df(cpu_data,     "cpu_usage_pct")
    df_in  = to_df(net_in_data,  "network_in_mb")
    df_out = to_df(net_out_data, "network_out_mb")

    df = pd.merge(df_cpu, df_in,  on="Timestamp", how="outer")
    df = pd.merge(df,     df_out, on="Timestamp", how="outer")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df = df.ffill().bfill()

    if "network_in_mb" in df.columns:
        df["network_in_mb"] /= (1024 * 1024)
    if "network_out_mb" in df.columns:
        df["network_out_mb"] /= (1024 * 1024)

    if len(df) < 10:
        return None, None, None, None, None, None, None

    # Z-score + labels
    df["z_score"]   = np.abs(stats.zscore(df["cpu_usage_pct"]))
    df["is_anomaly"] = (df["z_score"] > 2.5).astype(int)

    feature_names = ["cpu_usage_pct", "network_in_mb", "network_out_mb"]
    X = df[feature_names].values
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred, labels=[0, 1])
    coefs    = model.coef_[0]

    return df, accuracy, cm, coefs, feature_names, len(X_train), len(X_test)

# ─────────────────────────────────────────
# REFRESH BUTTON
# ─────────────────────────────────────────
col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Fetching live AWS telemetry..."):
    df, accuracy, cm, coefs, feature_names, n_train, n_test = run_pipeline()

if df is None:
    st.error("Not enough data points. Wait a few minutes for CloudWatch to populate, then refresh.")
    st.stop()

# ─────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

total      = len(df)
n_anomaly  = int(df["is_anomaly"].sum())
n_normal   = total - n_anomaly
latest_cpu = df["cpu_usage_pct"].iloc[-1]
max_zscore = df["z_score"].max()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card success">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">{accuracy*100:.1f}%</div>
        <div class="metric-sub">Logistic Regression</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Datapoints</div>
        <div class="metric-value">{total}</div>
        <div class="metric-sub">Last 24 hours · 5min intervals</div>
    </div>""", unsafe_allow_html=True)

with c3:
    card_class = "danger" if n_anomaly > 0 else "success"
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <div class="metric-label">Anomalies Detected</div>
        <div class="metric-value">{n_anomaly}</div>
        <div class="metric-sub">{n_normal} normal points</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card warning">
        <div class="metric-label">Peak Z-Score</div>
        <div class="metric-value">{max_zscore:.2f}σ</div>
        <div class="metric-sub">Threshold at 2.5σ</div>
    </div>""", unsafe_allow_html=True)

with c5:
    cpu_color = "danger" if latest_cpu > 20 else "success"
    st.markdown(f"""
    <div class="metric-card {cpu_color}">
        <div class="metric-label">Current CPU</div>
        <div class="metric-value">{latest_cpu:.2f}%</div>
        <div class="metric-sub">{df["Timestamp"].max().strftime("%H:%M UTC")}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHARTS ROW 1 — CPU + Z-SCORE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Live Telemetry</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

plt.style.use("dark_background")

with chart_col1:
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    normal_mask  = df["is_anomaly"] == 0
    anomaly_mask = df["is_anomaly"] == 1

    ax.plot(df.index, df["cpu_usage_pct"],
            color="#3b82f6", linewidth=1.2, alpha=0.8, zorder=2)
    ax.fill_between(df.index, df["cpu_usage_pct"],
                    alpha=0.1, color="#3b82f6")
    ax.scatter(df.index[anomaly_mask], df["cpu_usage_pct"][anomaly_mask],
               color="#ef4444", s=60, zorder=5, label="Anomaly")

    ax.set_title("CPU Utilization (%)", color="#94a3b8",
                 fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.spines[:].set_color("#1e293b")
    ax.set_xlabel("Datapoint Index", color="#475569", fontsize=8)
    ax.legend(facecolor="#0d1117", edgecolor="#1e293b",
              labelcolor="#f87171", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_col2:
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(df.index, df["z_score"],
            color="#06b6d4", linewidth=1.2, alpha=0.8)
    ax.fill_between(df.index, df["z_score"], alpha=0.1, color="#06b6d4")
    ax.axhline(y=2.5, color="#ef4444", linewidth=1,
               linestyle="--", alpha=0.8, label="Threshold (2.5σ)")
    ax.scatter(df.index[anomaly_mask], df["z_score"][anomaly_mask],
               color="#ef4444", s=60, zorder=5)

    ax.set_title("Z-Score over Time", color="#94a3b8",
                 fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.spines[:].set_color("#1e293b")
    ax.set_xlabel("Datapoint Index", color="#475569", fontsize=8)
    ax.legend(facecolor="#0d1117", edgecolor="#1e293b",
              labelcolor="#94a3b8", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# CHARTS ROW 2 — CONFUSION MATRIX + FEATURE IMPORTANCE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.Blues, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_yticklabels(["Normal", "Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_xlabel("Predicted", color="#64748b", fontsize=9)
    ax.set_ylabel("Actual", color="#64748b", fontsize=9)
    ax.set_title("Confusion Matrix", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.spines[:].set_color("#1e293b")
    ax.tick_params(colors="#475569")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_col4:
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#3b82f6" if c >= 0 else "#ef4444" for c in coefs]
    bars = ax.barh(feature_names, coefs, color=colors, alpha=0.85)
    ax.set_title("Feature Importance (Coefficients)",
                 color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.spines[:].set_color("#1e293b")
    ax.set_xlabel("Coefficient Value", color="#64748b", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5, axis="x")
    ax.axvline(x=0, color="#475569", linewidth=0.8)

    for bar, val in zip(bars, coefs):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color="#94a3b8", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# ANOMALY LOG
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Anomaly Event Log</div>', unsafe_allow_html=True)

anomalies = df[df["is_anomaly"] == 1].copy()

if len(anomalies) == 0:
    st.markdown("""
    <div class="info-box">
        ✅ No anomalies detected in the current 24-hour window.
    </div>""", unsafe_allow_html=True)
else:
    for _, row in anomalies.iterrows():
        ts = row["Timestamp"].strftime("%Y-%m-%d %H:%M UTC") if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
        st.markdown(f"""
        <div class="anomaly-row">
            🔴 &nbsp; <b>{ts}</b> &nbsp;|&nbsp;
            CPU: <b>{row['cpu_usage_pct']:.2f}%</b> &nbsp;|&nbsp;
            Z-Score: <b>{row['z_score']:.3f}σ</b> &nbsp;|&nbsp;
            Net In: {row['network_in_mb']:.4f} MB &nbsp;|&nbsp;
            Net Out: {row['network_out_mb']:.4f} MB
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL SUMMARY
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Model Summary</div>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""
    <div class="info-box">
        <b style="color:#e2e8f0">Training Split</b><br><br>
        Training samples: <b style="color:#f1f5f9">{n_train}</b><br>
        Testing samples: <b style="color:#f1f5f9">{n_test}</b><br>
        Test size: <b style="color:#f1f5f9">50%</b><br>
        Stratified: <b style="color:#4ade80">Yes</b>
    </div>""", unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="info-box">
        <b style="color:#e2e8f0">Detection Config</b><br><br>
        Z-score threshold: <b style="color:#f1f5f9">2.5σ</b><br>
        Anomaly rate: <b style="color:#f1f5f9">{n_anomaly/total*100:.1f}%</b><br>
        Class weight: <b style="color:#f1f5f9">Balanced</b><br>
        Max iterations: <b style="color:#f1f5f9">1000</b>
    </div>""", unsafe_allow_html=True)

with s3:
    st.markdown(f"""
    <div class="info-box">
        <b style="color:#e2e8f0">Data Source</b><br><br>
        Provider: <b style="color:#f1f5f9">AWS CloudWatch</b><br>
        Instance: <b style="color:#f1f5f9">{INSTANCE_ID[:12]}...</b><br>
        Region: <b style="color:#f1f5f9">{REGION}</b><br>
        Window: <b style="color:#f1f5f9">Last 24 hours</b>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#1e293b; font-family:'Space Mono',monospace; font-size:0.7rem; padding: 1rem 0;">
    CLOUDSENTINEL · MIT BENGALURU HACKATHON 2026 · AUTONOMOUS CLOUD COST INTELLIGENCE
</div>""", unsafe_allow_html=True)
