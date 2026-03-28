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

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="CloudSentinel", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #080c14; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }
.sentinel-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -1px; }
.sentinel-subtitle { font-size: 0.95rem; color: #64748b; margin-bottom: 2rem; font-weight: 300; }
.badge { display: inline-block; background: #0f1f0f; border: 1px solid #16a34a; color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; margin-left: 12px; vertical-align: middle; letter-spacing: 1px; }
.metric-card { background: #0d1117; border: 1px solid #1e293b; border-radius: 12px; padding: 1.4rem 1.6rem; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.danger::before  { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-card.success::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.metric-card.warning::before { background: linear-gradient(90deg, #f59e0b, #eab308); }
.metric-label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem; font-family: 'Space Mono', monospace; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-sub { font-size: 0.78rem; color: #475569; margin-top: 0.4rem; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
.instance-card { background: #0d1117; border: 1px solid #1e293b; border-radius: 12px; padding: 1.4rem 1.6rem; }
.instance-running { color: #4ade80; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem; }
.instance-stopped { color: #f87171; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem; }
.remediation-low    { background: #0f1a0f; border-left: 3px solid #22c55e; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #86efac; }
.remediation-medium { background: #1a140a; border-left: 3px solid #f59e0b; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #fcd34d; }
.remediation-high   { background: #1a0a0a; border-left: 3px solid #ef4444; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #fca5a5; }
.info-box { background: #0d1117; border: 1px solid #1e293b; border-radius: 10px; padding: 1.2rem 1.4rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
REGION      = "us-east-2"
INSTANCE_ID = "i-0372640781068e64b"
ENABLE_STOP = False  # 🔴 set True only for live demo

ec2_client        = boto3.client("ec2",        region_name=REGION)
cloudwatch_client = boto3.client("cloudwatch", region_name=REGION)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="sentinel-title">⚡ CloudSentinel <span class="badge">LIVE</span></div>
<div class="sentinel-subtitle">Autonomous Cloud Cost Intelligence · AWS EC2 · Logistic Regression · Auto-Remediation</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# REMEDIATION ENGINE
# ─────────────────────────────────────────
def classify_severity(z):
    if z >= 6.0:   return "HIGH"
    elif z >= 4.0: return "MEDIUM"
    else:          return "LOW"

def tag_instance(z, severity, ts):
    try:
        ec2_client.create_tags(Resources=[INSTANCE_ID], Tags=[
            {"Key": "AnomalyDetected",  "Value": "true"},
            {"Key": "AnomalySeverity",  "Value": severity},
            {"Key": "AnomalyZScore",    "Value": str(round(z, 4))},
            {"Key": "AnomalyTimestamp", "Value": ts},
            {"Key": "RemediatedBy",     "Value": "CloudSentinel"},
        ])
        return True, "Tagged instance in AWS console"
    except Exception as e:
        return False, str(e)

def create_alarm(z):
    try:
        threshold = 30.0 if z >= 6.0 else 50.0
        cloudwatch_client.put_metric_alarm(
            AlarmName="CloudSentinel-CPU-Anomaly",
            AlarmDescription=f"Auto-created by CloudSentinel. Z={z:.3f}",
            ActionsEnabled=False,
            MetricName="CPUUtilization", Namespace="AWS/EC2", Statistic="Average",
            Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
            Period=300, EvaluationPeriods=1, Threshold=threshold,
            ComparisonOperator="GreaterThanThreshold", TreatMissingData="notBreaching"
        )
        return True, f"CloudWatch alarm set — threshold {threshold}%"
    except Exception as e:
        return False, str(e)

def stop_instance():
    if not ENABLE_STOP:
        return False, "Stop skipped (safe mode)"
    try:
        ec2_client.stop_instances(InstanceIds=[INSTANCE_ID])
        return True, "Stop command issued"
    except Exception as e:
        return False, str(e)

def remediate(z, cpu, ts):
    severity = classify_severity(z)
    actions  = []
    if severity in ["LOW", "MEDIUM", "HIGH"]:
        ok, msg = tag_instance(z, severity, ts)
        actions.append({"action": "Tag Instance",    "ok": ok, "msg": msg})
    if severity in ["MEDIUM", "HIGH"]:
        ok, msg = create_alarm(z)
        actions.append({"action": "Create CW Alarm", "ok": ok, "msg": msg})
    if severity == "HIGH":
        ok, msg = stop_instance()
        actions.append({"action": "Stop Instance",   "ok": ok, "msg": msg})
    return severity, actions

# ─────────────────────────────────────────
# INSTANCE STATUS
# ─────────────────────────────────────────
def get_instance_status():
    try:
        r    = ec2_client.describe_instances(InstanceIds=[INSTANCE_ID])
        inst = r["Reservations"][0]["Instances"][0]
        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
        return {
            "state"    : inst["State"]["Name"],
            "type"     : inst["InstanceType"],
            "az"       : inst["Placement"]["AvailabilityZone"],
            "anomaly"  : tags.get("AnomalyDetected", "false"),
            "severity" : tags.get("AnomalySeverity", "—"),
            "z_score"  : tags.get("AnomalyZScore",   "—"),
            "remediated_by": tags.get("RemediatedBy",     "—"),
            "remediated_at": tags.get("AnomalyTimestamp", "—"),
        }
    except Exception as e:
        return {"state": "unknown", "error": str(e)}

# ─────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def run_pipeline():
    def get_metric(name):
        r = cloudwatch_client.get_metric_statistics(
            Namespace="AWS/EC2", MetricName=name,
            Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
            StartTime=datetime.now(timezone.utc) - timedelta(hours=24),
            EndTime=datetime.now(timezone.utc),
            Period=300, Statistics=["Average"]
        )
        return r.get("Datapoints", [])

    def to_df(data, col):
        if not data: return pd.DataFrame(columns=["Timestamp", col])
        d = pd.DataFrame(data)
        if "Timestamp" not in d.columns: return pd.DataFrame(columns=["Timestamp", col])
        return d.sort_values("Timestamp")[["Timestamp","Average"]].rename(columns={"Average": col})

    df = pd.merge(to_df(get_metric("CPUUtilization"), "cpu_usage_pct"),
                  to_df(get_metric("NetworkIn"),      "network_in_mb"),  on="Timestamp", how="outer")
    df = pd.merge(df, to_df(get_metric("NetworkOut"), "network_out_mb"), on="Timestamp", how="outer")
    df = df.sort_values("Timestamp").reset_index(drop=True).ffill().bfill()

    for col in ["network_in_mb", "network_out_mb"]:
        if col in df.columns: df[col] /= (1024*1024)

    if len(df) < 10: return None, None, None, None, None, None, None

    df["z_score"]    = np.abs(stats.zscore(df["cpu_usage_pct"]))
    df["is_anomaly"] = (df["z_score"] > 2.5).astype(int)

    features = ["cpu_usage_pct", "network_in_mb", "network_out_mb"]
    X, y     = df[features].values, df["is_anomaly"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return (df,
            accuracy_score(y_te, y_pred),
            confusion_matrix(y_te, y_pred, labels=[0,1]),
            model.coef_[0],
            features,
            len(X_tr), len(X_te))

# ─────────────────────────────────────────
# REFRESH
# ─────────────────────────────────────────
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Fetching live AWS telemetry..."):
    df, accuracy, cm, coefs, features, n_train, n_test = run_pipeline()

if df is None:
    st.error("Not enough data. Wait a few minutes then refresh.")
    st.stop()

# ─────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

total, n_anom = len(df), int(df["is_anomaly"].sum())
latest_cpu, max_z = df["cpu_usage_pct"].iloc[-1], df["z_score"].max()

c1,c2,c3,c4,c5 = st.columns(5)
cards = [
    (c1, "success", "Model Accuracy",     f"{accuracy*100:.1f}%", "Logistic Regression"),
    (c2, "",        "Total Datapoints",    str(total),             "Last 24h · 5min intervals"),
    (c3, "danger" if n_anom>0 else "success", "Anomalies", str(n_anom), f"{total-n_anom} normal"),
    (c4, "warning", "Peak Z-Score",        f"{max_z:.2f}σ",        "Threshold 2.5σ"),
    (c5, "danger" if latest_cpu>20 else "success", "Current CPU", f"{latest_cpu:.2f}%",
         df["Timestamp"].max().strftime("%H:%M UTC")),
]
for col, cls, label, val, sub in cards:
    with col:
        st.markdown(f"""<div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INSTANCE STATUS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Instance Status</div>', unsafe_allow_html=True)
status = get_instance_status()

is1, is2, is3 = st.columns(3)
state_cls  = "instance-running" if status.get("state") == "running" else "instance-stopped"
state_icon = "🟢" if status.get("state") == "running" else "🔴"

with is1:
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Instance State</div>
        <div class="{state_cls}">{state_icon} {status.get('state','—').upper()}</div>
        <div class="metric-sub">{INSTANCE_ID}</div>
        <div class="metric-sub">Type: {status.get('type','—')} · AZ: {status.get('az','—')}</div>
    </div>""", unsafe_allow_html=True)

with is2:
    flagged   = status.get("anomaly") == "true"
    sev_color = {"HIGH":"#f87171","MEDIUM":"#fcd34d","LOW":"#86efac"}.get(status.get("severity",""),"#64748b")
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Anomaly Tag</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:{sev_color}">
            {'🚨 FLAGGED' if flagged else '✅ CLEAN'}
        </div>
        <div class="metric-sub">Severity: {status.get('severity','—')}</div>
        <div class="metric-sub">Z-Score recorded: {status.get('z_score','—')}σ</div>
    </div>""", unsafe_allow_html=True)

with is3:
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Last Remediation</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#94a3b8;font-weight:600">
            {status.get('remediated_by','—')}
        </div>
        <div class="metric-sub">{status.get('remediated_at','—')}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TELEMETRY CHARTS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Live Telemetry</div>', unsafe_allow_html=True)
plt.style.use("dark_background")
a_mask = df["is_anomaly"] == 1

ch1, ch2 = st.columns(2)
for col, y_data, title, color, extra in [
    (ch1, df["cpu_usage_pct"], "CPU Utilization (%)",  "#3b82f6", True),
    (ch2, df["z_score"],       "Z-Score over Time",    "#06b6d4", False),
]:
    with col:
        fig, ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        ax.plot(df.index, y_data, color=color, linewidth=1.2, alpha=0.8)
        ax.fill_between(df.index, y_data, alpha=0.1, color=color)
        ax.scatter(df.index[a_mask], y_data[a_mask], color="#ef4444", s=60, zorder=5, label="Anomaly")
        if not extra:
            ax.axhline(y=2.5, color="#ef4444", linewidth=1, linestyle="--", label="Threshold (2.5σ)")
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=10, loc="left")
        ax.tick_params(colors="#475569", labelsize=8); ax.spines[:].set_color("#1e293b")
        ax.legend(facecolor="#0d1117", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)
        ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
ch3, ch4 = st.columns(2)

with ch3:
    fig, ax = plt.subplots(figsize=(4,3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_yticklabels(["Normal","Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_xlabel("Predicted", color="#64748b", fontsize=9)
    ax.set_ylabel("Actual",    color="#64748b", fontsize=9)
    ax.set_title("Confusion Matrix", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.spines[:].set_color("#1e293b")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with ch4:
    fig, ax = plt.subplots(figsize=(5,3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    colors = ["#3b82f6" if c >= 0 else "#ef4444" for c in coefs]
    bars   = ax.barh(features, coefs, color=colors, alpha=0.85)
    ax.set_title("Feature Importance", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#64748b", labelsize=9); ax.spines[:].set_color("#1e293b")
    ax.set_xlabel("Coefficient Value", color="#64748b", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5, axis="x")
    ax.axvline(x=0, color="#475569", linewidth=0.8)
    for bar, val in zip(bars, coefs):
        ax.text(val+(0.01 if val>=0 else -0.01), bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left" if val>=0 else "right",
                color="#94a3b8", fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# REMEDIATION ENGINE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Autonomous Remediation Engine</div>', unsafe_allow_html=True)

anomalies = df[df["is_anomaly"] == 1].copy()

if len(anomalies) == 0:
    st.markdown("""<div class="info-box">
        ✅ No anomalies in current window. No remediation triggered.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class="info-box">
        🤖 <b style="color:#e2e8f0">Auto-Remediation Active</b> — 
        {len(anomalies)} anomal{'y' if len(anomalies)==1 else 'ies'} detected. 
        Executing corrective actions scaled to z-score severity.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in anomalies.iterrows():
        ts  = row["Timestamp"].strftime("%Y-%m-%d %H:%M UTC") \
              if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
        sev, actions = remediate(row["z_score"], row["cpu_usage_pct"], ts)

        cls  = {"HIGH":"remediation-high","MEDIUM":"remediation-medium","LOW":"remediation-low"}.get(sev,"remediation-low")
        icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(sev,"⚪")
        acts = " &nbsp;·&nbsp; ".join([f"{'✅' if a['ok'] else '⚠️'} {a['action']}: {a['msg']}" for a in actions])

        st.markdown(f"""<div class="{cls}">
            {icon} <b>{ts}</b> &nbsp;|&nbsp; Severity: <b>{sev}</b> &nbsp;|&nbsp;
            CPU: <b>{row['cpu_usage_pct']:.2f}%</b> &nbsp;|&nbsp; Z-Score: <b>{row['z_score']:.3f}σ</b><br>
            <span style="opacity:0.8;font-size:0.75rem;">{acts}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        ℹ️ Instance status panel above reflects <b style="color:#e2e8f0">post-remediation state</b> 
        pulled live from AWS. Tags applied by CloudSentinel are visible in the EC2 console.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#1e293b;font-family:'Space Mono',monospace;font-size:0.7rem;padding:1rem 0;">
    CLOUDSENTINEL · MIT BENGALURU HACKATHON 2026 · AUTONOMOUS CLOUD COST INTELLIGENCE
</div>""", unsafe_allow_html=True)
