#  Autonomous Cloud Cost Intelligence Platform

> An autonomous system that monitors cloud infrastructure metrics, detects cost anomalies using machine learning, and lays the foundation for self-healing cloud environments without human intervention.

---

##  Problem Statement

Cloud infrastructure costs are notoriously difficult to monitor at scale. Unexpected spikes in CPU usage, runaway serverless functions, idle instances, and traffic surges can silently drain cloud budgets before any human notices. Existing solutions require manual threshold configuration and reactive human intervention.

**The Software** addresses this by building an autonomous cost intelligence pipeline that continuously watches cloud metrics, learns what "normal" looks like, flags genuine anomalies using ML, and is designed to automatically trigger corrective actions all in real time.

---

##  What It Does

```
Collect Metrics → Detect Anomalies → Decide Action → Show Impact
```

1. **Simulate / Ingest** cloud telemetry data (CPU, memory, network, cost, requests)
2. **Detect** anomalies using a Z-score based Logistic Regression model
3. **Decide** what corrective action to take (stop idle instance, limit function calls, clean storage)
4. **Visualise** cost trends, anomaly scores, and before/after impact on a real-time dashboard

---

## 🧠 ML Approach & Design Decisions

### Why Simulated Data?
Integrating a live AWS/GCP billing API introduces significant complexity authentication, SDK configuration, rate limits, and billing delays. For this proof of concept, we simulate realistic cloud telemetry using controlled statistical distributions. The architecture is identical to a live system; only the data source changes.

### Why Z-Score?
Z-score is a statistically grounded method to identify outliers data points that deviate significantly from the mean. Plotting Z-scores derived from our simulated dataset clearly reveals cost spikes, providing clean ground-truth labels to train our ML model.

### Why Linear Regression First?
We initially tested Linear Regression as a baseline classifier. While functional, it is fundamentally a regression tool not designed for binary classification of rare events like anomalies. It was retained as a documented baseline.

### Why Logistic Regression (Final Model)?
Logistic Regression is natively designed for binary classification. It outputs a probability between 0 and 1, handles the normal/anomaly classification task correctly, and critically its coefficients are fully interpretable. We can show exactly which cloud metrics (CPU, memory, network) contribute most to predicting a cost anomaly, giving actionable insight rather than a black-box alert. 

---

### Phase 2 — Live Cloud API Integration
- Connect to real AWS CloudWatch and apply the final Logistic Regression model
- Replace simulated data with live telemetry streams
- Retrain the model on real usage patterns

### Phase 3 — Autonomous Remediation
- When an anomaly is confirmed, automatically trigger corrective backend actions via cloud APIs:
  - Stop idle EC2 instances
  - Limit Lambda/Cloud Function concurrency
  - Clean unused S3/GCS storage volumes
  - Tag unreviewed assets for audit
- Full closed loop: **Detect → Decide → Act** with no human intervention

---

## 👥 Team & Roles

| Member | Role |
|---|---|
| Suryaansh and Rayaan| ML Engineering — model training, Z-score labelling, Logistic Regression |
| Shanmukesh | Cloud Integration — AWS/GCP API research, SDK setup, live data pipeline |
| Rohan | Cloud Integration — remediation engine design, backend action automation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Simulation | Python, NumPy, Pandas |
| Anomaly Detection | Scikit-learn (Logistic Regression, StandardScaler) |
| Statistical Analysis | SciPy (Z-score) |
| Dashboard | Streamlit, Matplotlib |
| Version Control | Git, GitHub |

---



##  Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
streamlit
```

---

