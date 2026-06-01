# 📞 Customer Call Pattern Analysis for a Voicebot System

<p align="center">
  <img src="reports/figures/fig1_kpi_dashboard.png" alt="Dashboard Preview" width="90%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Database-MySQL-blue?logo=mysql" />
  <img src="https://img.shields.io/badge/Visualisation-Plotly-purple?logo=plotly" />
  <img src="https://img.shields.io/badge/Records-50%2C000-green" />
</p>

---

## 📌 Project Overview

An end-to-end Data Science project that analyses **50,000 synthetic customer call records** from a voicebot system. The project identifies calling patterns, peak hours, call duration trends, repeat callers, and predictive signals for call success — enabling data-driven decisions around resource allocation, IVR optimisation, and customer experience improvement.

---

## ✨ Features

| Module | Description |
|---|---|
| 🗄️ **Dataset** | 50,000 realistic records with 14 fields, realistic distributions |
| 🗃️ **SQL** | Full MySQL schema + 12 analytical queries |
| 🔬 **EDA** | Data cleaning, outlier detection, feature engineering, 5 chart sets |
| 🤖 **ML** | Logistic Regression + Random Forest with 92.7% accuracy |
| 📊 **Dashboard** | Interactive Streamlit app with Plotly charts & dynamic filters |
| 💡 **Insights** | 22 actionable business insights from the analysis |

---

## 🗂️ Project Structure

```
Customer_Call_Pattern_Analysis/
│
├── data/
│   ├── call_records.csv             ← Raw generated dataset (50,000 rows)
│   └── call_records_cleaned.csv     ← Cleaned & feature-engineered dataset
│
├── notebooks/
│   └── analysis.ipynb               ← Jupyter notebook walkthrough
│
├── sql/
│   └── schema_and_queries.sql       ← MySQL schema + 12 analytical queries
│
├── src/
│   ├── generate_dataset.py          ← Synthetic data generator
│   ├── eda_analysis.py              ← EDA, cleaning, charts
│   └── ml_model.py                  ← ML model training & evaluation
│
├── dashboard/
│   └── app.py                       ← Streamlit interactive dashboard
│
├── reports/
│   ├── figures/                     ← All generated charts (PNG)
│   └── business_insights.txt        ← 22 business insights
│
├── models/
│   ├── random_forest.pkl            ← Trained RF model
│   ├── logistic_regression.pkl      ← Trained LR model
│   ├── scaler.pkl                   ← Feature scaler
│   └── label_encoders.pkl           ← Categorical encoders
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Customer-Call-Pattern-Analysis.git
cd Customer-Call-Pattern-Analysis
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline (in order)
```bash
# Step 1 — Generate the dataset
python src/generate_dataset.py

# Step 2 — Perform EDA & generate charts
python src/eda_analysis.py

# Step 3 — Train ML models
python src/ml_model.py

# Step 4 — Launch interactive dashboard
streamlit run dashboard/app.py
```

### 5. MySQL Setup (optional)
```bash
mysql -u root -p < sql/schema_and_queries.sql
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Total Records | 50,000 |
| Success Rate | 54.7% |
| Failure Rate | 18.4% |
| Avg Call Duration | 2.8 minutes |
| Peak Hour | 10:00 AM |
| Top City | Mumbai |
| Top Language | Hindi (30%) |
| ML Accuracy | **92.7%** |
| ML ROC-AUC | **0.921** |

---

## 📈 Visual Outputs

| Figure | Description |
|---|---|
| `fig1_kpi_dashboard.png` | 6-panel KPI overview |
| `fig2_heatmap_hour_dow.png` | Hour × Day heatmap |
| `fig3_duration_analysis.png` | Duration boxplot + flow avg |
| `fig4_success_rate.png` | Monthly & city success rates |
| `fig5_behaviour_flow.png` | Customer frequency + flow scatter |
| `fig6_ml_evaluation.png` | Model evaluation (CM, ROC, FI) |

---

## 🧠 ML Model Summary

Two models trained to predict call success:

```
Features used (12): hour, day_of_week, time_bucket, is_weekend,
call_type, language, voicebot_flow, city, sip_response_code,
call_count, repeat_caller, month

Target: is_success (1 = Completed, 0 = all others)
```

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 92.74% | 0.9378 | 0.9193 |
| **Random Forest** ✅ | **92.74%** | **0.9378** | **0.9207** |

---

## 💡 Sample Business Insights

1. **Peak hour is 10:00 AM** — schedule maximum agents during this window.
2. **61% of customers call more than 3 times** — indicates unresolved issues; invest in first-call resolution.
3. **Failed calls average only 0.4 min** — immediate drops point to SIP gateway capacity issues.
4. **Hindi dominates at 30%** — ensure NLU models are strongest in Hindi.
5. **SIP 503 errors occurred 1,553 times** — infrastructure scaling is required.

*(22 insights total in `reports/business_insights.txt`)*

---

## 🔮 Future Improvements

- [ ] Real-time data ingestion via Apache Kafka
- [ ] Sentiment analysis on call transcripts
- [ ] Prophet-based call volume forecasting
- [ ] SHAP explainability for the ML models
- [ ] Docker containerisation for easy deployment
- [ ] Integration with CRM/helpdesk systems (Salesforce, Freshdesk)
- [ ] A/B testing framework for voicebot flows
- [ ] Anomaly detection for SIP failures

---

## 🛠️ Technologies Used

`Python 3.11` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`
`Streamlit` · `Plotly` · `MySQL` · `Faker` · `Joblib` · `Jupyter`

---

## 👤 Author

**[Your Name]**
📧 your.email@example.com
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — free to use, share, and modify with attribution.
