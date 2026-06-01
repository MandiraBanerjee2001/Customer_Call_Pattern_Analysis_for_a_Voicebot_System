"""
Customer Call Pattern Analysis — Machine Learning
Predicts whether a call will succeed or fail
Models: Logistic Regression + Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, accuracy_score,
                                       precision_score, recall_score, f1_score)
from sklearn.inspection        import permutation_importance

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#2d3142', 'axes.labelcolor': '#e0e0e0',
    'axes.titlecolor': '#ffffff', 'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0', 'text.color': '#e0e0e0',
    'grid.color': '#2d3142', 'grid.linestyle': '--', 'grid.alpha': 0.5,
    'legend.facecolor': '#1a1d27',
})

# ══════════════════════════════════════════════════════════
# 1. LOAD & PREPARE FEATURES
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  ML — CALL SUCCESS PREDICTION")
print("=" * 60)

df = pd.read_csv('data/call_records_cleaned.csv')
print(f"\n📦 Loaded: {df.shape}")

# Binary target: Completed = 1, everything else = 0
df['target'] = (df['call_status'] == 'Completed').astype(int)
print(f"   Class balance  — 1 (Success): {df['target'].mean():.1%} | 0: {1-df['target'].mean():.1%}")

# ── Feature Selection ──────────────────────────────────
FEATURES = [
    'hour', 'day_of_week', 'time_bucket', 'is_weekend',
    'call_type', 'language', 'voicebot_flow', 'city',
    'sip_response_code', 'call_count', 'repeat_caller',
    'month',
]
TARGET = 'target'

df_ml = df[FEATURES + [TARGET]].copy()
df_ml.dropna(inplace=True)
print(f"   After NA drop  : {df_ml.shape}")

# ── Encode categoricals ────────────────────────────────
le_map = {}
for col in df_ml.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    le_map[col] = le

X = df_ml.drop(columns=[TARGET])
y = df_ml[TARGET]

# ── Train-Test Split (80/20 stratified) ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n   Train set      : {X_train.shape}")
print(f"   Test  set      : {X_test.shape}")

# ── Scale for LR ──────────────────────────────────────
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════
# 2. MODEL 1 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  MODEL 1: Logistic Regression")
print("─" * 60)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr   = lr.predict(X_test_s)
y_proba_lr  = lr.predict_proba(X_test_s)[:, 1]

print(f"\n  Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"\n{classification_report(y_test, y_pred_lr, target_names=['Fail','Success'])}")

cv_lr = cross_val_score(lr, X_train_s, y_train, cv=StratifiedKFold(5), scoring='f1')
print(f"  5-Fold CV F1: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")


# ══════════════════════════════════════════════════════════
# 3. MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  MODEL 2: Random Forest")
print("─" * 60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf   = rf.predict(X_test)
y_proba_rf  = rf.predict_proba(X_test)[:, 1]

print(f"\n  Accuracy  : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"\n{classification_report(y_test, y_pred_rf, target_names=['Fail','Success'])}")

cv_rf = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(5), scoring='f1')
print(f"  5-Fold CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")


# ══════════════════════════════════════════════════════════
# 4. SAVE MODELS
# ══════════════════════════════════════════════════════════
joblib.dump(lr,     'models/logistic_regression.pkl')
joblib.dump(rf,     'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_map, 'models/label_encoders.pkl')
print("\n✅ Models saved to models/")


# ══════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ML Model Evaluation — Call Success Prediction',
             fontsize=18, fontweight='bold', color='white', y=1.01)

ACCENT  = '#00d4ff'; SUCCESS = '#00ff88'; DANGER  = '#ff4757'
WARN    = '#ffa502'; PURPLE  = '#a29bfe'

# A) Confusion Matrix — LR
ax = axes[0, 0]
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fail','Success'], yticklabels=['Fail','Success'],
            linewidths=1, linecolor='#0f1117',
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Logistic Regression — Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# B) Confusion Matrix — RF
ax = axes[0, 1]
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Fail','Success'], yticklabels=['Fail','Success'],
            linewidths=1, linecolor='#0f1117',
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Random Forest — Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# C) ROC Curves
ax = axes[0, 2]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
ax.plot(fpr_lr, tpr_lr, color=ACCENT, linewidth=2,
        label=f'LR  AUC = {roc_auc_score(y_test, y_proba_lr):.3f}')
ax.plot(fpr_rf, tpr_rf, color=SUCCESS, linewidth=2,
        label=f'RF  AUC = {roc_auc_score(y_test, y_proba_rf):.3f}')
ax.plot([0,1],[0,1], 'w--', alpha=0.3, label='Random')
ax.set_title('ROC Curves', fontweight='bold')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(); ax.grid(True, alpha=0.3)

# D) Feature Importance — RF
ax = axes[1, 0]
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi)))
ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', alpha=0.9)
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance'); ax.grid(axis='x')

# E) Metrics Comparison Bar
ax = axes[1, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
lr_metrics = [accuracy_score(y_test, y_pred_lr), precision_score(y_test, y_pred_lr),
              recall_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr),
              roc_auc_score(y_test, y_proba_lr)]
rf_metrics = [accuracy_score(y_test, y_pred_rf), precision_score(y_test, y_pred_rf),
              recall_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf),
              roc_auc_score(y_test, y_proba_rf)]
x_pos = np.arange(len(metrics_names))
ax.bar(x_pos - 0.2, lr_metrics, 0.38, label='Logistic Regression', color=ACCENT, alpha=0.85)
ax.bar(x_pos + 0.2, rf_metrics, 0.38, label='Random Forest',       color=SUCCESS, alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(metrics_names, rotation=20, ha='right')
ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Metrics', fontweight='bold')
ax.legend(); ax.grid(axis='y')

# F) Prediction Probability Distribution
ax = axes[1, 2]
ax.hist(y_proba_rf[y_test == 1], bins=40, alpha=0.7, color=SUCCESS, label='Actual Success', edgecolor='none')
ax.hist(y_proba_rf[y_test == 0], bins=40, alpha=0.7, color=DANGER,  label='Actual Fail',    edgecolor='none')
ax.axvline(0.5, color='white', linestyle='--', linewidth=1.5, label='Decision Threshold')
ax.set_title('RF Prediction Probability Distribution', fontweight='bold')
ax.set_xlabel('Predicted Probability of Success')
ax.set_ylabel('Count'); ax.legend(); ax.grid(axis='y')

plt.tight_layout()
plt.savefig('reports/figures/fig6_ml_evaluation.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n📊 Saved: reports/figures/fig6_ml_evaluation.png")

# ══════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    'Metric':    metrics_names,
    'LR Score':  [f"{v:.4f}" for v in lr_metrics],
    'RF Score':  [f"{v:.4f}" for v in rf_metrics],
})
print(summary.to_string(index=False))
winner = "Random Forest" if np.mean(rf_metrics) > np.mean(lr_metrics) else "Logistic Regression"
print(f"\n🏆 Recommended Model: {winner}")
