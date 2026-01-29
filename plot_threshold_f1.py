import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# =========================
# LOAD ARTEFACTS
# =========================
model = joblib.load("lgb_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_columns = joblib.load("selected_columns.pkl")

# =========================
# LOAD DATA
# =========================
X_raw = pd.read_csv("features_train.csv")
y = pd.read_csv("train_targets_aligned.csv")["target"]

# =========================
# FEATURE ALIGNMENT
# =========================
for col in selected_columns:
    if col not in X_raw.columns:
        X_raw[col] = 0.0

X_raw = X_raw[selected_columns]
X = scaler.transform(X_raw)

# =========================
# THRESHOLD SCAN
# =========================
proba = model.predict_proba(X)[:, 1]
thresholds = np.arange(0.05, 0.55, 0.05)

rows = []
for t in thresholds:
    preds = (proba >= t).astype(int)
    rows.append({
        "threshold": t,
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "positives": preds.sum()
    })

df_results = pd.DataFrame(rows)

best_row = df_results.loc[df_results["f1"].idxmax()]
best_threshold = best_row["threshold"]

# =========================
# CONFUSION MATRIX
# =========================
final_preds = (proba >= best_threshold).astype(int)
cm = confusion_matrix(y, final_preds)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(10, 6))
plt.plot(df_results["threshold"], df_results["f1"], marker="o", label="F1 Score")
plt.axvline(best_threshold, color="red", linestyle="--",
            label=f"Best Threshold = {best_threshold:.2f}")

plt.xlabel("Decision Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold Optimization – F1 Score Analysis (LightGBM)")
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.98, 1.01)

summary = (
    f"Samples: {len(y)}\n"
    f"Best Threshold: {best_threshold:.2f}\n"
    f"Best F1: {best_row['f1']:.4f}\n\n"
    f"TN: {cm[0,0]}  FP: {cm[0,1]}\n"
    f"FN: {cm[1,0]}  TP: {cm[1,1]}"
)

plt.figtext(0.5, -0.25, summary, ha="center",
            bbox=dict(facecolor="whitesmoke", alpha=0.95))

plt.tight_layout()
plt.savefig("threshold_f1_analysis_pro.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n OPTIMAL THRESHOLD REPORT")
print(df_results.round(4))
print("\n BEST CONFIGURATION")
print(best_row)