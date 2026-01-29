import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import joblib

df = pd.read_csv("oof_probabilities_v2.csv")

y_true = df["target"].values
oof_probs = df["oof_probability"].values

thresholds = np.linspace(0.01, 0.99, 990)
f1_scores = []

for t in thresholds:
    preds = (oof_probs >= t).astype(int)
    f1_scores.append(f1_score(y_true, preds))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best OOF F1: {best_f1:.4f}")

joblib.dump(best_threshold, "best_threshold.pkl")

with open("best_threshold.txt", "w") as f:
    f.write(str(best_threshold))
