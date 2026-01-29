# model_train.py
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier


# =========================
# 0) AYARLAR
# =========================
RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_X_PATH = "features_train.csv"
TRAIN_Y_PATH = "train_targets_aligned.csv"

MODEL_OUT = "lgb_model.pkl"
COLS_OUT = "selected_columns.pkl"
THRESH_OUT = "best_threshold.pkl"


# =========================
# 1) VERİYİ OKU
# =========================
X = pd.read_csv(TRAIN_X_PATH)
y = pd.read_csv(TRAIN_Y_PATH)

# y tek kolon veya target isimli kolon olabilir
if isinstance(y, pd.DataFrame):
    if "target" in y.columns:
        y = y["target"].values
    else:
        y = y.iloc[:, 0].values
else:
    y = np.array(y)

# Kimlik kolonlarını çıkar
for id_col in ["object_id", "id", "ObjectID"]:
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

# Feature listesini sakla (submission için)
selected_columns = X.columns.tolist()

X_values = X.values


# =========================
# 2) CLASS IMBALANCE
# =========================
pos = int(np.sum(y == 1))
neg = int(np.sum(y == 0))
scale_pos_weight = neg / max(pos, 1)

print(f"X shape: {X_values.shape}")
print(f"y shape: {y.shape}")
print(f"Positive: {pos} | Negative: {neg}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


# =========================
# 3) MODEL PARAMETRELERİ
# =========================
model_params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=5,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.5,
    min_split_gain=0.0,
    scale_pos_weight=min(scale_pos_weight, 20),
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# =========================
# 4) CROSS-VALIDATION + OOF
# =========================
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_pred = np.zeros(len(y), dtype=float)
fold_aucs = []

best_fold_auc = -1.0
best_fold_model = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_values, y), 1):
    X_tr, X_va = X_values[tr_idx], X_values[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = LGBMClassifier(**model_params)

    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc"
    )

    p_va = clf.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = p_va

    auc = roc_auc_score(y_va, p_va)
    fold_aucs.append(auc)

    print(f"Fold {fold} ROC-AUC: {auc:.4f}")

    if auc > best_fold_auc:
        best_fold_auc = auc
        best_fold_model = clf

print(
    f"\nCV ROC-AUC: {np.mean(fold_aucs):.4f} "
    f"± {np.std(fold_aucs):.4f}"
)


# =========================
# 5) THRESHOLD OPTİMİZASYONU
# =========================
thresholds = np.linspace(0.01, 0.35, 300)

best_thr = 0.0
best_f1 = -1.0

for thr in thresholds:
    preds = (oof_pred >= thr).astype(int)
    f1 = f1_score(y, preds, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)

print(f"\nBest threshold: {best_thr:.4f}")
print(f"Best F1-score: {best_f1:.4f}")


# =========================
# 6) KAYDET
# =========================
joblib.dump(best_fold_model, MODEL_OUT)
joblib.dump(selected_columns, COLS_OUT)
joblib.dump(best_thr, THRESH_OUT)

print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY ")
