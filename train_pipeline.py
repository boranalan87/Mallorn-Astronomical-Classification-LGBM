import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

from feature_engineering import build_features


# =========================
# 1️⃣ LOAD DATA
# =========================
train_df = pd.read_csv("train_full_lightcurves.csv")
test_df  = pd.read_csv("test_full_lightcurves.csv")
targets  = pd.read_csv("train_targets_aligned.csv")

print("Train raw :", train_df.shape)
print("Test raw  :", test_df.shape)
print("Targets   :", targets.shape)


# =========================
# 2️⃣ FEATURE ENGINEERING
# =========================
train_feats = build_features(train_df)
test_feats  = build_features(test_df)

print("Train feats:", train_feats.shape)
print("Test feats :", test_feats.shape)


# =========================
# 3️⃣ MERGE TARGET
# =========================
train_final = train_feats.merge(
    targets[["object_id", "target"]],
    on="object_id",
    how="left"
)

X = train_final.drop(columns=["object_id", "target"])
y = train_final["target"]

print("Final train:", X.shape)


# =========================
# 4️⃣ SCALER (FIT SADECE TRAIN)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔐 FEATURE ORDER KİLİDİ
selected_columns = X.columns.tolist()

joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_columns, "selected_columns.pkl")

print("Scaler fit edildi.")


# =========================
# 5️⃣ TEST ALIGNMENT (KRİTİK)
# =========================
X_test = test_feats[selected_columns]
X_test_scaled = scaler.transform(X_test)

print("Train scaled:", X_scaled.shape)
print("Test scaled :", X_test_scaled.shape)


# =========================
# 6️⃣ IMBALANCE
# =========================
pos = (y == 1).sum()
neg = (y == 0).sum()
scale_pos_weight = neg / pos

print("\nClass imbalance:")
print("Pozitif:", pos)
print("Negatif:", neg)
print("scale_pos_weight:", round(scale_pos_weight, 2))


# =========================
# 7️⃣ MODEL
# =========================
model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled, y)
joblib.dump(model, "lgb_model.pkl")

print("\nModel eğitildi.")


# =========================
# 8️⃣ THRESHOLD TUNING
# =========================
proba = model.predict_proba(X_scaled)[:, 1]

thresholds = np.linspace(0.05, 0.5, 50)
f1s = []

for t in thresholds:
    preds = (proba >= t).astype(int)
    f1s.append(f1_score(y, preds))

best_t = thresholds[np.argmax(f1s)]
best_f1 = max(f1s)

print("\nBest threshold:", round(best_t, 3))
print("Best F1:", round(best_f1, 4))


# =========================
# 9️⃣ SAVE FEATURES
# =========================
train_feats.to_csv("features_train.csv", index=False)
test_feats.to_csv("features_test.csv", index=False)

print("\nPIPELINE TAMAMEN BİTTİ ")