import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# =========================
# 1️⃣ VERİYİ YÜKLE
# =========================
df = pd.read_csv("train_full_lightcurves.csv")

# =========================
# 2️⃣ FEATURE FONKSİYONU
# =========================
def extract_features(group):
    features = {}

    flux = group["flux"].values
    time = group["mjd"].values
    flux_err = group["flux_err"].values

    features["flux_mean"] = np.mean(flux)
    features["flux_std"] = np.std(flux)
    features["flux_min"] = np.min(flux)
    features["flux_max"] = np.max(flux)
    features["flux_skew"] = skew(flux)
    features["flux_kurtosis"] = kurtosis(flux)

    features["snr"] = np.mean(flux) / (np.mean(flux_err) + 1e-6)
    features["num_points"] = len(flux)

    # 🔹 Trend (slope)
    if len(time) > 1:
        lr = LinearRegression()
        lr.fit(time.reshape(-1, 1), flux)
        features["flux_slope"] = lr.coef_[0]
    else:
        features["flux_slope"] = 0

    return pd.Series(features)

# =========================
# 3️⃣ OBJECT_ID BAZLI ÖZET
# =========================
features_df = df.groupby("object_id").apply(extract_features).reset_index()

# =========================
# 4️⃣ KAYDET
# =========================
features_df.to_csv("features_train_v2.csv", index=False)

print(" Feature engineering tamamlandı:", features_df.shape)