import joblib
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1️⃣ MODEL VE FEATURE'LARI YÜKLE
# =========================
model = joblib.load("lgb_model.pkl")
selected_columns = joblib.load("selected_columns.pkl")

# =========================
# 2️⃣ FEATURE IMPORTANCE TABLOSU
# =========================
importances = model.feature_importances_

fi = pd.DataFrame({
    "feature": selected_columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_n = 20
fi_top = fi.head(top_n)

print(fi_top)

# =========================
# 3️⃣ GRAFİK
# =========================
plt.figure(figsize=(9, 6))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (LightGBM)")
plt.xlabel("Importance")

# =========================
# 4️⃣ SONUÇ BİLGİSİ (ALT YAZI)
# =========================
total_features = len(selected_columns)

plt.figtext(
    0.95, 0.01,
    f"Total Features Used: {total_features} | Top {top_n} Shown",
    ha="right",
    fontsize=9,
    style="italic"
)

# =========================
# 5️⃣ KAYDET + GÖSTER
# =========================
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()