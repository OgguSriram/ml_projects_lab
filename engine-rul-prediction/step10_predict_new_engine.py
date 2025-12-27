import pandas as pd
import numpy as np
import joblib

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = joblib.load("rul_random_forest_model.pkl")
print("Model loaded ✅")

# -----------------------------
# SIMULATE NEW ENGINE DATA
# (last 10 cycles of a running engine)
# -----------------------------
data = {
    "cycle": [120,121,122,123,124,125,126,127,128,129],
    "setting_1": [0.5]*10,
    "setting_2": [0.3]*10,
    "setting_3": [0.1]*10,
    "s1":  np.random.normal(500, 2, 10),
    "s2":  np.random.normal(640, 2, 10),
    "s3":  np.random.normal(1580, 3, 10),
    "s4":  np.random.normal(1400, 3, 10),
    "s5":  np.random.normal(14, 0.2, 10),
    "s6":  np.random.normal(21, 0.2, 10),
    "s7":  np.random.normal(550, 2, 10),
    "s8":  np.random.normal(2380, 5, 10),
    "s9":  np.random.normal(9050, 10, 10),
    "s10": np.random.normal(1.3, 0.05, 10),
    "s11": np.random.normal(47, 0.5, 10),
    "s12": np.random.normal(520, 3, 10),
    "s13": np.random.normal(2380, 5, 10),
    "s14": np.random.normal(8140, 10, 10),
    "s15": np.random.normal(8.4, 0.1, 10),
    "s16": np.random.normal(0.03, 0.005, 10),
    "s17": np.random.normal(392, 2, 10),
    "s18": np.random.normal(2388, 5, 10),
    "s19": np.random.normal(100, 1, 10),
    "s20": np.random.normal(39, 0.5, 10),
    "s21": np.random.normal(23, 0.3, 10),
}

df_new = pd.DataFrame(data)

# -----------------------------
# FEATURE ENGINEERING (same as training)
# -----------------------------
important_sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
window = 10

for sensor in important_sensors:
    df_new[f"{sensor}_roll_mean"] = df_new[sensor].rolling(window, min_periods=1).mean()
    df_new[f"{sensor}_roll_std"]  = df_new[sensor].rolling(window, min_periods=1).std().fillna(0)

# Normalize cycle (assume max life ~300 for new engine)
df_new["cycle_norm"] = df_new["cycle"] / 300

# -----------------------------
# USE LAST ROW FOR PREDICTION
# -----------------------------
latest_sample = df_new.iloc[-1:]

predicted_rul = model.predict(latest_sample)[0]

print("\n🔮 PREDICTED REMAINING USEFUL LIFE")
print(f"Estimated RUL: {predicted_rul:.2f} cycles")

print("\nSTEP 10 COMPLETED ✅")
