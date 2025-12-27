import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21"
]

df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# -----------------------------
# CREATE RUL
# -----------------------------
max_cycle = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(
    lambda row: max_cycle[row["engine_id"]] - row["cycle"],
    axis=1
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
important_sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
window = 10

for sensor in important_sensors:
    df[f"{sensor}_roll_mean"] = (
        df.groupby("engine_id")[sensor]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{sensor}_roll_std"] = (
        df.groupby("engine_id")[sensor]
        .rolling(window, min_periods=1)
        .std()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )

df["cycle_norm"] = df["cycle"] / df.groupby("engine_id")["cycle"].transform("max")

# -----------------------------
# TRAIN–TEST SPLIT (ENGINE LEVEL)
# -----------------------------
engine_ids = df["engine_id"].unique()
train_engines, test_engines = train_test_split(
    engine_ids, test_size=0.2, random_state=42
)

train_df = df[df["engine_id"].isin(train_engines)]
test_df  = df[df["engine_id"].isin(test_engines)]

X_train = train_df.drop(columns=["engine_id", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "RUL"])
y_test = test_df["RUL"]

# -----------------------------
# TRAIN FINAL MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

print("Training final model...")
model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "rul_random_forest_model.pkl")
print("Model saved as rul_random_forest_model.pkl ✅")

# -----------------------------
# LOAD MODEL
# -----------------------------
loaded_model = joblib.load("rul_random_forest_model.pkl")
print("Model loaded successfully ✅")

# -----------------------------
# VERIFY PREDICTIONS
# -----------------------------
y_pred = loaded_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\nVerification metrics after loading:")
print(f"MAE : {mae:.2f} cycles")
print(f"R²  : {r2:.3f}")

print("\nSTEP 9 COMPLETED ✅")
