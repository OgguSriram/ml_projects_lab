import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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
# ENGINE-LEVEL SPLIT
# -----------------------------
engine_ids = df["engine_id"].unique()
train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df["engine_id"].isin(train_engines)]
test_df  = df[df["engine_id"].isin(test_engines)]

X_train = train_df.drop(columns=["engine_id", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "RUL"])
y_test = test_df["RUL"]

# -----------------------------
# TRAIN BASELINE MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("MODEL EVALUATION RESULTS 📊\n")

print(f"MAE  (Mean Absolute Error): {mae:.2f} cycles")
print(f"MSE  (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} cycles")
print(f"R²   (R-squared Score): {r2:.3f}")

print("\nSTEP 7 COMPLETED ✅")
