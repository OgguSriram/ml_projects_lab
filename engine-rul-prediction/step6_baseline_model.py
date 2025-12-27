import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
# BASELINE MODEL
# -----------------------------
model = LinearRegression()

print("Training Linear Regression model...")
model.fit(X_train, y_train)

print("Model training completed ✅")

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

print("\nSample predictions (first 10):")
for i in range(10):
    print(f"Actual RUL: {y_test.iloc[i]:.1f} | Predicted RUL: {y_pred[i]:.1f}")

print("\nSTEP 6 COMPLETED ✅")
