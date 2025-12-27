import pandas as pd
from sklearn.model_selection import train_test_split

# Column names
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21"
]

# Load data
df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Create RUL again (same logic as Step 4)
max_cycle = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(
    lambda row: max_cycle[row["engine_id"]] - row["cycle"],
    axis=1
)

print("Data loaded with RUL ✅")

# -----------------------------
# ENGINE-LEVEL TRAIN-TEST SPLIT
# -----------------------------

# Get unique engines
engine_ids = df["engine_id"].unique()

# Split engines: 80% train, 20% test
train_engines, test_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

# Create train and test datasets
train_df = df[df["engine_id"].isin(train_engines)]
test_df  = df[df["engine_id"].isin(test_engines)]

print("\nTrain-Test split completed ✅")

# -----------------------------
# FEATURES & TARGET
# -----------------------------

# Drop engine_id and RUL separation
X_train = train_df.drop(columns=["engine_id", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "RUL"])
y_test = test_df["RUL"]

# -----------------------------
# PRINT SHAPES (VERY IMPORTANT)
# -----------------------------

print("\nTraining set shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest set shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nSTEP 5 COMPLETED ✅")
