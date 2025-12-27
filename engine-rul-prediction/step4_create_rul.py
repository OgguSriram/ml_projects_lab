import pandas as pd

# Column names
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21"
]

# Load training data
df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Data loaded ✅")

# 1. Find maximum cycle for each engine
max_cycle_per_engine = df.groupby("engine_id")["cycle"].max()

# 2. Create RUL column
df["RUL"] = df.apply(
    lambda row: max_cycle_per_engine[row["engine_id"]] - row["cycle"],
    axis=1
)

print("\nRUL column created ✅")

# 3. Show sample rows
print("\nSample rows with RUL:")
print(df[["engine_id", "cycle", "RUL"]].head(10))

# 4. Sanity check for one engine
engine_example = 1
engine_data = df[df["engine_id"] == engine_example]

print(f"\nSanity check for Engine {engine_example}:")
print(engine_data[["cycle", "RUL"]].head())
print(engine_data[["cycle", "RUL"]].tail())

# 5. RUL statistics
print("\nRUL statistics:")
print(df["RUL"].describe())

print("\nSTEP 4 COMPLETED ✅")
