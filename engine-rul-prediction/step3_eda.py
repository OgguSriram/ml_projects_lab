import pandas as pd
import matplotlib.pyplot as plt

# Column names for CMAPSS FD001
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21"
]

# ✅ CONFIRMED CORRECT PATH
df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("EDA STARTED ✅")

print("\nNumber of engines:")
print(df["engine_id"].nunique())

engine_cycles = df.groupby("engine_id")["cycle"].max()
print("\nEngine life statistics (cycles):")
print(engine_cycles.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Plot one sensor for one engine
engine_example = 1
sensor_example = "s3"

engine_data = df[df["engine_id"] == engine_example]

plt.figure()
plt.plot(engine_data["cycle"], engine_data[sensor_example])
plt.xlabel("Cycle")
plt.ylabel(sensor_example)
plt.title(f"Sensor {sensor_example} vs Cycle (Engine {engine_example})")
plt.show()

print("\nEDA COMPLETED ✅")
