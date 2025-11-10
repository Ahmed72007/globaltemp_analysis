# 🌍 Global Temperature & Climate Change Predictor
# ------------------------------------------------
# Requirements:
# pip install pandas numpy matplotlib requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load real NASA GISTEMP dataset
url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
print("📥 Downloading temperature data from NASA GISTEMP...")
df = pd.read_csv(url, skiprows=1)
print("✅ Data downloaded successfully.\n")

# Step 2: Clean and prepare the data
df = df.rename(columns={'Year': 'year'})

# Check that columns exist
if 'J-D' not in df.columns:
    raise ValueError("Column 'J-D' not found — please check dataset structure.")

# Keep useful columns: year, J-D, D-N, DJF, MAM, JJA, SON
columns_to_keep = ['year', 'J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON']
df = df[columns_to_keep]

# Convert all to numeric where possible
for col in columns_to_keep:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop invalid rows
df = df.dropna(subset=['year', 'J-D'])
df = df[df['year'] > 1800]  # remove header junk if any

print(f"📊 Cleaned dataset: {len(df)} yearly records from {int(df['year'].min())} to {int(df['year'].max())}\n")

# Step 3: Plot the global annual temperature change
plt.figure(figsize=(10,5))
plt.plot(df['year'], df['J-D'], color='orange', label='Annual Mean (J-D)')
plt.title("🌍 Global Average Temperature Anomaly (NASA GISTEMP)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Plot seasonal temperature trends
seasons = ['DJF', 'MAM', 'JJA', 'SON']
plt.figure(figsize=(10,5))
for s in seasons:
    plt.plot(df['year'], df[s], label=s)
plt.title("🌤️ Seasonal Temperature Trends (°C Anomaly)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Predict future temperatures using linear regression
x = df['year']
y = df['J-D']

coefficients = np.polyfit(x, y, 1)  # linear model
model = np.poly1d(coefficients)

# Predict 2025–2040
future_years = np.arange(2025, 2041)
future_predictions = model(future_years)

# Plot predictions
plt.figure(figsize=(10,5))
plt.scatter(x, y, color='blue', label='Observed Data')
plt.plot(x, model(x), color='red', label='Trend Line')
plt.plot(future_years, future_predictions, '--', color='green', label='Predicted (2025–2040)')
plt.title("📈 Global Temperature Trend and Future Prediction")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Print prediction results
print("🌡️ Future Temperature Predictions (°C Anomaly):")
for year, pred in zip(future_years, future_predictions):
    print(f"{year}: {pred:.3f}")

print(f"\n🔮 Predicted global temperature anomaly in 2030: {model(2030):.3f} °C")
