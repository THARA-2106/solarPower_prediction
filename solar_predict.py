# solar_predict_enhanced.py
# Solar Energy Prediction Using Machine Learning
# Dataset: Plant 1 - Solar Power Generation and Weather Data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Load datasets
gen_data = pd.read_csv("Plant_1_Generation_Data.csv")
weather_data = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

print("✅ Generation Data loaded:", gen_data.shape)
print("✅ Weather Data loaded:", weather_data.shape)

# STEP 2: Convert DATE_TIME columns to datetime (dayfirst=True to match DD-MM-YYYY format)
gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], dayfirst=True, errors='coerce')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], dayfirst=True, errors='coerce')

# Drop rows with invalid DATE_TIME
gen_data.dropna(subset=['DATE_TIME'], inplace=True)
weather_data.dropna(subset=['DATE_TIME'], inplace=True)

# STEP 3: Merge datasets on DATE_TIME (inner join)
merged_df = pd.merge(gen_data, weather_data, on='DATE_TIME', how='inner')

if merged_df.empty:
    raise ValueError("❌ Merge resulted in an empty DataFrame. Check DATE_TIME formats or data overlap.")

# STEP 4: Keep useful columns only
df = merged_df[['DATE_TIME', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].copy()

# Rename columns
df.rename(columns={
    'DC_POWER': 'solar_output',
    'AMBIENT_TEMPERATURE': 'temperature',
    'MODULE_TEMPERATURE': 'module_temp',
    'IRRADIATION': 'irradiance'
}, inplace=True)

# STEP 5: Filter daytime data (solar_output > 0)
df = df[df['solar_output'] > 0].copy()

# STEP 6: Add time-based features
df['hour'] = df['DATE_TIME'].dt.hour
df['month'] = df['DATE_TIME'].dt.month

print("✅ Data after merging, renaming, and feature engineering:")
print(df.head())

# STEP 7: Clean data
df.dropna(inplace=True)

# STEP 8: Define features and target
X = df[['temperature', 'module_temp', 'irradiance', 'hour', 'month']]
y = df['solar_output']

if X.empty or y.empty:
    raise ValueError("❌ No data available after cleaning. Cannot train the model.")

# STEP 9: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 10: Train Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# STEP 11: Evaluate performance
print("\n📊 Model Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# STEP 12: Visualization (daytime only)
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Actual', color='blue')
plt.plot(y_pred[:100], label='Predicted', color='orange')
plt.title('Solar Energy Prediction - Random Forest (Daytime Only)')
plt.xlabel('Sample Index')
plt.ylabel('Solar Output (DC Power)')
plt.legend()
plt.show()

print("\n✅ Enhanced program completed successfully!")
