
# Energy Demand Forecasting using Linear Regression

from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Upload the CSV file
uploaded = files.upload()

# Load the CSV file
df = pd.read_csv('energy_data_sample.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofyear

# Plot the data
plt.plot(df['Date'], df['Energy_Consumption'], marker='o')
plt.title("Daily Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.grid(True)
plt.show()

# Prepare & train model
X = df[['Day']]
y = df['Energy_Consumption']
model = LinearRegression()
model.fit(X, y)

# Forecast next 7 days
future_days = np.array([[df['Day'].max() + i] for i in range(1, 8)])
future_preds = model.predict(future_days)

# Display predictions
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=7)
for date, pred in zip(future_dates, future_preds):
    print(f"{date.date()} → Predicted Demand: {pred:.2f} kWh")

# Plot forecast
plt.plot(df['Date'], y, label='Actual')
plt.plot(future_dates, future_preds, label='Forecast', linestyle='--', marker='x')
plt.legend()
plt.title("Energy Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.grid(True)
plt.show()
