import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

api_key = "KV4DCWC488ZX1VPN"

st.title("Stock Price Graph Predictor Using Keras")

stock = st.text_input("Enter the Stock ID", "GOOG").upper()

ts = TimeSeries(key=api_key, output_format="pandas")
data, meta_data = ts.get_daily(symbol=stock, outputsize="full")

data = data.rename(columns={
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
})

google_data = data.iloc[::-1].copy()

model = load_model("stock_price_model.keras")

st.subheader("Stock Data")
st.write(google_data.tail(10))  

# Splitting Data
splitting_len = int(len(google_data) * 0.7)
train_data = google_data.iloc[:splitting_len]["Close"].values.reshape(-1, 1)
test_data = google_data.iloc[splitting_len:]["Close"].values.reshape(-1, 1)

# Scaling Data (fit on training data)
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

x_data, y_data = [], []
for i in range(100, len(test_scaled)):
    x_data.append(test_scaled[i-100:i])
    y_data.append(test_scaled[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Model Prediction
predictions = model.predict(x_data)

inverse_pred = scaler.inverse_transform(predictions)
inverse_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame({
    "Original_Test_Data": inverse_y_test.flatten(),
    "Predicted_Test_Data": inverse_pred.flatten()
}, index=google_data.index[splitting_len+100:])

# Plot Actual vs Predicted Data
st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15,5))
plt.plot(pd.concat([google_data["Close"][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Training Data", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)

# Predict Next 30 Days
future_days = 30
last_100_days = test_scaled[-100:]
future_predictions = []

for _ in range(future_days):
    next_input = np.array(last_100_days).reshape(1, 100, 1)
    next_pred = model.predict(next_input)
    future_predictions.append(next_pred[0, 0])
    last_100_days = np.append(last_100_days[1:], next_pred).reshape(-1, 1)

# Convert predictions back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_dates = pd.date_range(start=google_data.index[-1], periods=future_days + 1)[1:]

future_df = pd.DataFrame({"Predicted_Close": future_predictions.flatten()}, index=future_dates)

# Plot Predictions
st.subheader(f"Stock Price Prediction: Next 30 Days ({stock})")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(google_data.index[-100:], google_data["Close"].values[-100:], color="green", label="Last 100 Days Actual Price")
ax.plot(future_df.index, future_df["Predicted_Close"], color="red", label="Predicted Next 30 Days")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title(f"Stock Price Prediction for {stock} (Next 30 Days)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Display Predictions
st.subheader("Predicted Stock Prices for Next 30 Days")
st.write(future_df)
