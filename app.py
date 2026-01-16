import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
from newsapi import NewsApiClient
from textblob import TextBlob

# -------------------- Page Config --------------------
st.set_page_config(page_title="PrediStock", layout="wide")

# -------------------- Safe Yahoo Fetch Function --------------------
def fetch_stock_data(symbol, start=None, end=None, period=None):
    return yf.download(
        symbol,
        start=start,
        end=end,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=False
    )

# -------------------- Load Model --------------------
try:
    model = load_model("Stock_Predictions_Model.keras")
except:
    st.error("Model file not found")
    st.stop()

# -------------------- Header --------------------
st.title("PrediStock – AI Stock Predictor")
st.caption("Deep Learning powered Stock Price Forecasting")

# -------------------- Sidebar --------------------
st.sidebar.header("Configuration")
stock = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)").strip().upper()

if not stock:
    st.info("Enter a stock symbol to begin")
    st.stop()

# -------------------- Fetch Full History --------------------
raw = fetch_stock_data(stock, period="max")

if raw is None or len(raw) < 5:
    st.sidebar.error(
        "Unable to fetch stock data right now. "
        "This may be due to Yahoo Finance or Streamlit Cloud restrictions."
    )
    st.stop()

min_date = raw.index.min().date()
max_date = datetime.date.today()

st.sidebar.info(f"Select a date range (from {min_date} to {max_date})")

# -------------------- Date Inputs --------------------
start_date = st.sidebar.date_input(
    "Start Date", min_value=min_date, max_value=max_date
)
end_date = st.sidebar.date_input(
    "End Date", min_value=min_date, max_value=max_date
)

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# -------------------- Fetch Selected Range --------------------
data = fetch_stock_data(stock, start=start_date, end=end_date)

if data is None or len(data) < 150:
    st.error("Please select a larger date range (minimum 150 days)")
    st.stop()

# -------------------- Display Data --------------------
st.subheader(f"Stock Data – {stock}")
st.dataframe(data)

# -------------------- Moving Averages --------------------
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

# -------------------- Plot Moving Averages --------------------
st.subheader("Moving Averages")
fig1 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, label="Close")
plt.plot(ma50, label="MA50")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig1)

# -------------------- Prepare Data --------------------
train = data.Close[:int(len(data) * 0.8)]
test = data.Close[int(len(data) * 0.8):]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

past100 = train_scaled[-100:]
test_scaled = scaler.transform(test.values.reshape(-1, 1))
final_test = np.concatenate((past100, test_scaled))

x, y = [], []

for i in range(100, len(final_test)):
    x.append(final_test[i - 100:i])
    y.append(final_test[i])

x, y = np.array(x), np.array(y)

# -------------------- Prediction --------------------
predicted = model.predict(x)
predicted = scaler.inverse_transform(predicted)
y = scaler.inverse_transform(np.array(y).reshape(-1, 1))

# -------------------- Plot Prediction --------------------
st.subheader("Actual vs Predicted Stock Prices")
fig2 = plt.figure(figsize=(10, 5))
plt.plot(y, label="Actual")
plt.plot(predicted, label="Predicted")
plt.fill_between(
    range(len(predicted)),
    (predicted - 10).flatten(),
    (predicted + 10).flatten(),
    alpha=0.2
)
plt.legend()
st.pyplot(fig2)

# -------------------- Download Predictions --------------------
df = pd.DataFrame({
    "Actual": y.flatten(),
    "Predicted": predicted.flatten()
})

st.download_button(
    "Download Predictions CSV",
    df.to_csv(index=False),
    f"{stock}_prediction.csv"
)

# -------------------- News Sentiment --------------------
st.subheader("News Sentiment Analysis")

newsapi = NewsApiClient(api_key="93d80761b0fd4605986a09ff0a31f41e")
news = newsapi.get_everything(q=stock, language="en", page_size=10)

headlines = [a["title"] for a in news["articles"]]
sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
labels = ["Positive" if s > 0 else "Negative" if s < 0 else "Neutral" for s in sentiments]

sent_df = pd.DataFrame({
    "Headline": headlines,
    "Sentiment": labels
})

st.dataframe(sent_df)

fig3 = plt.figure()
sent_df["Sentiment"].value_counts().plot(kind="bar")
st.pyplot(fig3)

st.success("Forecast Complete")
