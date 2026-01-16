import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from newsapi import NewsApiClient
from textblob import TextBlob
import datetime

# ================= PAGE CONFIG =================
st.set_page_config(page_title="PrediStock", layout="wide")

# ================= LOAD MODEL =================
try:
    model = load_model("Stock_Predictions_Model.keras")
except:
    st.error("Model file not found")
    st.stop()

# ================= HEADER =================
st.title("PrediStock – AI Stock Predictor")
st.caption("Deep Learning powered Stock Price Forecasting")

# ================= SIDEBAR =================
st.sidebar.header("Configuration")
stock = st.sidebar.text_input(
    "Enter Stock Symbol (e.g. AAPL, TSLA)"
).strip().upper()

if not stock:
    st.info("Enter a stock symbol to begin")
    st.stop()

# ================= FETCH DATA (YAHOO FINANCE) =================
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    try:
        df = yf.download(
            symbol,
            start="2010-01-01",
            end=datetime.date.today(),
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            return None
        return df
    except:
        return None

data = fetch_stock_data(stock)

if data is None or len(data) < 200:
    st.error("Invalid stock symbol or data not available")
    st.stop()

# ================= DATE RANGE =================
min_date = data.index.min().date()
max_date = data.index.max().date()

start_date = st.sidebar.date_input(
    "Start Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

data = data.loc[str(start_date):str(end_date)]

if len(data) < 150:
    st.error("Minimum 150 trading days required")
    st.stop()

# ================= DISPLAY DATA =================
st.subheader(f"Stock Data – {stock}")
st.dataframe(data.tail(200))

# ================= MOVING AVERAGES =================
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

st.subheader("Moving Averages")
fig1 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, label="Close")
plt.plot(ma50, label="MA50")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig1)

# ================= PREPARE DATA =================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.Close.values.reshape(-1, 1))

x, y = [], []
for i in range(100, len(scaled_data)):
    x.append(scaled_data[i - 100:i])
    y.append(scaled_data[i])

x, y = np.array(x), np.array(y)

# ================= PREDICTION =================
predicted = model.predict(x)
predicted = scaler.inverse_transform(predicted)
y = scaler.inverse_transform(y)

# ================= PLOT PREDICTION =================
st.subheader("Actual vs Predicted Prices")
fig2 = plt.figure(figsize=(10, 5))
plt.plot(y, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
st.pyplot(fig2)

# ================= DOWNLOAD CSV =================
df = pd.DataFrame({
    "Actual": y.flatten(),
    "Predicted": predicted.flatten()
})

st.download_button(
    "Download Predictions CSV",
    df.to_csv(index=False),
    f"{stock}_prediction.csv"
)

# ================= NEWS SENTIMENT =================
st.subheader("News Sentiment Analysis")

try:
    newsapi = NewsApiClient(api_key="93d80761b0fd4605986a09ff0a31f41e")
    news = newsapi.get_everything(
        q=stock,
        language="en",
        page_size=10
    )

    headlines = [a["title"] for a in news["articles"]]
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]

    labels = [
        "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"
        for s in sentiments
    ]

    sent_df = pd.DataFrame({
        "Headline": headlines,
        "Sentiment": labels
    })

    st.dataframe(sent_df)

    fig3 = plt.figure()
    sent_df["Sentiment"].value_counts().plot(kind="bar")
    st.pyplot(fig3)

except:
    st.warning("News data unavailable")

st.success("Forecast Complete")
