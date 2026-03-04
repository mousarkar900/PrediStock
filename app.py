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
except Exception as e:
    st.error(f"Model file not found: {e}")
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

# ================= VALIDATE TICKER FIRST =================
@st.cache_data(ttl=3600)
def validate_ticker(symbol):
    """Check if the ticker symbol actually exists on Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # If ticker is invalid, yf returns minimal info or empty dict
        if not info or info.get("regularMarketPrice") is None:
            # Fallback check: try fetching 5 days of data
            test_df = ticker.history(period="5d")
            if test_df.empty:
                return False
        return True
    except Exception:
        return False

if not validate_ticker(stock):
    st.error(f"'{stock}' is not a valid stock symbol. Please check and retry.")
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
            auto_adjust=True  # This avoids Adj Close issues
        )

        # ------- KEY FIX: HANDLE MULTI-INDEX COLUMNS -------
        # yf.download() can return MultiIndex columns like ('Close', 'AAPL')
        # This happens especially when deployed on cloud servers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove any duplicate column names after flattening
        df = df.loc[:, ~df.columns.duplicated()]

        if df.empty:
            return None

        # ------- KEY FIX: VERIFY 'Close' COLUMN EXISTS -------
        if "Close" not in df.columns:
            st.error(f"Available columns: {list(df.columns)}")
            return None

        # Ensure the Close column has no issues
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)

        if df.empty:
            return None

        return df

    except Exception as e:
        st.error(f"Download error: {e}")
        return None

data = fetch_stock_data(stock)

if data is None or len(data) < 200:
    st.error(
        f"Insufficient data for '{stock}'. "
        f"Got {len(data) if data is not None else 0} rows, need at least 200."
    )
    st.stop()

# ================= DEBUG INFO (remove after fixing) =================
with st.expander("Debug Info"):
    st.write("Data shape:", data.shape)
    st.write("Columns:", list(data.columns))
    st.write("Column types:", data.dtypes.to_dict())
    st.write("First 3 rows:", data.head(3))

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
# ------- KEY FIX: Use data["Close"] instead of data.Close -------
# Attribute access can fail with certain column configurations
close_prices = data["Close"]

ma50 = close_prices.rolling(50).mean()
ma100 = close_prices.rolling(100).mean()
ma200 = close_prices.rolling(200).mean()

st.subheader("Moving Averages")
fig1 = plt.figure(figsize=(10, 5))
plt.plot(close_prices, label="Close")
plt.plot(ma50, label="MA50")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig1)

# ================= PREPARE DATA =================
# ------- KEY FIX: Ensure correct shape for scaler -------
close_values = close_prices.values.reshape(-1, 1).astype(np.float64)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_values)

x, y = [], []
for i in range(100, len(scaled_data)):
    x.append(scaled_data[i - 100:i])
    y.append(scaled_data[i])

x, y = np.array(x), np.array(y)

# ================= PREDICTION =================
predicted = model.predict(x)
predicted = scaler.inverse_transform(predicted)
y_actual = scaler.inverse_transform(y)

# ================= PLOT PREDICTION =================
st.subheader("Actual vs Predicted Prices")
fig2 = plt.figure(figsize=(10, 5))
plt.plot(y_actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
st.pyplot(fig2)

# ================= DOWNLOAD CSV =================
result_df = pd.DataFrame({
    "Actual": y_actual.flatten(),
    "Predicted": predicted.flatten()
})

st.download_button(
    "Download Predictions CSV",
    result_df.to_csv(index=False),
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

    if news and news.get("articles"):
        headlines = [a["title"] for a in news["articles"] if a.get("title")]
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
    else:
        st.warning("No news articles found")

except Exception as e:
    st.warning(f"News data unavailable: {e}")

st.success("Forecast Complete")