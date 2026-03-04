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

# ================= VALIDATE TICKER =================
@st.cache_data(ttl=3600)
def validate_ticker(symbol):
    """Check if the ticker symbol actually exists on Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        test_df = ticker.history(period="5d")
        return not test_df.empty
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
            auto_adjust=True
        )

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove duplicate column names
        df = df.loc[:, ~df.columns.duplicated()]

        if df.empty:
            return None

        if "Close" not in df.columns:
            st.error(f"Available columns: {list(df.columns)}")
            return None

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
close_prices = data["Close"]

ma50  = close_prices.rolling(50).mean()
ma100 = close_prices.rolling(100).mean()
ma200 = close_prices.rolling(200).mean()

st.subheader("Moving Averages")
fig1 = plt.figure(figsize=(10, 5))
plt.plot(close_prices, label="Close")
plt.plot(ma50,  label="MA50")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()
st.pyplot(fig1)

# ================= PREPARE DATA =================
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
y_actual  = scaler.inverse_transform(y)

# ================= PLOT PREDICTION =================
st.subheader("Actual vs Predicted Prices")
fig2 = plt.figure(figsize=(10, 5))
plt.plot(y_actual,  label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
st.pyplot(fig2)

# ================= DOWNLOAD CSV =================
result_df = pd.DataFrame({
    "Actual":    y_actual.flatten(),
    "Predicted": predicted.flatten()
})

st.download_button(
    "Download Predictions CSV",
    result_df.to_csv(index=False),
    f"{stock}_prediction.csv"
)

# ================= NEWS SENTIMENT =================
st.subheader("News Sentiment Analysis")

def get_news_api_key():
    """Try to get API key from secrets, then fallback to hardcoded."""
    try:
        return st.secrets["NEWS_API_KEY"]
    except Exception:
        return "93d80761b0fd4605986a09ff0a31f41e"

def fetch_news(symbol):
    """Fetch news articles for a given stock symbol."""
    try:
        api_key = get_news_api_key()
        newsapi = NewsApiClient(api_key=api_key)

        # Try full company name search first, then ticker
        news = newsapi.get_everything(
            q=symbol,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )

        articles = news.get("articles", []) if news else []

        # Filter out removed/deleted articles
        articles = [
            a for a in articles
            if a.get("title") and a["title"] != "[Removed]"
        ]

        return articles

    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

def analyze_sentiment(articles):
    """Run TextBlob sentiment on article headlines."""
    headlines  = [a["title"] for a in articles]
    urls       = [a.get("url", "#") for a in articles]
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]

    labels = [
        "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
        for s in sentiments
    ]

    return pd.DataFrame({
        "Headline":        headlines,
        "Sentiment":       labels,
        "Polarity Score":  [round(s, 3) for s in sentiments],
        "URL":             urls
    })

articles = fetch_news(stock)

if articles:
    sent_df = analyze_sentiment(articles)
    st.dataframe(sent_df[["Headline", "Sentiment", "Polarity Score"]])

    # Sentiment bar chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment Distribution**")
        fig3, ax3 = plt.subplots()
        counts = sent_df["Sentiment"].value_counts()
        colors = {
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral":  "#95a5a6"
        }
        bar_colors = [colors.get(c, "#95a5a6") for c in counts.index]
        counts.plot(kind="bar", ax=ax3, color=bar_colors)
        ax3.set_xlabel("")
        ax3.set_ylabel("Count")
        ax3.set_title(f"News Sentiment for {stock}")
        plt.xticks(rotation=0)
        st.pyplot(fig3)

    with col2:
        st.markdown("**Overall Sentiment Score**")
        avg_score = sent_df["Polarity Score"].mean()
        if avg_score > 0.05:
            overall = "🟢 Positive"
        elif avg_score < -0.05:
            overall = "🔴 Negative"
        else:
            overall = "🟡 Neutral"
        st.metric("Average Polarity", f"{avg_score:.3f}", overall)

else:
    st.warning("No news articles found for this stock. Try a different symbol or check your API key.")

st.success("Forecast Complete")
