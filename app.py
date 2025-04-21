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

# Load model
model = load_model('Stock Predictions Model.keras')

# Page Configuration
st.set_page_config(page_title="🔮 PrediStock", layout="wide")

# === Custom Styling for Light Mode ===
st.markdown("""
    <style>
    /* Force Light Mode */
    body {
        background-color: #ffffff;  /* White background for light mode */
        color: #333333;  /* Dark text for contrast */
    }

    .stApp {
        background: #f7f9fc;  /* Light gray background for the app container */
    }

    .stButton>button {
        background-color: #4CAF50;  /* Light green button */
        color: white;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }

    .stTitle {
        color: #333333;  /* Dark text for titles */
    }

    .stHeader, .stSidebar {
        background-color: #f1f1f1;  /* Light background for header and sidebar */
    }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.title("📈 PrediStock")
st.caption("Leverage deep learning to forecast stock trends with smooth visuals and modern design.")

# === Sidebar Input ===
st.sidebar.header("🔧 Configuration")
stock = st.sidebar.text_input("Enter Stock Symbol", "")

# === Determine dynamic date limits ===
start_date = None
end_date = None
start_date_min = None
end_date_max = datetime.date.today()

if stock:
    raw_data = yf.download(stock, period="max")
    if not raw_data.empty:
        start_date_min = raw_data.index.min().date()

if start_date_min:
    start_date = st.sidebar.date_input("Start Date", value=None, min_value=start_date_min, max_value=end_date_max, key='start')
    end_date = st.sidebar.date_input("End Date", value=None, min_value=start_date_min, max_value=end_date_max, key='end')
else:
    st.warning("Enter a valid stock symbol to load date range.")

# === Display the About Section Until Inputs are Provided ===
if not (stock and start_date and end_date):
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        **Advanced Stock Market Price Predictor** is a powerful and user-friendly platform that leverages deep learning models to forecast stock prices.

        🔍 **What You Can Do Here:**
        - View historical stock data for any listed company
        - Analyze trends using Moving Averages (MA50, MA100, MA200)
        - Compare actual stock prices with AI-predicted values
        - Make smarter, data-driven decisions in the stock market
        """)

if not (stock and start_date and end_date):
    st.warning("Please enter a stock symbol and select both start and end dates to begin the analysis.")
    st.stop()

raw_data = yf.download(stock, period="max")
if raw_data.empty:
    st.error("Stock symbol not found or data not available.")
    st.stop()

required_days = max(200, 100 + int(len(raw_data) * 0.2) + 1)
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < required_days:
    st.error(f"Selected date range is too short. Minimum gap required is {required_days} days.")
    st.stop()

data = yf.download(stock, start=start_date, end=end_date)
if data.empty or len(data) < required_days:
    st.error(f"Not enough data between selected dates. Try increasing the range (minimum ~{required_days} days).")
    st.stop()

st.subheader(f"🗂️ Latest Stock Data: {stock}")
st.dataframe(data, use_container_width=True)

data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

def plot_ma_chart(title, *args):
    fig = plt.figure(figsize=(10, 5))
    for line, label, color in args:
        plt.plot(line, color, label=label)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    return fig

st.markdown("## 📊 Price vs MA50")
st.pyplot(plot_ma_chart("Price vs Moving Average 50", (ma_50, 'MA50', 'r'), (data.Close, 'Close Price', 'g')))

st.markdown("## 📊 Price vs MA50 vs MA100")
st.pyplot(plot_ma_chart("Price vs MA50 vs MA100", (ma_50, 'MA50', 'r'), (ma_100, 'MA100', 'b'), (data.Close, 'Close Price', 'g')))

st.markdown("## 📊 Price vs MA100 vs MA200")
st.pyplot(plot_ma_chart("Price vs MA100 vs MA200", (ma_100, 'MA100', 'r'), (ma_200, 'MA200', 'b'), (data.Close, 'Close Price', 'g')))

x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

st.markdown("## 📊 Actual vs Predicted with Confidence Band")
fig4 = plt.figure(figsize=(10, 5))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.fill_between(range(len(predict)), (predict-10).flatten(), (predict+10).flatten(), color='orange', alpha=0.2, label='Confidence Band')
plt.title('Predicted vs Actual with Confidence Range')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
st.pyplot(fig4)

st.download_button(
    label="📅 Download Prediction Data",
    data=pd.DataFrame({'Actual': y.flatten(), 'Predicted': predict.flatten()}).to_csv(index=False).encode('utf-8'),
    file_name=f"{stock}_predictions.csv",
    mime='text/csv'
)

st.markdown("## 📰 News Sentiment Analysis")
newsapi = NewsApiClient(api_key="93d80761b0fd4605986a09ff0a31f41e")
news = newsapi.get_everything(q=stock, language='en', sort_by='publishedAt', page_size=10)

if news['status'] == 'ok' and news['totalResults'] > 0:
    headlines = [article['title'] for article in news['articles']]
    sentiments = [TextBlob(title).sentiment.polarity for title in headlines]
    sentiment_labels = ['Positive' if s > 0 else 'Negative' if s < 0 else 'Neutral' for s in sentiments]

    sentiment_df = pd.DataFrame({
        'Headline': headlines,
        'Sentiment Score': sentiments,
        'Sentiment': sentiment_labels
    })

    st.dataframe(sentiment_df, use_container_width=True)

    fig_sentiment = plt.figure(figsize=(6, 4))
    sentiment_counts = sentiment_df['Sentiment'].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
    plt.title(f"News Sentiment Summary for {stock}")
    plt.ylabel("Number of Headlines")
    st.pyplot(fig_sentiment)
else:
    st.warning("No recent news articles found or API limit reached.")

with st.expander("📚 How to Interpret the Charts"):
    st.markdown("""
    - **MA50, MA100, MA200:** Moving averages help visualize trends in stock price over time.
    - **Predicted vs Actual:** Compare model predictions with real closing prices.
    - **Confidence Band:** Illustrates possible variance. A tighter band = more confident model.
    """)

st.markdown("---")
st.success("✅ Forecasting complete. Explore the charts and insights above.")
