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
try:
    model = load_model('Stock Predictions Model.keras')
except FileNotFoundError:
    st.error("Error: 'Stock Predictions Model.keras' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Page Configuration
st.set_page_config(page_title="🔮 PrediStock", layout="wide")

# === Custom Styling ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #f7f9fc; /* Default light background */
        color: #2c3e50; /* Default dark text */
        transition: background-color 0.3s, color 0.3s;
    }
    .stApp {
        background: linear-gradient(to bottom right, #fdfbfb, #ebedee); /* Default light gradient */
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
        margin-bottom: 0.4em;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1558d6;
        transform: scale(1.03);
    }

    /* Dark Mode Styles */
    @media (prefers-color-scheme: dark) {
        html, body, [class*="css"] {
            background-color: #1e1e1e; /* Dark background for dark mode */
            color: #dcdcdc; /* Light text for dark mode */
        }
        .stApp {
            background: linear-gradient(to bottom right, #2c2c2c, #1e1e1e); /* Dark gradient for dark mode */
        }
        h1, h2, h3 {
            color: #f0f0f0;
        }
        .stButton>button {
            background-color: #3a85e9;
            color: #e0e0e0;
        }
        .stButton>button:hover {
            background-color: #2d6cbd;
        }
        .dataframe {
            background-color: #333;
            color: #f0f0f0;
            border: 1px solid #555;
        }
        .dataframe th {
            background-color: #444;
            color: #f0f0f0;
        }
        .dataframe td {
            color: #dcdcdc;
        }
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
    try:
        raw_data = yf.download(stock, period="max")
        if not raw_data.empty:
            start_date_min = raw_data.index.min().date()
    except Exception as e:
        st.error(f"Error fetching data for {stock}: {e}")
        st.stop()

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

try:
    raw_data = yf.download(stock, period="max")
    if raw_data.empty:
        st.error("Stock symbol not found or data not available.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data for {stock}: {e}")
    st.stop()

required_days = max(200, 100 + int(len(raw_data) * 0.2) + 1)
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < required_days:
    st.error(f"Selected date range is too short. Minimum gap required is {required_days} days.")
    st.stop()

try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty or len(data) < required_days:
        st.error(f"Not enough data between selected dates. Try increasing the range (minimum ~{required_days} days).")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data for {stock} within the selected date range: {e}")
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
    plt.title(title, color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#1a1a2e')
    plt.xlabel('Time', color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.ylabel('Price', color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.legend(labelcolor='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.grid(True, linestyle='--', alpha=0.4, color='#555' if st.session_state.get('prefers_dark_mode') else '#ccc')
    plt.gca().spines['bottom'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.gca().spines['top'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.gca().spines['left'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.gca().spines['right'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.tick_params(axis='x', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    plt.tick_params(axis='y', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
    fig.patch.set_facecolor('#1e1e1e' if st.session_state.get('prefers_dark_mode') else 'white') # Set figure background
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
try:
    predict = model.predict(x)
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

st.markdown("## 📊 Actual vs Predicted with Confidence Band")
fig4 = plt.figure(figsize=(10, 5))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.fill_between(range(len(predict)), (predict-10).flatten(), (predict+10).flatten(), color='orange', alpha=0.2, label='Confidence Band')
plt.title('Predicted vs Actual with Confidence Range', color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#1a1a2e')
plt.xlabel('Time', color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.ylabel('Price', color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.legend(labelcolor='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.grid(True, linestyle='--', alpha=0.4, color='#555' if st.session_state.get('prefers_dark_mode') else '#ccc')
plt.gca().spines['bottom'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.gca().spines['top'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.gca().spines['left'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.gca().spines['right'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.tick_params(axis='x', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
plt.tick_params(axis='y', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
fig4.patch.set_facecolor('#1e1e1e' if st.session_state.get('prefers_dark_mode') else 'white') # Set figure background
st.pyplot(fig4)

st.download_button(
    label="📅 Download Prediction Data",
    data=pd.DataFrame({'Actual': y.flatten(), 'Predicted': predict.flatten()}).to_csv(index=False).encode('utf-8'),
    file_name=f"{stock}_predictions.csv",
    mime='text/csv'
)

st.markdown("## 📰 News Sentiment Analysis")
news_api_key = st.secrets.get("NEWS_API_KEY")
if news_api_key:
    newsapi = NewsApiClient(api_key=news_api_key)
    try:
        news = newsapi.get_everything(q=stock, language='en', sort_by='publishedAt', page_size=10)
    except Exception as e:
        st.error(f"Error fetching news data: {e}")
        news = {'status': 'error'} #  set news to error, so the rest of the code does not break.

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
        colors = ['green', 'gray', 'red']
        if st.session_state.get('prefers_dark_mode'):
            colors = ['#90EE90', '#808080', '#FF6961'] # Lighter shades for dark mode
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title(f"News Sentiment Summary for {stock}", color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#1a1a2e')
        plt.ylabel("Number of Headlines", color='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.gca().spines['bottom'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.gca().spines['top'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.gca().spines['left'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.gca().spines['right'].set_color('#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.tick_params(axis='x', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        plt.tick_params(axis='y', colors='#dcdcdc' if st.session_state.get('prefers_dark_mode') else '#2c3e50')
        fig_sentiment.patch.set_facecolor('#1e1e1e' if st.session_state.get('prefers_dark_mode') else 'white') # Set figure background
        st.pyplot(fig_sentiment)
    else:
        st.warning("No recent news articles found or API limit reached.")
else:
    st.warning("News API key is missing.  News sentiment analysis is disabled.")

with st.expander("📚 How to Interpret the Charts"):
    st.markdown("""
    - **MA50, MA100, MA200:** Moving averages help visualize trends in stock price over time.
    - **Predicted vs Actual:** Compare model predictions with real closing prices.
    - **Confidence Band:** Illustrates possible variance. A tighter band = more confident model.
    """)

st.markdown("---")
st.success("✅ Forecasting complete. Explore the charts and insights above.")
