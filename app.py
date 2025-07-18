import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime
import requests
from textblob import TextBlob

st.set_page_config(page_title="Bitcoin Predictor + News Sentiment", layout="centered")
st.title("🪙 Bitcoin Future Predictor + Internet Facts (News Sentiment)")

# 1. Price Prediction
start = st.date_input("Start date", datetime(2018, 1, 1))
end = st.date_input("End date", datetime.now())
if st.button("Fetch & Predict Bitcoin Price"):
    data = yf.download('BTC-USD', start=start, end=end)
    if not data.empty and data['Close'].notna().sum() > 2:
        st.subheader("Historical Bitcoin Prices")
        st.line_chart(data['Close'])
        df = data.reset_index()[['Date', 'Close']]
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        model = Prophet(daily_seasonality=True)
        with st.spinner("Training prediction model..."):
            model.fit(df)
            future = model.make_future_dataframe(periods=90)  # 3 months
            forecast = model.predict(future)
        st.subheader("Forecast (Next 3 Months)")
        forecast_chart = forecast[['ds', 'yhat']].set_index('ds').tail(90)
        st.line_chart(forecast_chart)
        st.write("Forecast Data (Last 10):")
        st.dataframe(forecast[['ds', 'yhat']].tail(10))
    else:
        st.warning("No data found for Bitcoin for these dates.")

# 2. Bitcoin News + Sentiment
st.header("📰 Latest Bitcoin News & Internet Sentiment")

NEWS_API_KEY = "78d8661731a840e3baddda94e3f3c90e"  # <-- replace with your key from https://newsapi.org

def get_bitcoin_news(api_key):
    url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json().get('articles', [])
    else:
        return []

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

if NEWS_API_KEY != "YOUR_NEWSAPI_KEY":
    news = get_bitcoin_news(NEWS_API_KEY)
    if news:
        sentiments = []
        st.subheader("Top 5 Latest News Headlines")
        for article in news:
            title = article['title']
            url = article['url']
            st.write(f"[{title}]({url})")
            sentiment = get_sentiment(title)
            sentiments.append(sentiment)
        avg_sent = sum(sentiments) / len(sentiments)
        st.write(f"**Average News Sentiment:** `{avg_sent:.2f}` (from -1 [very negative] to +1 [very positive])")
        if avg_sent > 0.15:
            st.success("General news sentiment is POSITIVE 🚀")
        elif avg_sent < -0.15:
            st.error("General news sentiment is NEGATIVE ⚠️")
        else:
            st.info("General news sentiment is NEUTRAL.")
    else:
        st.warning("Could not fetch news. Try again later.")
else:
    st.info("Get a free NewsAPI.org API key, paste it in the code, and reload.")

