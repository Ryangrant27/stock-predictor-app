import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime

start = st.date_input("Start date", datetime(2018, 1, 1))
today = datetime.today()
end = st.date_input("End date", today, max_value=today)

st.set_page_config(page_title="Bitcoin Predictor", layout="centered")
st.title("🪙 Bitcoin Future Predictor")

start = st.date_input("Start date", datetime(2018, 1, 1))
end = st.date_input("End date", datetime.now(), max_value=datetime.now())  # no future

future_days = st.slider("How many days into the future to predict?", 30, 365, 90)

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
            future = model.make_future_dataframe(periods=future_days)
            forecast = model.predict(future)
        st.subheader(f"Forecast (Next {future_days} Days)")
        forecast_chart = forecast[['ds', 'yhat']].set_index('ds').tail(future_days)
        st.line_chart(forecast_chart)
        st.write("Forecast Data (Last 10 Days):")
        st.dataframe(forecast[['ds', 'yhat']].tail(10))
    else:
        st.warning("No data found for Bitcoin for these dates.")
