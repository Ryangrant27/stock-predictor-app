import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Stock Growth Predictor", layout="centered")
st.title("📈 Stock Growth Predictor (AI)")

# Input fields
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
start = st.date_input("Start date", datetime(2018, 1, 1))
end = st.date_input("End date", datetime.now())

if st.button("Fetch & Predict"):
    with st.spinner("Fetching data..."):
        data = yf.download(ticker, start=start, end=end)
    if not data.empty:
        st.subheader("Historical Closing Prices")
        st.line_chart(data['Close'])

        df = data.reset_index()[['Date', 'Close']]
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        model = Prophet(daily_seasonality=True)
        with st.spinner("Training prediction model..."):
            model.fit(df)
            future = model.make_future_dataframe(periods=180)  # 6 months
            forecast = model.predict(future)

        st.subheader("Forecast (Next 6 Months)")
        forecast_chart = forecast[['ds', 'yhat']].set_index('ds').tail(180)
        st.line_chart(forecast_chart)

        st.write("Forecast Data (Last 10):")
        st.dataframe(forecast[['ds', 'yhat']].tail(10))
    else:
        st.warning("No data found for this ticker or dates.")
