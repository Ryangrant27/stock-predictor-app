import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title="Bitcoin Future Predictor", layout="centered")
st.title("🪙 Bitcoin Future Predictor")

# Set today's date (so the end date can't be in the future)
today = date.today()

start = st.date_input("Start date", datetime(2018, 1, 1))
end = st.date_input("End date", today, max_value=today)  # Prevents picking future dates

future_days = st.slider("How many days into the future to predict?", 30, 365, 90)
if st.button("Fetch & Predict Bitcoin Price"):
    data = yf.download('BTC-USD', start=start, end=end)
    st.write("Debug: BTC-USD data (first 5 rows):")
    st.write(data.head())
    st.write("Debug: Number of rows:", data.shape[0])
    # Bulletproof error handling:
    if data.empty:
        st.warning("No data was returned from yfinance. Try a wider or different date range.")
    else:
        if 'Close' not in data.columns:
            st.warning("No 'Close' column found in data! Something is wrong with the returned DataFrame.")
        else:
            if data['Close'].notna().sum() <= 2:
                st.warning("Not enough valid closing price data in this range. Try a wider or different date range.")
            else:
                st.subheader("Historical Bitcoin Prices")
                st.line_chart(data['Close'])
                df = data.reset_index()
                date_col = 'Date' if 'Date' in df.columns else 'Datetime' if 'Datetime' in df.columns else df.columns[0]
                df = df[[date_col, 'Close']].copy()
                df.rename(columns={date_col: 'ds', 'Close': 'y'}, inplace=True)
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.dropna()
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

