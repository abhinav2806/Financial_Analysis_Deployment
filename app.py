import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

# Function for Hyperparameter Tuning
def tune_sarima(data):
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for P in range(3):
                    for D in range(2):
                        for Q in range(3):
                            seasonal_order = (P, D, Q, 12)
                            try:
                                model = SARIMAX(data['Close'], order=(p, d, q), seasonal_order=seasonal_order, enforce_stationarity=False)
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = seasonal_order
                            except:
                                continue
    return best_order, best_seasonal_order

# App title 
st.title('Stock Price Prediction App')
st.subheader('Time Series Forecasting with SARIMA')
st.subheader('All prices are in USD')

# Get data
ticker = st.text_input("Enter stock ticker (e.x. Apple is AAPL)", "AAPL")

if st.button("Predict"):
    data = yf.download(ticker, start='2021-10-01', end='2023-09-01', progress=False)  
    data.reset_index(inplace=True)

    # Hyperparameter Tuning
    best_order, best_seasonal_order = tune_sarima(data)

    # Fit SARIMA model with best parameters
    model = SARIMAX(data['Close'], order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False)
    results = model.fit(disp=False)

    # Forecast
    forecast_steps = 30  # Number of steps to forecast
    forecast = results.get_forecast(steps=forecast_steps)

    # Generate future dates
    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps+1)]

    # Display forecasted values
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': forecast_values,
        'Lower CI': conf_int.iloc[:, 0].values,
        'Upper CI': conf_int.iloc[:, 1].values
    })

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(data['Date'], data['Close'], label='Actual Price', color='blue')
    plt.plot(forecast_df['Date'], forecast_df['Predicted Price'], color='red', label='Predicted Price')
    plt.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (Close)')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.legend()
    st.pyplot(plt)

    # Display forecasted values in a table
    st.subheader("Predicted Stock Prices")
    st.write(forecast_df)

    # Set y-axis domain to start from the minimum predicted price
    y_min = forecast_values.min()
    y_max = forecast_values.max()

    # Create a separate chart for predicted prices with adjusted y-axis domain
    predicted_chart = alt.Chart(forecast_df).mark_line(color='red').encode(
        x=alt.X('Date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('Predicted Price:Q', axis=alt.Axis(title='Predicted Price'), scale=alt.Scale(domain=[y_min, y_max]))
    ).properties(
        width=700,
        height=500
    )

    st.subheader("Predicted Prices Chart")
    st.altair_chart(predicted_chart)

else:
    st.error("This ticker may not be listed on NYSE. Please enter another ticker.")