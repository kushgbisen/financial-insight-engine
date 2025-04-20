# pages/1_📈_Index_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import traceback # For detailed error printing

# --- Constants ---
# Assuming 'models' folder is relative to the main app.py directory
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'nifty_high_lr_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'ohlc_scaler.joblib')
NIFTY_TICKER = "^NSEI"

# --- Helper Functions (Specific to this page) ---

@st.cache_resource # Cache resource loading for the session
def load_artifacts(model_path, scaler_path):
    """Loads the saved model and scaler."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Artifacts loaded successfully from {model_path} and {scaler_path}")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Ensure '{model_path}' and '{scaler_path}' exist relative to the main app directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

@st.cache_data(ttl=900) # Cache data for 15 minutes
def get_latest_nifty_data(ticker):
    """Fetches the most recent trading day's data for the ticker."""
    try:
        today = date.today()
        start_date = today - timedelta(days=7) # Look back 7 days
        stock_data = yf.download(ticker, start=start_date, end=today, auto_adjust=False, progress=False)
        if stock_data.empty:
            print(f"Could not fetch recent data for {ticker}.")
            return None
        latest_data = stock_data.iloc[-1:].copy()
        # Robust column renaming
        if isinstance(latest_data.columns, pd.MultiIndex):
            latest_data.columns = latest_data.columns.get_level_values(0)
        latest_data.columns = [str(col).lower() for col in latest_data.columns]
        latest_data = latest_data.loc[:,~latest_data.columns.duplicated()]
        print(f"Fetched latest data for date: {latest_data.index[0].date()}")
        return latest_data
    except Exception as e:
        print(f"Error fetching latest data: {e}")
        return None

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_historical_data(ticker, period="1y"):
    """Fetches historical data for the specified period."""
    try:
        print(f"Fetching historical data for {ticker}, period={period}...")
        stock_data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if stock_data.empty:
            st.warning(f"Could not fetch historical data for {ticker} and period {period}.")
            return None
        # Robust column renaming
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data.columns = [str(col).lower() for col in stock_data.columns]
        stock_data = stock_data.loc[:,~stock_data.columns.duplicated()]
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in stock_data.columns for col in required_cols):
             st.error("Downloaded historical data is missing required OHLC columns.")
             return None
        print(f"Successfully fetched {len(stock_data)} rows for period {period}.")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching historical data from yfinance: {e}")
        return None

def create_ohlc_chart(df, ticker_name="NIFTY 50"):
    """Creates an interactive Plotly Candlestick chart."""
    try:
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                           open=df['open'], high=df['high'],
                                           low=df['low'], close=df['close'],
                                           name=ticker_name)])
        fig.update_layout(title=f'{ticker_name} Price Chart', yaxis_title='Price',
                          xaxis_title='Date', xaxis_rangeslider_visible=False,
                          template='plotly_dark', height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# --- Page Title ---
st.header("📈 Nifty 50 Next Day High Prediction")

# --- Load Model and Scaler ---
lr_model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# --- Main Content ---
if lr_model and scaler:

    # --- Chart Section ---
    st.subheader("Historical Chart & Trend")
    timeframe_options = { "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                          "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max" }
    selected_timeframe_label = st.selectbox("Select Chart Timeframe:",
                                            options=list(timeframe_options.keys()), index=3)
    selected_period = timeframe_options[selected_timeframe_label]
    hist_data = get_historical_data(NIFTY_TICKER, period=selected_period)

    # Calculate and Display Trend Metric
    if hist_data is not None and not hist_data.empty:
        hist_data['sma_5'] = hist_data['close'].rolling(window=5).mean()
        latest_close = hist_data['close'].iloc[-1]
        latest_sma_5 = hist_data['sma_5'].iloc[-1]
        if pd.notna(latest_close) and pd.notna(latest_sma_5): # Check if values are valid
             delta_sma_5 = latest_close - latest_sma_5
             st.metric(label="Latest Close vs 5-Day SMA", value=f"{latest_close:.2f}",
                       delta=f"{delta_sma_5:.2f} ({'Above' if delta_sma_5 >= 0 else 'Below'} SMA)")
        else:
             st.info("Trend indicator N/A for the selected period.")

        # Display Chart
        fig = create_ohlc_chart(hist_data, ticker_name="Nifty 50")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not display historical chart.") # Error handled in create_ohlc_chart
    else:
        st.warning("Could not fetch historical data for chart/trend.") # Error handled in get_historical_data

    st.markdown("---") # Separator

    # --- Prediction Section ---
    st.subheader("Predict Next Day's High")
    st.markdown("Click the button below to fetch the latest Nifty 50 data and predict the *next trading day's High*.")
    fetch_predict_button = st.button("Fetch Latest Data & Predict High", type="primary")

    if fetch_predict_button:
        with st.spinner("Fetching latest Nifty 50 data..."):
            latest_ohlc = get_latest_nifty_data(NIFTY_TICKER)

        if latest_ohlc is not None:
            st.markdown("**Latest Available Nifty 50 Data Used:**")
            required_pred_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in latest_ohlc.columns for col in required_pred_cols):
                st.dataframe(latest_ohlc[required_pred_cols])
                latest_date = latest_ohlc.index[0].strftime('%Y-%m-%d')
                st.caption(f"Data fetched for: {latest_date}")
                input_data = latest_ohlc[['open', 'high', 'low', 'close']]
                # Scale and Predict
                try:
                    input_scaled = scaler.transform(input_data)
                    prediction = lr_model.predict(input_scaled)
                    predicted_high = prediction[0]
                    st.success(f"Predicted High for the next trading day: **{predicted_high:.2f}**")
                    st.caption(f"Prediction based on data from {latest_date}. Model R² on test data: ~0.998") # Add context
                except Exception as e:
                    st.error(f"Error during scaling or prediction: {e}")
                    traceback.print_exc() # Show detailed error in logs/terminal
            else:
                st.error("Latest fetched data is missing required columns for prediction (open, high, low, close).")
        else:
            st.error("Failed to fetch latest data for prediction. Please try again later.")

else:
    # Error message if model/scaler failed to load
    st.error("Prediction functionality is unavailable because model artifacts could not be loaded.")
