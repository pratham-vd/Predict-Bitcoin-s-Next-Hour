import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import scipy.stats as stats
from arch import arch_model
import warnings
import requests

# Import from GBM.py
from GBM import (
    get_binance_klines,
    rolling_entropy,
    simulate_mc
)

warnings.filterwarnings('ignore')


# Page config
st.set_page_config(
    page_title="BTC Prediction",
    layout="wide"
)


# Prediction function
def predict_next_hour(prices_df, train_window=168, n_sims=10_000):
    # Generate BTC prediction for next hour using GBM logic from GBM.py
    # Inputs:
    #     prices_df: DataFrame from get_binance_klines() with 'close' column
    #     train_window: hours for training (default 168 = 1 week)
    #     n_sims: Monte Carlo simulations
    try:
        # Extract close prices
        prices = prices_df['close']
        
        # Calculate log returns (all available bars)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        
        if len(log_ret) < train_window:
            st.error(f"Not enough data: {len(log_ret)} bars, need {train_window}")
            return None, None, None
        
        # NO PEEKING: Use last train_window returns (bars i-train_window to i)
        # This is same as backtest: train_ret = log_ret.iloc[i - train_window:i]
        train_ret = log_ret.iloc[-train_window:]
        train_prices = prices.iloc[-train_window-1:]
        
        # Estimate volatility with Figarch
        try:
            am = arch_model(train_ret * 100, vol='FIGARCH', 
                           p=1, o=0, q=1, dist='studentst')
            res = am.fit(disp='off', rescale=False)
            sigma_fig = res.conditional_volatility / 100
            resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
            nu = max(4, stats.t.fit(resid.values, floc=0, fscale=1)[0])
        except:
            # Fallback to simple volatility
            sigma_fig = pd.Series([float(train_ret.std())] * len(train_ret))
            train_ret_std = float(train_ret.std())
            if train_ret_std > 0:
                resid = train_ret / train_ret_std
            else:
                resid = pd.Series(np.ones(len(train_ret)))
            nu = 4
        
        # Compute Entropy & Momentum 
        H_series = rolling_entropy(resid).iloc[:-1] if len(resid) > 60 else pd.Series(np.ones(len(resid)))
        M_series = train_ret.abs().rolling(60).mean() if len(train_ret) > 60 else pd.Series(np.ones(len(train_ret)))
        
        # Fill NaN
        H_series = H_series.fillna(0)
        M_series = M_series.fillna(0)
        
        bar_sigma2 = float((sigma_fig**2).mean())
        
        # Info filter
        if len(H_series) > 0 and H_series.max() > H_series.min():
            info_filter = (H_series > float(H_series.mean())).astype(float)
        else:
            info_filter = None
        
        # Redundancy (from GBM.py logic)
        try:
            rolling_var_5 = train_prices.rolling(5).var()
            rolling_var_20 = train_prices.rolling(20).var()
            rolling_var_20 = rolling_var_20.replace(0, 1e-6)
            redundancy = 1 + 0.1 * np.log1p(rolling_var_5 / rolling_var_20)
            redundancy = redundancy.iloc[:-1].fillna(1.0)
        except:
            redundancy = pd.Series(np.ones(len(train_ret)))
        
        # Predict next hour
        # Same as backtest: S0 = prices.iloc[i] (latest bar), predict bar i+1
        S0 = float(prices.iloc[-1])  # Latest bar close (NO PEEKING)
        mu = float(train_ret.mean())
        
        # Align series length
        min_len = min(len(sigma_fig), len(H_series), len(M_series), len(redundancy))
        sigma_fig = sigma_fig.iloc[:min_len].reset_index(drop=True)
        H_series = H_series.iloc[:min_len].reset_index(drop=True)
        M_series = M_series.iloc[:min_len].reset_index(drop=True)
        redundancy = redundancy.iloc[:min_len].reset_index(drop=True)
        
        # Use simulate_mc from GBM.py
        paths = simulate_mc(
            S0, mu, sigma_fig, H_series, M_series, bar_sigma2, nu,
            n_sims=n_sims, n_steps=1,
            info_filter=info_filter, redundancy=redundancy
        )
        
        # Extract 95% confidence interval
        S_t1 = paths[:, 1]
        low_95 = float(np.percentile(S_t1, 2.5))
        high_95 = float(np.percentile(S_t1, 97.5))
        
        return S0, low_95, high_95
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


# Main App
def main():
    st.subheader("Backtested performance metrics of BTCUSDT over the past 30 days using a Geometric Brownian Motion (GBM) model to predict the next hour's potential price range.")
    
    # Part-A Metrics (Backtest Results)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Coverage", "96.91%")
    with col2:
        st.metric("Avg Width", "$1,534")
    with col3:
        st.metric("Winkler Score", "1,900")
    with col4:
        st.metric("Predictions", "551")
    
    st.divider()
    
    # Fetch last 500 closed bars
    with st.spinner("Fetching last 500 closed hourly bars..."):
        df_all = get_binance_klines(symbol='BTCUSDT', interval='1h', days=30)
        
        if df_all is not None:
            # Get current time in UTC
            now = datetime.now(timezone.utc)
            
            # Latest CLOSED bar: started at (current hour - 1)
            # If now is 10:30, latest closed bar started at 09:00
            latest_closed_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
            # Remove timezone info to match df_all.index (which is naive UTC timestamps)
            # Then convert to pandas Timestamp
            latest_closed_start = pd.Timestamp(latest_closed_start.replace(tzinfo=None))
            
            # ONLY include bars that closed BEFORE latest_closed_start
            # This excludes the current incomplete bar (10:00-10:59)
            df_closed = df_all[df_all.index <= latest_closed_start]
            
            if len(df_closed) >= 500:
                # Keep only last 500 CLOSED bars
                df_prices = df_closed.tail(500)
            else:
                df_prices = None
        else:
            df_prices = None
    
    if df_prices is None or len(df_prices) < 500:
        st.error("Cannot fetch data. Please refresh.")
        return
    
    st.divider()
    
    # Display Prices
    # 1. Latest Closed Bar Close (Fixed)
    latest_close = float(df_prices['close'].iloc[-1])
    
    # 2. Current Live Price from Binance
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {'symbol': 'BTCUSDT'}
        response = requests.get(url, params=params, timeout=5)
        current_price = float(response.json()['price'])
    except:
        current_price = latest_close  # Fallback to latest closed bar
    
    # Display both in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Latest Closed Bar Close price", f"${latest_close:,.2f}")
    
    with col2:
        st.metric("Current Live Price", f"${current_price:,.2f}")
    
    st.divider()
    
    # Predict Next hour
    with st.spinner("Running model on last 500 bars..."):
        # Set random seed based on latest bar's timestamp
        # This ensures SAME prediction within same hour, DIFFERENT prediction across hours
        seed = int(df_prices.index[-1].timestamp())
        np.random.seed(seed)
        
        S0, low, high = predict_next_hour(df_prices)
    
    if S0 is None:
        st.error("Prediction failed.")
        return
    
    st.divider()
    
    # Predicted range for next hour
    st.subheader("Predicted next-hour price range based on the most recent 500 closed bars.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        ### 🔴 Lower (2.5%)
        ## ${low:,.2f}
        """)
    
    with col2:
        st.markdown(f"""
        ### 🟢 Upper (97.5%)
        ## ${high:,.2f}
        """)
    
    with col3:
        st.markdown(f"""
        ### Range Width
        ## ${high-low:,.2f}
        """)
    
    st.divider()
    
    # Chart
    st.subheader("Price Chart (Last 50 Hours)")
    
    last_50 = df_prices['close'].tail(50)
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=last_50.index,
        y=last_50.values,
        mode='lines',
        name='BTC Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Prediction zone (next hour)
    next_hour = df_prices.index[-1] + timedelta(hours=1)
    
    fig.add_trace(go.Scatter(
        x=[df_prices.index[-1], next_hour],
        y=[high, high],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[df_prices.index[-1], next_hour],
        y=[low, low],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 208, 92, 0.3)',
        fill='tonexty',
        name='95% Range'
    ))
    
    # Marker at latest bar
    fig.add_vline(x=df_prices.index[-1], line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="BTCUSDT with Predicted Range for Next Hour",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=450,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Footer
    st.info(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
