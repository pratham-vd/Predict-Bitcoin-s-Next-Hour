import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import scipy.stats as stats
from arch import arch_model
import warnings
import requests
import json
import os

from GBM import (
    get_binance_klines,
    rolling_entropy,
    simulate_mc
)

warnings.filterwarnings('ignore')

HISTORY_FILE = 'prediction_history.json'

st.set_page_config(page_title="BTC Prediction", layout="wide")


# Load Part A backtest metrics from the jsonl file produced by GBM.py.
@st.cache_data
def load_backtest_metrics():
    try:
        records = []
        with open('backtest_results.jsonl', 'r') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        return df['coverage'].mean(), df['width'].mean(), df['winkler'].mean(), len(df)
    except Exception:
        return None, None, None, None


# Fetch price data once per hour and cache it.
# Must be at module level so Streamlit recognises the same function across reruns.
@st.cache_data(ttl=3600)
def fetch_price_data():
    return get_binance_klines(symbol='BTCUSDT', interval='1h', days=30)


# Run the GBM model on the last 168 bars and return the 95% predicted range.
def predict_next_hour(prices_df, train_window=168, n_sims=10_000):
    try:
        prices = prices_df['close']
        log_ret = np.log(prices / prices.shift(1)).dropna()

        if len(log_ret) < train_window:
            st.error(f"Not enough data: {len(log_ret)} bars, need {train_window}")
            return None, None, None

        train_ret = log_ret.iloc[-train_window:]
        train_prices = prices.iloc[-train_window - 1:]

        try:
            am = arch_model(train_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
            res = am.fit(disp='off', rescale=False)
            sigma_fig = res.conditional_volatility / 100
            resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
            nu = max(4, stats.t.fit(resid.values, floc=0, fscale=1)[0])
        except Exception:
            sigma_fig = pd.Series([float(train_ret.std())] * len(train_ret))
            train_ret_std = float(train_ret.std())
            resid = train_ret / train_ret_std if train_ret_std > 0 else pd.Series(np.ones(len(train_ret)))
            nu = 4

        H_series = rolling_entropy(resid).iloc[:-1] if len(resid) > 60 else pd.Series(np.ones(len(resid)))
        M_series = train_ret.abs().rolling(60).mean() if len(train_ret) > 60 else pd.Series(np.ones(len(train_ret)))
        H_series = H_series.fillna(0)
        M_series = M_series.fillna(0)

        bar_sigma2 = float((sigma_fig ** 2).mean())
        info_filter = (H_series > float(H_series.mean())).astype(float) if (len(H_series) > 0 and H_series.max() > H_series.min()) else None

        try:
            rv5 = train_prices.rolling(5).var()
            rv20 = train_prices.rolling(20).var().replace(0, 1e-6)
            redundancy = 1 + 0.05 * np.log1p(rv5 / rv20)
            redundancy = redundancy.iloc[:-1].fillna(1.0)
        except Exception:
            redundancy = pd.Series(np.ones(len(train_ret)))

        S0 = float(prices.iloc[-1])
        mu = float(train_ret.mean())

        min_len = min(len(sigma_fig), len(H_series), len(M_series), len(redundancy))
        sigma_fig = sigma_fig.iloc[:min_len].reset_index(drop=True)
        H_series = H_series.iloc[:min_len].reset_index(drop=True)
        M_series = M_series.iloc[:min_len].reset_index(drop=True)
        redundancy = redundancy.iloc[:min_len].reset_index(drop=True)

        paths = simulate_mc(S0, mu, sigma_fig, H_series, M_series, bar_sigma2, nu,
                            n_sims=n_sims, n_steps=1,
                            info_filter=info_filter, redundancy=redundancy)
        S_t1 = paths[:, 1]

        return S0, float(np.percentile(S_t1, 2.5)), float(np.percentile(S_t1, 97.5))

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


# Load prediction history from file.
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


# Save prediction history to file.
def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save prediction history: {e}")


# Append the current hour's prediction. Skips if already saved this hour.
def save_current_prediction(history, predicted_for_hour, low, high):
    already_saved = any(r['predicted_for_hour'] == predicted_for_hour for r in history)

    if not already_saved:
        history.append({
            'predicted_for_hour': predicted_for_hour,
            'predicted_low': round(low, 2),
            'predicted_high': round(high, 2),
            'actual_price': None,
            'hit': None,
            'saved_at': datetime.now(timezone.utc).isoformat()
        })
        save_history(history)

    return history


# Fill actual prices for past predictions using already-fetched price data.
# No extra API calls needed.
def fill_in_actuals(history, df_prices):
    updated = False
    now_utc = datetime.now(timezone.utc)

    for record in history:
        if record['actual_price'] is not None:
            continue

        pred_hour = datetime.fromisoformat(record['predicted_for_hour'])
        if pred_hour.tzinfo is None:
            pred_hour = pred_hour.replace(tzinfo=timezone.utc)

        if pred_hour >= now_utc:
            continue

        pred_hour_naive = pd.Timestamp(pred_hour.replace(tzinfo=None))
        if pred_hour_naive in df_prices.index:
            actual = float(df_prices.loc[pred_hour_naive, 'close'])
            record['actual_price'] = round(actual, 2)
            record['hit'] = int(record['predicted_low'] <= actual <= record['predicted_high'])
            updated = True

    if updated:
        save_history(history)

    return history


# Convert history list to a display DataFrame. Most recent first.
def build_history_df(history):
    if not history:
        return pd.DataFrame()

    rows = []
    for r in history:
        hit_str = '✅' if r['hit'] == 1 else ('❌' if r['hit'] == 0 else '⏳ Pending')
        actual_str = f"${r['actual_price']:,.2f}" if r['actual_price'] is not None else '-'
        rows.append({
            'Predicted For (UTC)': r['predicted_for_hour'],
            'Lower ($)': f"${r['predicted_low']:,.2f}",
            'Upper ($)': f"${r['predicted_high']:,.2f}",
            'Width ($)': f"${round(r['predicted_high'] - r['predicted_low'], 2):,.2f}",
            'Actual ($)': actual_str,
            'Hit?': hit_str,
        })

    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def main():
    st.title("🪙 BTC/USDT Next-Hour Price Predictor")
    st.subheader("Backtested performance metrics of BTCUSDT over the past 30 days using a "
                 "Geometric Brownian Motion (GBM) model to predict the next hour's potential price range.")

    coverage, avg_width, winkler, n = load_backtest_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coverage", f"{coverage * 100:.2f}%" if coverage is not None else "N/A")
    with col2:
        st.metric("Avg Width", f"${avg_width:,.0f}" if avg_width is not None else "N/A")
    with col3:
        st.metric("Winkler Score", f"{winkler:,.0f}" if winkler is not None else "N/A")
    with col4:
        st.metric("Predictions", str(n) if n is not None else "N/A")

    st.divider()

    with st.spinner("Fetching last 500 closed hourly bars..."):
        df_all = fetch_price_data()

        if df_all is not None:
            now = datetime.now(timezone.utc)
            latest_closed_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            latest_closed_start = pd.Timestamp(latest_closed_start.replace(tzinfo=None))
            df_closed = df_all[df_all.index <= latest_closed_start]
            df_prices = df_closed.tail(500) if len(df_closed) >= 500 else None
        else:
            df_prices = None

    if df_prices is None or len(df_prices) < 500:
        st.error("Cannot fetch enough data. Please refresh.")
        return

    latest_close = float(df_prices['close'].iloc[-1])
    # Get live price from the currently forming bar using the same klines endpoint.
    # limit=1 with no startTime returns the current open (incomplete) bar.
    # Its close field reflects the latest traded price — no separate ticker API needed.
    current_price = latest_close
    try:
        for base_url in ["https://api.binance.com/api/v3/klines",
                         "https://api.binance.us/api/v3/klines"]:
            resp = requests.get(base_url,
                                params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 1},
                                timeout=5)
            if resp.status_code == 200 and resp.json():
                current_price = float(resp.json()[0][4])
                break
    except Exception:
        current_price = latest_close

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Closed Bar", f"${latest_close:,.2f}")
    with col2:
        st.metric("Current Live Price", f"${current_price:,.2f}")

    st.divider()

    with st.spinner("Running model on last 500 bars..."):
        # Seed tied to the latest bar's timestamp so every visitor in the same
        # hour gets the same prediction.
        seed = int(df_prices.index[-1].timestamp())
        np.random.seed(seed)
        S0, low, high = predict_next_hour(df_prices)

    if S0 is None:
        st.error("Prediction failed.")
        return

    st.subheader("Predicted next-hour price range based on the most recent 500 closed bars.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower (2.5%)", f"${low:,.2f}")
    with col2:
        st.metric("Upper (97.5%)", f"${high:,.2f}")
    with col3:
        st.metric("Range Width", f"${high - low:,.2f}")

    st.divider()

    st.subheader("Price Chart (Last 50 Hours)")
    last_50 = df_prices['close'].tail(50)
    next_hour_ts = df_prices.index[-1] + timedelta(hours=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_50.index, y=last_50.values,
        mode='lines', name='BTC Price',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[df_prices.index[-1], next_hour_ts], y=[high, high],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[df_prices.index[-1], next_hour_ts], y=[low, low],
        mode='lines', line=dict(width=0),
        fillcolor='rgba(68, 208, 92, 0.3)', fill='tonexty', name='95% Range'
    ))
    fig.add_vline(x=df_prices.index[-1], line_dash="dash", line_color="red")
    fig.update_layout(
        title="BTCUSDT with Predicted Range for Next Hour",
        xaxis_title="Time", yaxis_title="Price (USD)",
        hovermode='x unified', height=450, template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("📋 Prediction History")
    st.caption("Every visit saves the current prediction. Actuals are filled in automatically once that hour closes.")

    history = load_history()

    predicted_for_hour_iso = next_hour_ts.strftime('%Y-%m-%dT%H:%M:%S')
    history = save_current_prediction(history, predicted_for_hour_iso, low, high)
    history = fill_in_actuals(history, df_prices)

    resolved = [r for r in history if r['hit'] is not None]
    if resolved:
        live_coverage = sum(r['hit'] for r in resolved) / len(resolved)
        live_hits = sum(r['hit'] for r in resolved)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Live Predictions Made", len(history))
        with col2:
            st.metric("Resolved Predictions", len(resolved))
        with col3:
            st.metric("Live Coverage", f"{live_coverage * 100:.1f}% ({live_hits}/{len(resolved)})")
    else:
        st.info("No resolved predictions yet. Check back after the current hour closes.")

    hist_df = build_history_df(history)
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction history yet.")

    st.divider()
    st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
