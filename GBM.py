import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import requests
from datetime import datetime, timedelta, timezone
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import json
import warnings

# Suppress arch model scaling warnings
warnings.filterwarnings('ignore', category=UserWarning, module='arch')


# 1. Fetch BTCUSDT 1H Data from Binance
# Returns DataFrame with OHLCV data, indexed by timestamp, No API key needed fully public.
def get_binance_klines(symbol='BTCUSDT', interval='1h', days=30):

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    all_data = []
    current_time = start_time
    
    print(f"Fetching {symbol} {interval} bars from {start_time} to {end_time}...")
    
    while current_time < end_time:
        # Binance API expects milliseconds
        start_ms = int(current_time.timestamp() * 1000)
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'limit': 1000  # Max 1000 per request
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break
            
            klines = response.json()
            if not klines:
                break
            
            all_data.extend(klines)
            
            # Move to next batch (last timestamp + 1 interval)
            last_time_ms = klines[-1][0]
            current_time = datetime.fromtimestamp(last_time_ms / 1000, tz=timezone.utc)
            current_time += timedelta(hours=1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Clean up types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    print(f"Fetched {len(df)} bars. Date range: {df.index[0]} to {df.index[-1]}")
    
    return df[['open', 'high', 'low', 'close', 'volume']]



# 2. Volatility and Entropy functions
def rolling_entropy(x, window=60, bins=20):
    #Compute rolling Shannon entropy of residuals.
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p)) if len(p) > 0 else 0  #Shannon entropy Formula 
    
    return x.rolling(window).apply(ent, raw=True)



# 3. GBM Simulator
def simulate_cyber_gbm(S0, mu, sigma_series, H, M, params, bar_sigma2, 
                       n_steps, dt=1, eps=1e-6, nu=4, info_filter=None, 
                       redundancy=None):
    #Inputs:
    #S0: initial price
    #mu: drift (mean log return)
    #sigma_series: FIGARCH-estimated volatility series
    #H: entropy series
    #M: rolling momentum (abs returns)
    #params: dict with alpha, delta, gamma, kappa, eta
    #bar_sigma2: average historical variance
    #n_steps: number of simulation steps ahead
    #dt: time step (1 hour = 1)
    #nu: degrees of freedom for Student-t
    #info_filter: binary filter for high-information periods
    #redundancy: multiplier for market microstructure
    
    # Returns:
    #S: price path (length n_steps + 1)
    #V: volatility path

    S = np.zeros(n_steps + 1)
    V = np.zeros(n_steps + 1)
    S[0] = S0
    
    if info_filter is None:
        info_filter = pd.Series(np.ones(len(H)))
    if redundancy is None:
        redundancy = pd.Series(np.ones(len(H)))
    
    sigma2 = sigma_series.iloc[-1] ** 2
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    
    for t in range(1, n_steps + 1):
        # Use most recent values (index -1 refers to last known bar)
        idx = -1
        H_val = min(H.iloc[idx] / H_max, 1.0) if len(H) > 0 else 0
        M_val = min(M.iloc[idx] / M_max, 1.0) if len(M) > 0 else 0
        
        # Crisis detection
        crisis = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params.get('delta', 0.3) if crisis else 0.0
        
        # Update sigma2 with volatility clustering
        alpha = params.get('alpha', 0.5)
        gamma = params.get('gamma', 0.2)
        
        sigma2 = (
            sigma_series.iloc[idx]**2 * (1 + alpha * H_val + delta_t * M_val)
            + gamma * (bar_sigma2 - sigma2)
        )
        
        # Apply microstructure adjustments
        if idx < len(redundancy):
            sigma2 *= max(1e-12, redundancy.iloc[idx])
        if idx < len(info_filter):
            sigma2 *= 1 + 0.5 * info_filter.iloc[idx]
        
        sigma2 = max(eps, min(sigma2, 0.5))
        V[t] = sigma2
        
        # GBM step with Student-t shocks (fat tails)
        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
    
    return S, V



# Monte Carlo ensemble of GBM paths.
def simulate_mc(S0, mu, sigma_series, H, M, bar_sigma2, nu, 
                n_sims=10_000, n_steps=1, info_filter=None, redundancy=None):
    
    # Calibrated parameters for ~95% coverage target
    # Reduced alpha (0.5 → 0.40) and delta (0.3 → 0.15) to narrow ranges
    base_params = {'alpha': 0.40, 'delta': 0.15, 'gamma': 0.2, 'eta': 1e-3}
    out = np.zeros((n_sims, n_steps + 1))
    
    for i in range(n_sims):
        paths, _ = simulate_cyber_gbm(
            S0, mu, sigma_series, H, M,
            base_params.copy(),
            bar_sigma2, n_steps, dt=1, 
            nu=nu, info_filter=info_filter, redundancy=redundancy
        )
        out[i] = paths
    
    return out



# 4. Backtest function
def backtest_btc_hourly(prices, train_window=168, test_window=None, n_sims=10_000):

    # 30-day backtest on BTCUSDT 1-hour bars.
    # KEY CONSTRAINT: At bar i, only use data up to bar i-1.
    # No lookahead bias.
    
    # Inputs:
    #     prices: Series of closing prices, already sorted
    #     train_window: hours to use for volatility estimation (default 1 week = 168)
    #     test_window: hours to test (default = len(prices) - train_window)
    #     n_sims: Monte Carlo simulations per prediction
    
    # Returns:
    #     DataFrame with predictions and actuals
    
    log_ret = np.log(prices / prices.shift(1)).dropna()
    
    if test_window is None:
        test_window = len(prices) - train_window - 1
    
    results = []
    
    print(f"\nBacktesting {test_window} hourly predictions...")
    print(f"Training window: {train_window} hours, Simulations per bar: {n_sims}")
    
    # Loop: for each test bar i, predict bar i+1 using only data up to bar i
    for i in tqdm(range(train_window, train_window + test_window)):
        # DATA UP TO BAR i (no peeking at bar i+1)
        train_ret = log_ret.iloc[i - train_window:i]  # Up to bar i-1
        train_prices = prices.iloc[i - train_window:i + 1]  # Prices for training
        
        if len(train_ret) < 50:
            continue  # Skip if not enough data
        
        #ESTIMATE VOLATILITY WITH FIGARCH
        try:
            am = arch_model(train_ret * 100, vol='FIGARCH', 
                           p=1, o=0, q=1, dist='studentst')
            res = am.fit(disp='off', rescale=False)
            sigma_fig = res.conditional_volatility / 100
            resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
            nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
        except:
            # Fallback to simple volatility
            sigma_fig = pd.Series([train_ret.std()] * len(train_ret))
            resid = train_ret / train_ret.std()
            nu = 4
        
        #COMPUTE ENTROPY & MOMENTUM
        H_series = rolling_entropy(resid).iloc[:-1] if len(resid) > 60 else pd.Series(np.ones(len(resid)))
        M_series = train_ret.abs().rolling(60).mean() if len(train_ret) > 60 else pd.Series(np.ones(len(train_ret)))
        
        bar_sigma2 = (sigma_fig**2).mean()
        info_filter = (H_series > H_series.mean()).astype(float) if len(H_series) > 0 else None
        
        # Redundancy (price momentum)
        try:
            redundancy = 1 + 0.1 * np.log1p(train_prices.rolling(5).var() / train_prices.rolling(20).var())
            redundancy = redundancy.iloc[:-1]
        except:
            redundancy = pd.Series(np.ones(len(train_ret)))
        
        #PREDICT NEXT HOUR (bar i+1)
        S0 = prices.iloc[i]  # Current price (bar i close)
        mu = train_ret.mean()
        
        # Ensure series align for simulation
        sigma_fig_trim = sigma_fig.iloc[:-1] if len(sigma_fig) > len(H_series) else sigma_fig
        
        paths = simulate_mc(
            S0, mu, sigma_fig_trim, H_series, M_series, bar_sigma2, nu,
            n_sims=n_sims, n_steps=1,
            info_filter=info_filter, redundancy=redundancy
        )
        
        # Extract 95% confidence interval from 10k simulations
        S_t1 = paths[:, 1]
        low_95 = np.percentile(S_t1, 2.5)
        high_95 = np.percentile(S_t1, 97.5)
        
        #CHECK AGAINST ACTUAL (bar i+1)
        actual = prices.iloc[i + 1]
        width_95 = high_95 - low_95
        coverage = 1 if (low_95 <= actual <= high_95) else 0
        
        # Winkler score
        alpha = 0.05
        if actual < low_95:
            winkler = width_95 + (2 / alpha) * (low_95 - actual)
        elif actual > high_95:
            winkler = width_95 + (2 / alpha) * (actual - high_95)
        else:
            winkler = width_95
        
        results.append({
            'timestamp': prices.index[i + 1],
            'bar_index': i + 1,
            'predicted_low': float(low_95),
            'predicted_high': float(high_95),
            'width': float(width_95),
            'actual_price': float(actual),
            'coverage': int(coverage),
            'winkler': float(winkler),
            'price_at_prediction_time': float(S0)
        })
    
    df_results = pd.DataFrame(results)
    return df_results



# 5. Evaluation metrics
def evaluate_predictions(df_results):
    # Compute coverage, avg width, mean Winkler
    coverage = df_results['coverage'].mean()
    avg_width = df_results['width'].mean()
    mean_winkler = df_results['winkler'].mean()
    
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'mean_winkler': mean_winkler,
        'n_predictions': len(df_results)
    }



# 6. Main execution
if __name__ == '__main__':

    print("\nBTC HOURLY GBM BACKTEST")
    
    # Fetch data
    df = get_binance_klines(symbol='BTCUSDT', interval='1h', days=30)
    prices = df['close']
    
    
    # Run backtest
    backtest_df = backtest_btc_hourly(
        prices, 
        train_window=168,  # 1 week
        n_sims=10_000
    )
    
    # Evaluate
    metrics = evaluate_predictions(backtest_df)
    
    print("\nBACKTEST RESULTS")
    print(f"Total Predictions: {metrics['n_predictions']}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    print(f"Average Width: ${metrics['avg_width']:.2f}")
    print(f"Mean Winkler Score: {metrics['mean_winkler']:.4f}")
    
    # Save results
    output_file = 'backtest_results.jsonl'
    print(f"\nSaving predictions to {output_file}...")
    with open(output_file, 'w') as f:
        for _, row in backtest_df.iterrows():
            row_dict = row.to_dict()
            # Convert Timestamp to ISO string
            if 'timestamp' in row_dict:
                row_dict['timestamp'] = row_dict['timestamp'].isoformat()
            f.write(json.dumps(row_dict) + '\n')
    print(f"Saved {len(backtest_df)} predictions")
    
    #plot
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Price with predicted ranges
        ax = axes[0]
        ax.plot(backtest_df['timestamp'], backtest_df['actual_price'], 
                label='Actual', color='black', linewidth=2)
        ax.fill_between(backtest_df['timestamp'], 
                        backtest_df['predicted_low'], 
                        backtest_df['predicted_high'],
                        alpha=0.3, color='blue', label='95% Predicted Range')
        ax.set_ylabel('BTC Price (USD)')
        ax.set_title('BTCUSDT Hourly Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Winkler score over time
        ax = axes[1]
        ax.plot(backtest_df['timestamp'], backtest_df['winkler'], 
                label='Winkler Score', color='red', alpha=0.7)
        ax.axhline(metrics['mean_winkler'], color='green', linestyle='--',
                   label=f'Mean: {metrics["mean_winkler"]:.2f}')
        ax.set_ylabel('Winkler Score (lower is better)')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_visualization.png', dpi=150)
        print("\nSaved visualization to backtest_visualization.png")
        plt.show()
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
