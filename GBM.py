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
    
    base_params = {'alpha': 0.5, 'delta': 0.3, 'gamma': 0.2, 'eta': 1e-3}
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

