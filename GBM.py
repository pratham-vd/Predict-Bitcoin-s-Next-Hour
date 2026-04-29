import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import requests
from datetime import datetime, timedelta
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import json



# Fetch BTCUSDT 1H data from Binance
def get_binance_klines(symbol='BTCUSDT', interval='1h', days=30):

    # No API key needed
    end_time = datetime.utcnow()
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
            current_time = datetime.fromtimestamp(last_time_ms / 1000)
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
