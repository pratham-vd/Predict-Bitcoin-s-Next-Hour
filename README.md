# BTC/USDT Next-Hour Price Predictor

A quantitative forecasting system that predicts the 95% price range for Bitcoin one hour ahead, using Geometric Brownian Motion with FIGARCH volatility estimation and Monte Carlo simulation.

Live dashboard: https://predict-bitcoins-next-hour-mellqxzr8cl92hnfjr9hgs.streamlit.app

---

## What it does

Every hour, a new BTC/USDT candle closes on Binance. This system:

1. Looks at the past 168 hours of price data
2. Estimates current market volatility using a FIGARCH model
3. Simulates 10,000 possible price paths for the next hour
4. Reports the range where 95% of those paths land

The goal is to be right approximately 95% of the time while keeping the predicted range as narrow as possible.

---

## Project Structure

| File | Purpose |
|---|---|
| `GBM.py` | Core model — data fetching, volatility estimation, simulation, backtest |
| `streamlit_dashboard.py` | Live dashboard — real-time prediction and history tracking |
| `backtest_results.jsonl` | 551 hourly predictions from the 30-day backtest |
| `prediction_history.json` | Live prediction history saved on each dashboard visit |
| `requirements.txt` | Python dependencies |

---

## How the model works

### Data
Hourly BTCUSDT OHLCV bars are fetched from Binance's public API. No API key required.

### Volatility estimation
A FIGARCH(1, d, 1) model is fitted on the most recent 168 hours of log returns. FIGARCH is chosen over standard GARCH because Bitcoin's volatility exhibits long memory — a spike today continues to influence volatility for days, not just hours.

### Entropy and momentum
Shannon entropy of the model's standardized residuals is computed over a rolling 60-hour window. High entropy indicates the market is behaving more unpredictably than the model expects, which increases the simulated volatility. A rolling mean of absolute returns captures momentum — large recent moves signal elevated uncertainty.

### Simulation
10,000 price paths are simulated using the GBM equation:

```
S(t+1) = S(t) * exp((mu - 0.5 * sigma²) + sqrt(sigma²) * Z)
```

Where Z is drawn from a Student-t distribution (degrees of freedom estimated from the data) rather than a normal distribution. This captures Bitcoin's fat tails — extreme hourly moves happen more frequently than a normal distribution would predict.

The (mu - 0.5 * sigma²) term is the Itô correction, required to ensure the expected log return is unbiased under continuous-time finance.

### Prediction interval
The 2.5th and 97.5th percentiles of the 10,000 simulated next-hour prices form the 95% prediction interval.

---

## Backtest methodology

The backtest runs on 30 days of hourly data with a strict no-lookahead constraint. At each bar i, the model uses only data from bars 1 through i to predict bar i+1. The actual price at i+1 is revealed only after the prediction is locked in.

This mirrors what a live trading system would experience — no information from the future leaks into any prediction.

### Evaluation metrics

**Coverage** — what fraction of actual prices fell inside the predicted 95% range. Target is 0.95.

**Average width** — the mean dollar width of the predicted interval. Narrower is better, provided coverage stays near 0.95.

**Winkler score** — a penalty-based scoring rule that rewards narrow intervals and penalizes misses heavily. If the actual price falls inside the range, the score equals the interval width. If it misses, a penalty of (2 / 0.05) times the miss distance is added. Lower is better.

### Results

| Metric | Value |
|---|---|
| Coverage | 96.19% |
| Average Width | $1,293 |
| Winkler Score | 1,697 |
| Total Predictions | 551 |

---

## Live dashboard

The dashboard fetches the latest 500 closed hourly bars from Binance, runs the model, and displays the predicted range for the next hour.

**Part C — Prediction history:** Every visit to the dashboard saves the current prediction. Once that hour closes, the actual BTC price is automatically filled in alongside the prediction, building a growing timeline of predictions and outcomes.

---

## Running locally

```bash
pip install -r requirements.txt

# Run the 30-day backtest
python GBM.py

# Launch the dashboard
streamlit run streamlit_dashboard.py
```

---

## Key design decisions

**Why FIGARCH over GARCH?** Standard GARCH models assume volatility shocks decay exponentially. FIGARCH uses fractional integration to model the slower, power-law decay observed in crypto markets.

**Why Student-t shocks?** Bitcoin returns have heavier tails than a normal distribution. Fitting the degrees of freedom parameter to the model's residuals at each bar gives an empirical estimate of tail thickness.

**Why 168-hour training window?** One week of hourly data captures recent volatility regimes without being so long that it includes stale market conditions. A shorter window increases noise; a longer window reduces responsiveness.

**Why 10,000 simulations?** At this count, the Monte Carlo variance on the 2.5th and 97.5th percentiles is approximately ±0.16%, which translates to ±0.8% variance in coverage across runs. Increasing to 50,000 would reduce this further with minimal speed cost due to the vectorised implementation.
