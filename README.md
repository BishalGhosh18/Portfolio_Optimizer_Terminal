# NSE/BSE Portfolio Optimizer Terminal

A Bloomberg-style terminal UI for Indian equity portfolio analysis and prediction built with Python and Streamlit.

---

## What This Tool Does

This tool lets you pick any combination of stocks listed on the **NSE or BSE**, analyse their individual and combined risk, optimise portfolio weights across 5 strategies, and forecast future prices using multiple ML/statistical models — all inside a dark terminal-themed interface that auto-refreshes live data every 5 seconds.

---

## Features

| Feature | Details |
|---|---|
| **Live Data Feed** | Real-time prices, 52-week high/low, volume, market cap — refreshes every 5 seconds |
| **130+ Stocks** | Full NSE & BSE universe: Tata, Reliance, Adani, HDFC, Infosys, and 100+ more across all sectors |
| **Preset Baskets** | One-click presets: Tata Group, Top IT, Top Banking, Top Auto, Pharma, Energy |
| **Interactive Charts** | Candlestick, Volume, RSI, MACD, Bollinger Bands |
| **Risk Analytics** | VaR (Historical, Parametric), CVaR/ES, Sharpe, Sortino, Calmar, Beta, Max Drawdown |
| **Portfolio Optimisation** | 5 strategies: Max Sharpe, Min Volatility, Risk Parity, Max Diversification, Equal Weight |
| **Risk Scorecard** | Composite risk score (0–100) with radar chart for each stock |
| **Price Prediction** | 5 models: ARIMA, Linear Regression, Random Forest, Monte Carlo GBM, EMA Trend |

---

## Screenshots

> Terminal dark theme — green on black, JetBrains Mono font, live ticker strip at the top

---

## Project Structure

```
portfolio-optimizer/
├── app.py            # Streamlit UI — terminal theme, all 6 tabs, auto-refresh
├── data_fetcher.py   # yfinance data layer — stock universe, live quotes, OHLCV
├── risk_engine.py    # Risk metrics — VaR, CVaR, Sharpe, Beta, drawdown, scorecard
├── optimizer.py      # Markowitz optimisation — 5 strategies + efficient frontier
├── predictor.py      # Prediction engine — ARIMA, LinReg, RF, Monte Carlo, EMA
├── requirements.txt  # Python dependencies
└── run.sh            # One-command launcher (conda/venv auto-detect)
```

### Module Breakdown

**`data_fetcher.py`**
- Maintains a dictionary of 130+ NSE/BSE stocks with their Yahoo Finance tickers
- `get_stock_universe(exchange)` — returns the full name→ticker map
- `fetch_price_data(tickers, period)` — downloads adjusted close prices via yfinance
- `fetch_ohlcv(ticker, period)` — OHLCV data for candlestick charts
- `fetch_live_quote(ticker)` — current price, day change, volume, 52w high/low, market cap
- `fetch_all_live_quotes(names, exchange)` — batch live quotes for the ticker strip
- `fetch_benchmark(period)` — Nifty 50 and Sensex returns for beta calculation

**`risk_engine.py`**
- `annualised_return / annualised_volatility` — core return/vol statistics
- `sharpe_ratio / sortino_ratio / calmar_ratio` — risk-adjusted return metrics
- `historical_var / parametric_var / cvar` — Value-at-Risk at configurable confidence levels
- `max_drawdown / drawdown_series` — peak-to-trough drawdown analysis
- `beta(returns, benchmark_returns)` — market sensitivity vs Nifty 50 / Sensex
- `correlation_matrix / rolling_volatility` — portfolio-level diagnostics
- `risk_scorecard(returns, benchmark_returns)` — composite 0–100 risk score per stock

**`optimizer.py`**
- `max_sharpe_weights(returns)` — SLSQP optimisation maximising Sharpe ratio
- `min_volatility_weights(returns)` — minimum portfolio variance
- `risk_parity_weights(returns)` — equal risk contribution from each asset
- `max_diversification_weights(returns)` — maximises diversification ratio
- `equal_weight(returns)` — uniform 1/N baseline
- `all_strategies_summary(returns)` — runs all 5 strategies and returns a comparison table
- `efficient_frontier_monte_carlo(returns, n)` — random portfolio simulation for frontier plot

**`predictor.py`**
- `arima_forecast(prices, horizon)` — ARIMA(p,d,q) fitted via auto-order selection
- `linear_regression_forecast(prices, horizon)` — Ridge regression with lag features, MA, RSI, MACD
- `random_forest_forecast(prices, horizon)` — Random Forest with walk-forward validation
- `monte_carlo_forecast(prices, horizon, simulations)` — Geometric Brownian Motion, 1000 paths
- `ema_forecast(prices, horizon)` — Exponential Moving Average trend extrapolation
- `compute_technical_indicators(ohlcv)` — adds MA(20/50/200), MACD, RSI, Bollinger Bands, ATR

---

## Installation

**Requirements:** Python 3.10 or 3.11, internet connection (live data via Yahoo Finance)

> Python 3.12 may have CVXPY/NumPy conflicts — 3.11 is recommended.

### Option A — One-command launcher

```bash
git clone https://github.com/your-username/portfolio-optimizer
cd portfolio-optimizer
bash run.sh
```

The script auto-detects conda or venv, installs all dependencies, and opens the app.

### Option B — Conda (recommended)

```bash
conda create -n portfolio_optimizer python=3.11 -y
conda activate portfolio_optimizer
pip install -r requirements.txt
streamlit run app.py
```

### Option C — Plain pip / venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

### Step 1 — Select stocks
- Choose **NSE** or **BSE** in the sidebar
- Use a **preset basket** (Tata Group, Top IT, Top Banking, Top Auto, Pharma, Energy) or
- Search and select any companies from the full list

### Step 2 — Set parameters
- **Lookback period** — 6 months to 5 years of historical data
- **Risk-free rate** — defaults to 6.5% (Indian 10-yr G-Sec)

### Step 3 — Run analysis
Click **► RUN ANALYSIS** to load data and compute all metrics.

### Step 4 — Navigate tabs

| Tab | What you see |
|---|---|
| **LIVE FEED** | Real-time prices for all selected stocks + scrolling ticker strip |
| **CHARTS** | Candlestick, volume, RSI, MACD, Bollinger Bands for any selected stock |
| **RISK** | VaR, CVaR, drawdown chart, correlation matrix, rolling volatility |
| **OPTIMIZER** | Portfolio weights & performance across all 5 strategies |
| **SCORECARD** | Per-stock composite risk score with radar chart |
| **PREDICT** | Future price forecast using your chosen model and horizon |

### Step 5 — Run a prediction
- In the sidebar, select a **stock**, **model**, and **forecast horizon** (days)
- Click **► RUN PREDICTION**
- Go to the **PREDICT** tab to see the forecast chart with confidence bands

---

## Supported Sectors

| Sector | Example Stocks |
|---|---|
| Tata Group | TCS, Tata Motors, Tata Steel, Tata Power, Titan |
| Banking & Finance | HDFC Bank, ICICI Bank, SBI, Kotak, Bajaj Finance |
| IT & Technology | Infosys, Wipro, HCL Tech, Tech Mahindra |
| Pharma | Sun Pharma, Dr. Reddy's, Cipla, Divi's |
| Auto | Maruti, M&M, Hero MotoCorp, Bajaj Auto, Eicher |
| Energy | Reliance, ONGC, Adani Green, Tata Power, NTPC |
| FMCG | HUL, ITC, Nestle, Britannia, Dabur |
| Infrastructure | L&T, Adani Ports, DLF |
| Metals | JSW Steel, Hindalco, Vedanta, SAIL |
| Telecom | Bharti Airtel, Vodafone Idea |

---

## Technical Notes

- **Data source:** Yahoo Finance via `yfinance`. NSE/BSE prices are delayed ~15 minutes.
- **NumPy version:** CVXPY requires `numpy < 2.0`. The `requirements.txt` pins this.
- **Auto-refresh:** Live data refreshes every 5 seconds using `streamlit-autorefresh`.
- **Plotly compatibility:** Uses `add_shape` + `add_annotation` instead of `add_vline` to avoid a bug in Plotly 6.x with timezone-aware timestamps.
- **Walk-forward forecasting:** Linear Regression and Random Forest models use walk-forward validation to avoid look-ahead bias.

---

## Dependencies

```
streamlit>=1.32.0
yfinance>=0.2.50
pandas>=2.0.0
numpy>=1.26.0,<2.0.0
scipy>=1.12.0
plotly>=5.20.0
cvxpy>=1.4.0
statsmodels>=0.14.0
scikit-learn>=1.4.0
streamlit-autorefresh>=1.0.1
```

---

## License

MIT — free to use, modify, and distribute.
