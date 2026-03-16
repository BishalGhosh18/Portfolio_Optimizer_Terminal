# NSE/BSE Portfolio Optimizer

A Groww-style portfolio analytics dashboard for Indian equities — built with Python and Streamlit.

Live price feed · Risk analysis · Portfolio optimisation · Price prediction

---

## What This Tool Does

Pick any combination of stocks listed on the **NSE or BSE**, analyse their individual and combined risk, optimise portfolio weights across 5 strategies, and forecast future prices using multiple ML/statistical models — all inside a clean, modern UI that auto-refreshes live data every 5 seconds.

---

## Features

| | Feature | Details |
|---|---|---|
| 📊 | **Live Market Feed** | Real-time prices, 52-week high/low, volume, market cap — auto-refreshes every 5 seconds |
| 📉 | **Interactive Charts** | Candlestick, Volume, RSI, MACD, Bollinger Bands, normalised performance comparison |
| ⚠️ | **Risk Analytics** | VaR (Historical, Parametric), CVaR/ES, Sharpe, Sortino, Calmar, Beta, Max Drawdown, Rolling Volatility |
| ⚙️ | **Portfolio Optimisation** | 5 strategies: Max Sharpe, Min Volatility, Risk Parity, Max Diversification, Equal Weight |
| 🎯 | **Risk Scorecard** | Composite risk score (0–100) per stock with radar chart |
| 🔮 | **Price Prediction** | 5 models: ARIMA, Linear Regression, Random Forest, Monte Carlo GBM, EMA Trend |
| 🏢 | **130+ Stocks** | Full NSE & BSE universe across all major sectors |
| 🎛️ | **Preset Baskets** | One-click presets: Tata Group, Top IT, Top Banking, Top Auto, Pharma, Energy |

---

## UI

Clean, modern design inspired by Groww:
- White card layout with soft shadows
- Green `#00D09C` for gains · Red `#FF5370` for losses
- Purple gradient `#5367FF → #8B5CF6` for accents and buttons
- Dark navy `#1B2236` sidebar
- Live blinking dot indicator
- Scrollable ticker strip at the top

---

## Project Structure

```
portfolio-optimizer/
├── app.py            — Streamlit UI (Groww theme, 6 tabs, auto-refresh)
├── data_fetcher.py   — yfinance data layer, 130+ stock universe, live quotes
├── risk_engine.py    — Risk metrics: VaR, CVaR, Sharpe, Beta, drawdown, scorecard
├── optimizer.py      — Markowitz optimisation, 5 strategies
├── predictor.py      — Price prediction: ARIMA, LinReg, RF, Monte Carlo, EMA
├── requirements.txt  — Python dependencies
└── run.sh            — One-command launcher
```

### Module Breakdown

**`data_fetcher.py`**
- `get_stock_universe(exchange)` — returns full `{company_name: ticker}` map for NSE or BSE
- `fetch_price_data(tickers, period)` — downloads adjusted close prices via yfinance
- `fetch_ohlcv(ticker, period)` — OHLCV bars for candlestick charts
- `fetch_live_quote(ticker)` — current price, day change %, volume, 52w high/low, market cap using `fast_info`
- `fetch_all_live_quotes(names, exchange)` — batch live quotes returned as `{company_name: quote_dict}`
- `fetch_benchmark(period)` — Nifty 50 and Sensex returns for beta calculation

**`risk_engine.py`**
- `annualised_return / annualised_volatility` — core return/vol statistics
- `sharpe_ratio / sortino_ratio / calmar_ratio` — risk-adjusted return metrics
- `historical_var / parametric_var / cvar` — Value-at-Risk at 95% and 99%
- `max_drawdown / drawdown_series` — peak-to-trough drawdown
- `beta(returns, benchmark_returns)` — market sensitivity vs Nifty 50 / Sensex
- `correlation_matrix / rolling_volatility` — portfolio-level diagnostics
- `risk_scorecard(returns, benchmark_returns)` — composite 0–100 risk score per stock

**`optimizer.py`**
- `max_sharpe_weights` — SLSQP optimisation maximising Sharpe ratio
- `min_volatility_weights` — minimum portfolio variance
- `risk_parity_weights` — equal risk contribution from each asset
- `max_diversification_weights` — maximises diversification ratio
- `equal_weight` — uniform 1/N baseline
- `all_strategies_summary` — runs all 5 strategies and returns a comparison table

**`predictor.py`**
- `arima_forecast` — ARIMA(p,d,q) with auto-order selection; confidence intervals
- `linear_regression_forecast` — Ridge regression with lag features, MA, RSI, MACD; walk-forward validation
- `random_forest_forecast` — Random Forest ensemble; walk-forward validation
- `monte_carlo_forecast` — Geometric Brownian Motion (1000 paths), median + confidence band
- `ema_forecast` — EMA trend extrapolation with residual-based confidence band
- `compute_technical_indicators` — MA(20/50/200), EMA(12/26), MACD, RSI(14), Bollinger Bands, ATR

---

## Installation

**Requirements:** Python 3.10 or 3.11 · Internet connection (data via Yahoo Finance)

> Python 3.12 may have CVXPY/NumPy conflicts — Python 3.11 is recommended.

### Option A — One-command launcher

```bash
git clone https://github.com/BishalGhosh18/Portfolio_Optimizer_Terminal
cd Portfolio_Optimizer_Terminal
bash run.sh
```

The script auto-detects conda or venv, installs all dependencies, and opens the app at [http://localhost:8501](http://localhost:8501).

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

---

## How to Use

### 1 — Select stocks
- Choose **NSE** or **BSE** in the sidebar
- Pick a **preset basket** (Tata Group, Top IT, Top Banking, Top Auto, Pharma, Energy) or search any company
- Changing the selection automatically re-fetches data

### 2 — Set parameters
- **Lookback period** — 6 months to 5 years
- **Optimization strategy** — choose one of the 5 strategies
- **Risk-free rate** — defaults to 6.5% (Indian 10-yr G-Sec)

### 3 — Run analysis
Click **▶ Run Analysis** to load historical price data and compute all metrics.

### 4 — Navigate tabs

| Tab | Contents |
|---|---|
| 📊 **Live Feed** | Live stock cards with price, change %, 52w high/low, volume, market cap · Market breadth (gainers/losers) · Benchmark chart |
| 📉 **Charts** | Candlestick with MA overlays + Bollinger Bands · Volume · RSI · MACD · Normalised performance · Correlation matrix |
| ⚠️ **Risk** | Per-stock risk table (Return, Volatility, Sharpe, Max Drawdown, VaR, CVaR) · Rolling volatility · Drawdown chart |
| ⚙️ **Optimizer** | Portfolio weights pie & bar chart · Key portfolio metrics · Strategy comparison table |
| 🎯 **Scorecard** | Risk score cards per stock · Radar chart · Full metrics table |
| 🔮 **Predict** | Forecast chart with confidence bands · Scenario table (1W / 1M / horizon) · Model accuracy metrics |

### 5 — Run a prediction
Select a **stock**, **model**, and **forecast horizon** in the sidebar, then click **🔮 Run Prediction**.

---

## Supported Sectors

| Sector | Example Stocks |
|---|---|
| Tata Group | TCS, Tata Motors, Tata Steel, Tata Power, Titan, Trent, Voltas |
| Banking & Finance | HDFC Bank, ICICI Bank, SBI, Kotak, Axis, Bajaj Finance |
| IT & Technology | Infosys, Wipro, HCL Tech, Tech Mahindra |
| Pharma | Sun Pharma, Dr. Reddy's, Cipla, Divi's, Lupin |
| Auto | Maruti, M&M, Hero MotoCorp, Bajaj Auto, Eicher |
| Energy | Reliance, ONGC, Adani Green, NTPC, Power Grid |
| FMCG | HUL, ITC, Nestle, Britannia, Dabur |
| Metals & Mining | JSW Steel, Hindalco, Vedanta, SAIL |
| Infrastructure | L&T, Adani Ports, DLF |
| Telecom | Bharti Airtel |

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
matplotlib>=3.8.0
```

---

## Technical Notes

- **Data source:** Yahoo Finance via `yfinance 1.2+`. NSE/BSE prices are delayed ~15 minutes.
- **Live quotes:** Uses `fast_info.last_price` and `fast_info.previous_close` directly — no history call, no cache.
- **Auto-refresh:** Page reruns every 5 seconds (configurable) using `streamlit-autorefresh`. Live quotes are re-fetched on every rerun; historical prices only reload on **Run Analysis**.
- **Selection change detection:** Changing the stock selection in the sidebar automatically invalidates the price cache and triggers a fresh fetch.
- **NumPy version:** CVXPY requires `numpy < 2.0`. The `requirements.txt` pins this.
- **Plotly compatibility:** Uses `add_shape` + `add_annotation` instead of `add_vline` to avoid a bug in Plotly 6.x with timezone-aware timestamps.
- **Walk-forward forecasting:** Linear Regression and Random Forest models use walk-forward validation to avoid look-ahead bias.
- **ARIMA confidence intervals:** Handles both DataFrame and ndarray return types from `statsmodels` `conf_int()` across versions.

---

## License

MIT — free to use, modify, and distribute.
