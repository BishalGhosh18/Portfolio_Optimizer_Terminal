# NSE/BSE Portfolio Optimizer

A Groww-style portfolio analytics dashboard for Indian equities — built with Python and Streamlit.

Live price feed · Risk analysis · Portfolio optimisation · Price prediction · Market terminal

---

## What This Tool Does

Pick any combination of stocks listed on the **NSE or BSE**, analyse their individual and combined risk, optimise portfolio weights across 5 strategies, forecast future prices using an ensemble of time-series and ML models, and monitor live market intelligence — all inside a clean, modern UI that auto-refreshes every 5 seconds.

---

## Features

| | Feature | Details |
|---|---|---|
| 📊 | **Live Market Feed** | Real-time prices, 52-week high/low, volume, market cap — auto-refreshes every 5 seconds |
| 🖥️ | **Terminal** | 7-panel market intelligence dashboard — indices, TA chart, order book, news, geo map, strategy signals, sector heatmap |
| 📉 | **Interactive Charts** | Candlestick, Volume, RSI, MACD, Bollinger Bands, normalised performance comparison, correlation matrix |
| ⚠️ | **Risk Analytics** | VaR (Historical, Parametric), CVaR/ES, Sharpe, Sortino, Calmar, Beta, Max Drawdown, Rolling Volatility |
| ⚙️ | **Portfolio Optimisation** | 5 strategies: Max Sharpe, Min Volatility, Risk Parity, Max Diversification, Equal Weight |
| 🔮 | **Price Prediction** | 8 models: TS Ensemble, Prophet, Holt-Winters, SARIMA, Theta, XGBoost, LightGBM, Monte Carlo — with confidence bands |
| 🏢 | **130+ Stocks** | Full NSE & BSE universe across all major sectors |
| 🎛️ | **Preset Baskets** | One-click presets: Tata Group, Top IT, Top Banking, Top Auto, Pharma, Energy |

---

## Tabs

| Tab | Contents |
|---|---|
| 📊 **Live Feed** | Live stock cards with price, change %, 52w high/low, volume, market cap · Market breadth (gainers vs losers) · Benchmark chart |
| 🖥️ **Terminal** | Full-screen market intelligence — sidebar hides automatically for maximum space |
| 📉 **Charts** | Candlestick with EMA/SMA/Bollinger overlays · Volume · RSI · MACD · Normalised performance · Correlation matrix |
| ⚠️ **Risk** | Per-stock risk table (Return, Volatility, Sharpe, Max Drawdown, VaR 95/99%, CVaR) · Rolling volatility · Drawdown chart |
| ⚙️ **Optimizer** | Portfolio weights pie & bar chart · Key portfolio metrics · 5-strategy comparison table |
| 🔮 **Predict** | Forecast chart with 90% confidence bands · Accuracy leaderboard · Model comparison · Training data filter |

---

## Terminal Tab — 7 Panels

The Terminal tab is a full-screen market intelligence dashboard. The sidebar hides automatically when the tab is active.

| Panel | Description |
|---|---|
| **Market Overview** | Live NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT — price, change %, 52w range bar, FII/DII flow indicator |
| **TA Chart** | Candlestick with EMA 20, EMA 50, SMA 200, Bollinger Bands · Volume subplot · Timeframe: 5D / 1M / 3M / 6M / 1Y / 2Y |
| **Order Book** | Simulated 5-level bid/ask depth around live LTP · Depth bar visualisation · Spread % · Total volume |
| **Live News Feed** | Financial news with TextBlob sentiment tags (BULL / BEAR / NEUTRAL) · NewsAPI primary · ET Markets & Moneycontrol RSS fallback |
| **Geopolitical Map** | Interactive world map (Plotly Scattergeo) · 8 hotspots: conflicts, tensions, economic hubs, home market India |
| **Strategy Signals** | 3 algo signals: EMA Crossover, BB Squeeze, RSI Divergence · Entry / Target / Stop Loss / R:R / Win probability |
| **Sector Heatmap** | Live treemap of top 20 NSE stocks · Dark green → strong gain · Dark red → strong loss |

---

## Prediction Models

| Model | Approach |
|---|---|
| **TS Ensemble** | Inverse-MAPE weighted blend of all 6 time-series models — best overall accuracy |
| **Prophet** | Facebook Prophet: trend + weekly/yearly seasonality decomposition |
| **Holt-Winters (ETS)** | Exponential smoothing with damped trend — robust for medium horizons |
| **SARIMA** | Seasonal ARIMA with auto-order selection; weekly seasonality (s=5) |
| **Theta** | Theta decomposition — often outperforms ARIMA on financial data |
| **XGBoost** | Gradient boosting on 80+ TS features with TimeSeriesSplit cross-validation |
| **LightGBM** | Fast gradient boosting with TimeSeriesSplit CV |
| **Monte Carlo (GBM)** | 5,000 simulated GBM paths — range estimation and stress testing |

All ML models predict **log returns** (stationary) then reconstruct the price path via `last_price × exp(cumsum(log_returns))`. Confidence bands are built from bootstrap residual resampling (500 iterations).

---

## APIs & Data Sources

| Source | Used For | Key Required |
|---|---|---|
| **Yahoo Finance** (`yfinance`) | All stock prices, OHLCV, live quotes, index data | No |
| **NewsAPI.org** | Live financial news headlines | Optional — free tier |
| **ET Markets RSS** | News fallback if NewsAPI key missing | No |
| **Moneycontrol RSS** | News fallback if NewsAPI key missing | No |
| **Plotly built-in geo** | Geopolitical world map | No |

To enable live news from NewsAPI, add to `.env`:
```
NEWS_API_KEY=your_newsapi_org_key
```
Without it, the RSS fallback activates automatically.

---

## Project Structure

```
portfolio-optimizer/
├── app.py              — Streamlit UI: 6 tabs, Groww theme, auto-refresh, sidebar hide
├── data_fetcher.py     — yfinance data layer, 130+ stock universe, live quotes
├── risk_engine.py      — Risk metrics: VaR, CVaR, Sharpe, Beta, drawdown
├── optimizer.py        — Markowitz optimisation, 5 strategies
├── predictor.py        — Price prediction: TS Ensemble, Prophet, HW, SARIMA, Theta, XGB, LGB, Monte Carlo
├── terminal_tab.py     — Terminal tab UI: 7 panels, full-screen layout
├── terminal_utils.py   — Terminal back-end: index quotes, order book sim, news, signals, heatmap
├── requirements.txt    — Python dependencies
└── run.sh              — One-command launcher
```

### Module Breakdown

**`data_fetcher.py`**
- `get_stock_universe(exchange)` — returns full `{company_name: ticker}` map for NSE or BSE
- `fetch_price_data(tickers, period)` — downloads adjusted close prices via yfinance
- `fetch_ohlcv(ticker, period)` — OHLCV bars for charting and prediction
- `fetch_live_quote(ticker)` — current price, day change %, 52w high/low, market cap
- `fetch_all_live_quotes(names, exchange)` — batch live quotes as `{company_name: quote_dict}`
- `fetch_benchmark(period)` — Nifty 50 and Sensex returns for beta calculation

**`risk_engine.py`**
- `annualised_return / annualised_volatility` — core return/vol statistics
- `sharpe_ratio / sortino_ratio / calmar_ratio` — risk-adjusted return metrics
- `var_summary` — VaR at 95% and 99% (historical and parametric)
- `max_drawdown / drawdown_series` — peak-to-trough drawdown
- `beta(returns, benchmark_returns)` — market sensitivity vs Nifty 50
- `correlation_matrix / rolling_volatility` — portfolio-level diagnostics

**`optimizer.py`**
- `max_sharpe_weights` — SLSQP optimisation maximising Sharpe ratio
- `min_volatility_weights` — minimum portfolio variance
- `risk_parity_weights` — equal risk contribution from each asset
- `max_diversification_weights` — maximises diversification ratio
- `equal_weight` — uniform 1/N baseline
- `all_strategies_summary` — runs all 5 strategies and returns a comparison table

**`predictor.py`**
- `ts_ensemble_forecast` — inverse-MAPE weighted blend of all 6 TS models
- `prophet_forecast` — Facebook Prophet with changepoint tuning
- `holtwinters_forecast` — ETS with damped trend, best-of-4 variant selection
- `sarima_forecast` — SARIMA with auto-order search, seasonal period s=5
- `theta_forecast` — Theta model via statsmodels
- `xgboost_ts_forecast` — XGBoost with 80+ features, TimeSeriesSplit CV
- `lightgbm_ts_forecast` — LightGBM with TimeSeriesSplit CV
- `monte_carlo_forecast` — 5,000 GBM paths with GARCH-like volatility

**`terminal_utils.py`**
- `fetch_index_quotes()` — live quotes for 4 major Indian indices (cached 30s)
- `simulate_order_book(ltp)` — realistic 5-level bid/ask depth simulation
- `fetch_news(api_key)` — NewsAPI → RSS fallback, TextBlob sentiment scoring (cached 60s)
- `compute_strategy_signals(ticker, ohlcv)` — EMA Crossover, BB Squeeze, RSI Divergence using pure pandas/numpy
- `fetch_heatmap_data()` — live % change for top-20 NSE stocks (cached 60s)

**`terminal_tab.py`**
- `render_terminal_tab(universe, exchange)` — renders all 7 panels in a 3+4 column layout

---

## Installation

**Requirements:** Python 3.11 recommended · Internet connection (data via Yahoo Finance)

> CVXPY requires `numpy < 2.0`. The `requirements.txt` pins this. Python 3.14+ is not supported by some dependencies.

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
- Pick a **preset basket** or search any company
- Changing the selection automatically re-fetches data

### 2 — Set parameters
- **Lookback period** — 6 months to 5 years
- **Optimization strategy** — choose one of 5 strategies
- **Risk-free rate** — defaults to 6.5% (Indian 10-yr G-Sec)

### 3 — Run analysis
Click **▶ Run Analysis** to load historical price data and compute all metrics.

### 4 — Run a prediction
- Switch to the **🔮 Predict** tab
- Select a **stock**, **model**, and **forecast horizon** in the sidebar
- Optionally enable **Training Data Filter** to restrict the training date range
- Click **🔮 Run Prediction**

### 5 — Use the Terminal
- Click the **🖥️ Terminal** tab — the sidebar hides for maximum screen space
- Select stocks independently for the TA chart, order book, and strategy signals
- News refreshes every 60 s · Index quotes refresh every 30 s · Order book updates every 5 s

---

## Supported Sectors

| Sector | Example Stocks |
|---|---|
| Tata Group | TCS, Tata Motors CV, Tata Steel, Tata Power, Titan, Trent, Voltas |
| Banking & Finance | HDFC Bank, ICICI Bank, SBI, Kotak, Axis, Bajaj Finance |
| IT & Technology | Infosys, Wipro, HCL Tech, Tech Mahindra, LTIMindtree |
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
streamlit>=1.32.0          # UI framework
yfinance>=0.2.50           # Market data
pandas>=2.0.0
numpy>=1.26.0,<2.0.0       # Pinned — CVXPY requires <2.0
scipy>=1.12.0
plotly>=5.20.0             # Charts and geo map
cvxpy>=1.4.0               # Portfolio optimisation
statsmodels>=0.14.0        # SARIMA, Holt-Winters, Theta
scikit-learn>=1.4.0
streamlit-autorefresh>=1.0.1
matplotlib>=3.8.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
prophet>=1.1.5
textblob>=0.18.0           # News sentiment
feedparser>=6.0.11         # RSS news fallback
pydeck>=0.8.0              # (installed, map replaced with Plotly Scattergeo)
requests>=2.31.0           # NewsAPI HTTP calls
python-dotenv>=1.0.0       # .env for NEWS_API_KEY
```

---

## Technical Notes

- **Data source:** Yahoo Finance via `yfinance`. NSE/BSE prices are delayed ~15 minutes.
- **Live quotes:** Uses `fast_info.last_price` and `fast_info.previous_close` — no history call on live feed.
- **Auto-refresh:** Page reruns every 5 seconds using `streamlit-autorefresh`.
- **Terminal sidebar:** Injected CSS hides `[data-testid="stSidebar"]` when Terminal tab is active; other tabs restore it normally.
- **Order book:** Fully simulated — yfinance does not provide real Level 2 data. Clearly labelled in the UI.
- **Strategy signals:** Computed on tab activation only (cached in `st.session_state`) — not on every rerun.
- **Indicators:** EMA, RSI, ATR, ADX, Bollinger Bands implemented in pure pandas/numpy — no `pandas-ta` dependency (incompatible with Python 3.14).
- **Tata Motors tickers:** Yahoo Finance renamed `TATAMOTORS.NS` on 2026-03-17. Use `TMCV.NS` (commercial vehicles) and `TMPV.NS` (passenger vehicles, demerged entity).
- **Log-return prediction:** ML models predict log returns (stationary) then reconstruct the price path via `last_price × exp(cumsum(log_returns))`.
- **Bootstrap CI:** Confidence intervals built by resampling model residuals 500 times.
- **Plotly vline bug:** Uses `add_shape` + `add_annotation` instead of `add_vline` to avoid a bug in Plotly 6.x with string-date x-values and `annotation_position`.
- **NumPy version:** CVXPY requires `numpy < 2.0`. The `requirements.txt` pins this.

---

## License

MIT — free to use, modify, and distribute.
