# Nifty-50 Portfolio Optimizer & Movement Predictor

A Groww-style portfolio analytics dashboard for Indian equities — built with Python and Streamlit.

Live price feed · Risk analysis · Portfolio optimisation · Stock Movement Predictor · Long-term price forecast · Market terminal

---

## What This Tool Does

Work across the **Nifty 50** universe: analyse individual and combined risk, optimise portfolio weights across 5 strategies, predict the **next-day direction** of any stock with a machine-learning ensemble, project its **price 1–4 months out**, and monitor live market intelligence — all inside a clean, modern UI that auto-refreshes every 5 seconds.

The **Stock Movement Predictor** doesn't just look at past prices. It trains on price/technical signals, **volume buy-sell pressure**, **earnings-cycle** features and **market regime** (Nifty + India VIX), scores itself on a **walk-forward backtest** (accuracy, precision, recall, F1 vs a baseline), then overlays a live read of **news, fundamentals (Screener.in) and analyst views** — with every signal honestly labelled as backtested or not.

---

## Features

| | Feature | Details |
|---|---|---|
| 📊 | **Live Market Feed** | Real-time prices, 52-week high/low, volume, market cap — auto-refreshes every 5 seconds |
| 🖥️ | **Terminal** | 7-panel market intelligence dashboard — indices, TA chart, order book, news, geo map, strategy signals, sector heatmap |
| 📉 | **Interactive Charts** | Candlestick, Volume, RSI, MACD, Bollinger Bands, normalised performance comparison, correlation matrix |
| ⚠️ | **Risk Analytics** | VaR (Historical, Parametric), CVaR/ES, Sharpe, Sortino, Calmar, Beta, Max Drawdown, Rolling Volatility |
| ⚙️ | **Portfolio Optimisation** | 5 strategies: Max Sharpe, Min Volatility, Risk Parity, Max Diversification, Equal Weight |
| 🔮 | **Stock Movement Predictor** | Next-day up/down direction — LightGBM + XGBoost + Logistic Regression **soft-voting ensemble** on 27 features (price · volume pressure · earnings · market regime) · walk-forward backtested (accuracy / precision / recall / F1) · live news/fundamentals/analyst overlay |
| 📈 | **Long-Term Price Forecast** | Monte-Carlo price projection for 1 / 2 / 3 / 4 months — drift dampened & anchored to analyst targets, with 50% and 90% confidence bands |
| 🏢 | **Nifty 50** | The entire app operates on the Nifty 50 (NSE) — a curated, high-liquidity universe (easily extendable) |
| 🎛️ | **Preset Baskets** | One-click sector presets: IT, Banking, Auto, Pharma, Energy, FMCG |

---

## Tabs

| Tab | Contents |
|---|---|
| 📊 **Live Feed** | Live stock cards with price, change %, 52w high/low, volume, market cap · Market breadth (gainers vs losers) · Benchmark chart |
| 🖥️ **Terminal** | Full-screen market intelligence — sidebar hides automatically for maximum space |
| 📉 **Charts** | Candlestick with EMA/SMA/Bollinger overlays · Volume · RSI · MACD · Normalised performance · Correlation matrix |
| ⚠️ **Risk** | Per-stock risk table (Return, Volatility, Sharpe, Max Drawdown, VaR 95/99%, CVaR) · Rolling volatility · Drawdown chart |
| ⚙️ **Optimizer** | Portfolio weights pie & bar chart · Key portfolio metrics · 5-strategy comparison table |
| 🔮 **Predict** | Self-contained (sidebar hides). Three headline outputs — **Movement** (up/down + conviction), **Future Price** (1–4 mo median + range), **Accuracy** (backtested). Plus: long-term forecast fan chart & table · evaluation cards · live-context panel (news / Screener.in / analysts / earnings) · strategy backtest vs buy&hold/random · classifier leaderboard · feature importance · confusion matrix · Monte-Carlo simulation |
| 💡 **Insights** | Auto-generated portfolio insights and observations |

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

## Stock Movement Predictor

Prediction isn't just past prices — direction is driven by earnings, market regime, buy/sell pressure and news. The predictor separates its signals into two honest tiers:

### Tier 1 — Backtestable (fed into the ML model)

These features have real historical dates, so the walk-forward backtest genuinely validates them:

| Group | Features |
|---|---|
| **Price / technical** | Lagged returns (1/2/3/5/10d), moving-average ratios, rolling volatility, RSI(14), MACD histogram, 10-day momentum |
| **Volume / buy-sell pressure** | Volume ratio, **OBV** slope (on-balance volume), **MFI(14)** (money-flow index), **A/D** slope (accumulation-distribution) — inferring buying vs selling from where price closes within its range × volume |
| **Earnings cycle** | Days to / since the last report, earnings-window flag, last EPS surprise % (recency-decayed) — from yfinance earnings dates |
| **Market regime** | Nifty 50 (`^NSEI`) 1- & 5-day returns and trend, India VIX (`^INDIAVIX`) level & change — usually the single strongest signal for next-day direction |

### Models

| Model | Role |
|---|---|
| **Ensemble** (soft-voting) | Averages the base classifiers' probabilities — the most stable call, and the default headline pick |
| **LightGBM** | Gradient-boosted trees — best-in-class for tabular data |
| **XGBoost** | Gradient-boosted trees — the second workhorse |
| **Logistic Regression** | Scaled linear baseline — an honest, well-behaved reference |

Training is strictly **chronological** (no shuffling / look-ahead): the most-recent slice is held out as an unseen test window. Each model reports **accuracy, precision, recall, F1**, a **confusion matrix**, and a **trading backtest** (go long when it predicts "up") compared against **buy & hold** and a **random guess**.

### Tier 2 — Live context overlay (not backtested)

Free data can't provide years of point-in-time news/fundamentals, so these are **not** fed to the backtest. Instead they form a live *bias* that nudges today's call (±10 pts), clearly labelled:

- **News sentiment** — per-stock headlines (yfinance) scored with TextBlob
- **Fundamentals** — Screener.in scrape: P/E, ROCE, ROE, dividend yield, pros/cons
- **Analyst view** — yfinance recommendation + mean target-price upside

The three headline outputs are **Movement** (direction + conviction), **Future Price** (see below) and **Accuracy** (the backtested hit-rate vs a majority baseline).

## Long-Term Price Forecast (1–4 months)

A point price months out is a *distribution*, not a number — so the forecast is a **Monte-Carlo GBM** simulation calibrated to the stock's own drift & volatility, with two corrections:

- **Dampened drift** — raw historical drift over-extrapolates, so it's shrunk
- **Analyst anchor** — the yfinance mean target (~12-month view) is blended into the drift so the projection leans toward where professionals expect the stock to go

For each horizon (1 / 2 / 3 / 4 months) it reports the **median** plus **50% (p25–p75)** and **90% (p5–p95)** confidence bands, rendered as a fan chart and a table.

> No model predicts stock prices reliably. **1-day direction accuracy realistically hovers near 50%** and the tool states this plainly — the value is a transparent, well-evaluated pipeline that tells you exactly how much (if any) edge each model has, and never dresses up an un-backtested signal as a validated one.

---

## APIs & Data Sources

| Source | Used For | Key Required |
|---|---|---|
| **Yahoo Finance** (`yfinance`) | Stock prices, OHLCV, live quotes, index data (Nifty/VIX), **earnings dates & EPS surprise**, **per-stock news**, analyst targets & recommendations | No |
| **Screener.in** | Fundamentals scrape — P/E, ROCE, ROE, dividend yield, pros/cons (public company page) | No |
| **NewsAPI.org** | Live market-wide news headlines (Terminal tab) | Optional — free tier |
| **ET Markets / Moneycontrol RSS** | News fallback if NewsAPI key missing | No |
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
├── app.py                 — Streamlit UI: 7 tabs, Groww theme, auto-refresh, sidebar hide
├── data_fetcher.py        — yfinance data layer, live quotes, OHLCV
├── fundamentals.py        — Nifty-50 universe · earnings & market-regime features · Screener.in / news / analyst live overlay
├── movement_predictor.py  — Stock Movement Predictor: feature engineering, LGBM/XGB/LogReg + soft-voting ensemble, backtest & evaluation
├── price_forecast.py      — Long-term (1–4 mo) Monte-Carlo price forecast with dampened drift + analyst anchor
├── risk_engine.py         — Risk metrics: VaR, CVaR, Sharpe, Beta, drawdown
├── optimizer.py           — Markowitz optimisation, 5 strategies
├── predictor.py           — Legacy time-series price engine; still supplies technical indicators for the Charts tab
├── terminal_tab.py        — Terminal tab UI: 7 panels, full-screen layout
├── terminal_utils.py      — Terminal back-end: index quotes, order book sim, news, signals, heatmap
├── requirements.txt       — Python dependencies
└── run.sh                 — One-command launcher
```

### Module Breakdown

**`data_fetcher.py`**
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

**`fundamentals.py`**
- `NIFTY_50` — the app's `{company_name: ticker}` universe (NSE)
- `earnings_features(ticker, index)` / `market_features(index)` — backtestable earnings-cycle + Nifty/VIX regime features aligned point-in-time to a date index
- `fetch_news_sentiment(ticker)` — per-stock headlines + TextBlob score
- `fetch_screener(ticker)` — Screener.in fundamentals scrape (requests + BeautifulSoup)
- `fetch_analyst_snapshot(ticker)` — yfinance recommendation, target price, valuation
- `fundamental_context(ticker)` — reduces the live overlays to one bias in `[-1, +1]` with per-signal contributions

**`movement_predictor.py`**
- `run_movement_analysis(ohlcv, models, test_frac, ticker)` — single entry point: engineers features, trains the base classifiers, builds the soft-voting ensemble, backtests each, and returns results + the selected model
- `engineer_features(ohlcv, ticker)` — ~27 features (price/technical + volume pressure + earnings + market regime), chronological with a kept live-forecast row
- `_build_model` — LightGBM / XGBoost / Logistic Regression (RandomForest dropped)
- `_assemble` — computes accuracy/precision/recall/F1, confusion matrix, trading backtest, next-day expected price
- `monte_carlo_paths(close)` — GBM path simulation for the tab's Monte-Carlo panel

**`price_forecast.py`**
- `forecast_price(close, analyst_target, months=(1,2,3,4))` — Monte-Carlo GBM with dampened drift blended toward the analyst target; returns median + 50%/90% bands per horizon and path arrays for charting

**`predictor.py`** (legacy)
- `compute_technical_indicators(ohlcv)` — indicator overlays still used by the Charts tab
- `run_all_predictions(...)` — the previous time-series price engine; retained but no longer wired into the Predict tab

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
- Pick a **preset basket** (IT, Banking, Auto, Pharma, Energy, FMCG) or choose from the **Nifty 50** list
- Changing the selection automatically re-fetches data

### 2 — Set parameters
- **Lookback period** — 6 months to 5 years
- **Optimization strategy** — choose one of 5 strategies
- **Risk-free rate** — defaults to 6.5% (Indian 10-yr G-Sec)

### 3 — Run analysis
Click **▶ Run Analysis** to load historical price data and compute all metrics.

### 4 — Predict movement & price
- Switch to the **🔮 Predict** tab (self-contained — the sidebar hides)
- Pick a **Nifty-50 stock**, a **history window** and a **backtest size** inline; optionally toggle classifiers under **⚙️ Classifiers & sources**
- Click **🎯 Run Predictor** — the ensemble trains on price/volume/earnings/market features and is walk-forward backtested, the live context (news / Screener.in / analysts) is read, and the 1–4 month price forecast is projected
- Read the three headline cards — **Movement**, **Future Price**, **Accuracy** — then the forecast chart/table, evaluation, live-context panel, backtest and leaderboard below (cached per stock · history · models · test%)

### 5 — Use the Terminal
- Click the **🖥️ Terminal** tab — the sidebar hides for maximum screen space
- Select stocks independently for the TA chart, order book, and strategy signals
- News refreshes every 60 s · Index quotes refresh every 30 s · Order book updates every 5 s

---

## Universe — Nifty 50

The app operates on the 50 Nifty constituents (NSE). Preset baskets group them by sector:

| Sector | Example Stocks |
|---|---|
| IT & Technology | TCS, Infosys, Wipro, HCL Tech, Tech Mahindra |
| Banking & Finance | HDFC Bank, ICICI Bank, SBI, Kotak, Axis, IndusInd, Bajaj Finance, Bajaj Finserv, Shriram Finance |
| Auto | Maruti, M&M, Hero MotoCorp, Bajaj Auto, Tata Motors, Eicher |
| Pharma & Healthcare | Sun Pharma, Dr Reddy's, Cipla, Apollo Hospitals |
| Energy | Reliance, ONGC, NTPC, Coal India, BPCL, Power Grid |
| FMCG | HUL, ITC, Nestle, Tata Consumer, Titan |
| Metals & Materials | JSW Steel, Hindalco, Tata Steel, Grasim, UltraTech Cement, Asian Paints |
| Others | Bharti Airtel, L&T, Adani Enterprises, Adani Ports, Trent, Eternal (Zomato) |

> Extending beyond Nifty 50 is a one-line change — add names to `NIFTY_50` in `fundamentals.py`.

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
torch>=2.2.0              # LSTM prediction model
textblob>=0.18.0           # News sentiment
feedparser>=6.0.11         # RSS news fallback
pydeck>=0.8.0              # (installed, map replaced with Plotly Scattergeo)
requests>=2.31.0           # NewsAPI / Screener.in HTTP calls
python-dotenv>=1.0.0       # .env for NEWS_API_KEY
lxml>=5.0.0                # yfinance earnings dates + Screener.in parsing
beautifulsoup4>=4.12.0     # Screener.in fundamentals scraping
```

---

## Technical Notes

- **Data source:** Yahoo Finance via `yfinance`. NSE prices are delayed ~15 minutes.
- **Live quotes:** Uses `fast_info.last_price` and `fast_info.previous_close` — no history call on live feed.
- **Auto-refresh:** Page reruns every 5 seconds using `streamlit-autorefresh`.
- **Terminal sidebar:** Injected CSS hides `[data-testid="stSidebar"]` when Terminal tab is active; other tabs restore it normally.
- **Order book:** Fully simulated — yfinance does not provide real Level 2 data. Clearly labelled in the UI.
- **Strategy signals:** Computed on tab activation only (cached in `st.session_state`) — not on every rerun.
- **Indicators:** EMA, RSI, ATR, ADX, Bollinger Bands implemented in pure pandas/numpy — no `pandas-ta` dependency (incompatible with Python 3.14).
- **Tata Motors tickers:** Yahoo Finance renamed `TATAMOTORS.NS` on 2026-03-17. Use `TMCV.NS` (commercial vehicles) and `TMPV.NS` (passenger vehicles, demerged entity).
- **No look-ahead:** The Movement Predictor splits data chronologically — the most-recent slice is a held-out test window the model never trains on. The last valid row (with unknown target) is kept solely to forecast tomorrow.
- **Backtestable vs overlay:** Only signals with real historical dates (price, volume, earnings, market regime) feed the ML model and its backtest. News, Screener.in fundamentals and analyst views are a **live overlay** that nudges today's call by ±10 pts — never fed to the backtest and clearly labelled as such in the UI.
- **Predicted price:** Direction classifiers don't output a price, so the next-day close is estimated as `last_close × (1 + P(up)·avg_up_move + P(down)·avg_down_move)`. The long-term (1–4 mo) forecast is a separate Monte-Carlo range, not a point regression.
- **Screener.in scrape:** Public company page parsed with `requests` + BeautifulSoup, cached ~6 h and fails soft (empty) so the predictor keeps working offline.
- **Earnings data:** yfinance `get_earnings_dates` requires `lxml`; the earnings-cycle features fall back to neutral defaults if it's unavailable.
- **Dev hot-reload:** `predictor.py` still imports torch (legacy engine), and Streamlit's file-watcher can segfault when it reloads while torch is imported. If you're editing live, launch with `streamlit run app.py --server.fileWatcherType none`.
- **Plotly vline bug:** Uses `add_shape` + `add_annotation` instead of `add_vline` to avoid a bug in Plotly 6.x with string-date x-values and `annotation_position`.
- **NumPy version:** CVXPY requires `numpy < 2.0`. The `requirements.txt` pins this.

---

## License

MIT — free to use, modify, and distribute.
