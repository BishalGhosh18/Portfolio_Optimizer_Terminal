# NSE/BSE Portfolio Optimizer

A Bloomberg-style terminal UI for Indian equity portfolio analysis — built with Streamlit.

## Features

- **Live quotes** — real-time prices, 52-week high/low, volume, market cap
- **130+ stocks** — full NSE & BSE universe across all major sectors
- **Interactive charts** — candlestick, volume, RSI, MACD, Bollinger Bands
- **Risk analytics** — VaR, CVaR, Sharpe, Sortino, Calmar, Beta, Max Drawdown
- **Portfolio optimisation** — 5 strategies: Max Sharpe, Min Volatility, Risk Parity, Max Diversification, Equal Weight
- **Efficient frontier** — Monte Carlo simulation
- **Risk scorecard** — composite risk score with radar chart
- **Price prediction** — ARIMA, Linear Regression, Random Forest, Monte Carlo GBM, EMA Trend

---

## Quick Start

### Option A — One-command launcher (recommended)

```bash
git clone https://github.com/your-username/portfolio-optimizer
cd portfolio-optimizer
bash run.sh
```

The script auto-detects conda/venv and installs dependencies if needed. Open [http://localhost:8501](http://localhost:8501).

### Option B — Manual setup

**With conda (recommended for avoiding NumPy/CVXPY conflicts):**

```bash
conda create -n portfolio_optimizer python=3.11 -y
conda activate portfolio_optimizer
pip install -r requirements.txt
streamlit run app.py
```

**With plain pip:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

- Python 3.10 or 3.11 (3.12 may have CVXPY conflicts)
- Internet connection (live data via yfinance)

All Python dependencies are in `requirements.txt`.

---

## Usage

1. **Select Exchange** — NSE or BSE in the sidebar
2. **Pick stocks** — use a preset basket (Tata Group, Top IT, Banking, etc.) or search any company
3. **Set lookback** — 6 months to 5 years
4. **Click ► RUN ANALYSIS**
5. Navigate tabs: **LIVE FEED → CHARTS → RISK → OPTIMIZER → FRONTIER → SCORECARD → PREDICT**

To run a price prediction:
- Select a stock and model in the sidebar
- Set the forecast horizon (days)
- Click **► RUN PREDICTION**

---

## Project Structure

```
portfolio-optimizer/
├── app.py            # Streamlit UI (terminal theme)
├── data_fetcher.py   # yfinance data layer, stock universe
├── risk_engine.py    # Risk metrics (VaR, CVaR, Sharpe, Beta, …)
├── optimizer.py      # Markowitz optimisation strategies
├── predictor.py      # ML/statistical prediction models
├── requirements.txt  # Python dependencies
└── run.sh            # Cross-platform launcher
```

---

## Supported Sectors

Nifty 50, Nifty 500 and popular counters across:
Banking & Finance · IT & Technology · Pharma · Auto · Energy & Power ·
FMCG · Infrastructure · Metals & Mining · Telecom · Real Estate · Media

---

## Notes

- Data is sourced from Yahoo Finance via `yfinance`. Prices are delayed by ~15 min for NSE/BSE.
- CVXPY requires NumPy < 2.0 — the `requirements.txt` pins this.
- For best results use Python 3.11 inside a conda environment.
# Portfolio_Optimizer_Terminal
