"""
Groww-Style NSE/BSE Portfolio Optimizer & Risk Dashboard
=========================================================
• Clean, modern UI inspired by Groww
• White cards, purple/teal accents, Inter font
• Live price data, risk analysis, portfolio optimisation
• Future price prediction (ARIMA · Linear Regression · Random Forest ·
  Monte Carlo GBM · EMA Trend)

Run:  streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

from data_fetcher import (
    get_stock_universe, fetch_price_data, fetch_ohlcv,
    fetch_all_live_quotes, fetch_benchmark, compute_returns, RISK_FREE_RATE,
)
from risk_engine import (
    annualised_return, annualised_volatility, sharpe_ratio,
    drawdown_series, var_summary,
    correlation_matrix, rolling_volatility,
    risk_scorecard, portfolio_returns, TRADING_DAYS,
)
from optimizer import run_strategy, all_strategies_summary, STRATEGIES
from predictor import PREDICTION_MODELS, run_prediction, compute_technical_indicators

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer | Groww Style",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Groww-style CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F6F7F8 !important;
    color: #1B2236 !important;
}
.main .block-container {
    background-color: #F6F7F8 !important;
    padding-top: 1rem;
    max-width: 100%;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1B2236 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #2D3748 !important;
    border: 1px solid #4A5568 !important;
    color: #E5E7EB !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #5367FF 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(83,103,255,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(83,103,255,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] label {
    color: #9CA3AF !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Main buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #5367FF 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(83,103,255,0.25) !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 24px rgba(83,103,255,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}
[data-testid="stTabs"] button {
    background: transparent !important;
    color: #6B7280 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #5367FF 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(83,103,255,0.35) !important;
}
[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
    background: #F3F4F6 !important;
    color: #1B2236 !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: #1B2236 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] { font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; font-weight: 600 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

/* ── Divider ── */
hr { border-color: #E5E7EB !important; }

/* ── Spinner ── */
.stSpinner > div { border-color: #5367FF transparent transparent transparent !important; }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(135deg, #5367FF, #8B5CF6) !important; border-radius: 4px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F6F7F8; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ── Custom components ── */

.gw-header {
    background: linear-gradient(135deg, #1B2236 0%, #2D3748 100%);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(27,34,54,0.15);
}
.gw-header-title {
    color: #FFFFFF;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.gw-header-sub {
    color: #9CA3AF;
    font-size: 0.78rem;
    margin-top: 2px;
}
.gw-live-badge {
    background: rgba(0,208,156,0.15);
    border: 1px solid rgba(0,208,156,0.4);
    color: #00D09C;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.gw-dot {
    width: 7px; height: 7px;
    background: #00D09C;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    display: inline-block;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(0.8); }
}

.gw-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    margin-bottom: 12px;
    border: 1px solid #F3F4F6;
    transition: box-shadow 0.2s ease;
}
.gw-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.gw-stock-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F3F4F6;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}
.gw-stock-card:hover {
    box-shadow: 0 4px 16px rgba(83,103,255,0.1);
    border-color: rgba(83,103,255,0.2);
    transform: translateY(-1px);
}
.gw-stock-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1B2236;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.gw-stock-ticker {
    font-size: 0.7rem;
    color: #9CA3AF;
    font-weight: 400;
}
.gw-price {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1B2236;
    margin-top: 6px;
}
.gw-change-up {
    font-size: 0.78rem;
    font-weight: 600;
    color: #00D09C;
    background: rgba(0,208,156,0.1);
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 4px;
}
.gw-change-down {
    font-size: 0.78rem;
    font-weight: 600;
    color: #FF5370;
    background: rgba(255,83,112,0.1);
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 4px;
}
.gw-change-flat {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6B7280;
    background: #F3F4F6;
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 4px;
}
.gw-meta {
    font-size: 0.68rem;
    color: #9CA3AF;
    margin-top: 8px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px;
}
.gw-meta span { color: #6B7280; }
.gw-meta b { color: #374151; }

.gw-section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1B2236;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.gw-section-title::before {
    content: '';
    width: 3px; height: 18px;
    background: linear-gradient(180deg, #5367FF, #8B5CF6);
    border-radius: 2px;
    display: inline-block;
}

.gw-ticker-bar {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F3F4F6;
    overflow-x: auto;
    white-space: nowrap;
    margin-bottom: 16px;
    font-size: 0.78rem;
    font-weight: 500;
}
.gw-tick-up   { color: #00D09C; }
.gw-tick-down { color: #FF5370; }
.gw-tick-flat { color: #9CA3AF; }

.gw-stat-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F3F4F6;
}
.gw-stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.gw-stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1B2236;
    margin-top: 4px;
}
.gw-stat-sub {
    font-size: 0.75rem;
    margin-top: 2px;
    font-weight: 600;
}

.gw-pill-green {
    background: rgba(0,208,156,0.1);
    color: #00D09C;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}
.gw-pill-red {
    background: rgba(255,83,112,0.1);
    color: #FF5370;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}
.gw-pill-purple {
    background: rgba(83,103,255,0.1);
    color: #5367FF;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
}

.gw-score-bar-wrap {
    background: #F3F4F6;
    border-radius: 20px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.gw-score-bar {
    height: 8px;
    border-radius: 20px;
}

.gw-welcome {
    background: linear-gradient(135deg, #1B2236 0%, #2D3748 50%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    color: #FFFFFF;
    margin: 20px 0;
}

/* model info box */
.gw-info-box {
    background: rgba(83,103,255,0.05);
    border: 1px solid rgba(83,103,255,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #374151;
    margin-bottom: 16px;
}
.gw-info-box b { color: #5367FF; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

GW_GREEN  = "#00D09C"
GW_RED    = "#FF5370"
GW_PURPLE = "#5367FF"
GW_NAVY   = "#1B2236"
GW_GRAY   = "#F6F7F8"

PALETTE = [GW_PURPLE, "#8B5CF6", GW_GREEN, GW_RED,
           "#F59E0B", "#06B6D4", "#EC4899", "#10B981", "#F97316", "#6366F1"]

def _vline(fig: go.Figure, x_val, label: str = "Today"):
    xs = str(x_val.date()) if hasattr(x_val, "date") else str(x_val)
    fig.add_shape(type="line", x0=xs, x1=xs, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=GW_PURPLE, width=1.5, dash="dot"))
    fig.add_annotation(x=xs, y=0.97, yref="paper", text=label,
                       showarrow=False,
                       font=dict(color=GW_PURPLE, size=10, family="Inter"),
                       xanchor="left")

def groww_fig(fig: go.Figure, height: int = 380, title: str = "") -> go.Figure:
    """Apply Groww-style light theme to a plotly figure."""
    fig.update_layout(
        height=height,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter", size=13, color=GW_NAVY),
            x=0.01, y=0.99,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter", color="#6B7280", size=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
            font=dict(family="Inter", color="#374151", size=10),
        ),
        xaxis=dict(
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            tickfont=dict(family="Inter", color="#9CA3AF", size=9),
            title_font=dict(family="Inter", color="#6B7280"),
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            tickfont=dict(family="Inter", color="#9CA3AF", size=9),
            title_font=dict(family="Inter", color="#6B7280"),
            showgrid=True, zeroline=False,
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(
            bgcolor="#1B2236",
            font=dict(family="Inter", color="#FFFFFF", size=11),
            bordercolor="#2D3748",
        ),
    )
    return fig

def _risk_color(level: str) -> str:
    return {"Low": GW_GREEN, "Moderate": "#F59E0B",
            "High": GW_RED, "Very High": "#7F1D1D"}.get(level, "#6B7280")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 12px;">
        <div style="font-size:1.2rem;font-weight:700;color:#FFFFFF;letter-spacing:-0.02em;">
            📈 Portfolio Optimizer
        </div>
        <div style="font-size:0.72rem;color:#9CA3AF;margin-top:2px;">NSE · BSE · Live Data</div>
    </div>
    <hr style="border-color:#2D3748;margin:4px 0 16px;">
    """, unsafe_allow_html=True)

    auto_refresh = st.toggle("Live Auto-Refresh", value=True)
    refresh_secs = st.select_slider(
        "Refresh Every",
        options=[5, 10, 15, 30, 60],
        value=5,
        format_func=lambda x: f"{x}s",
        disabled=not auto_refresh,
    )

    exchange = st.selectbox("Exchange", ["NSE", "BSE"])
    universe = get_stock_universe(exchange)
    all_names = sorted(universe.keys())

    PRESETS = {
        "── Select Preset ──": [],
        "🏢 Tata Group":     [n for n in all_names if "Tata" in n or n == "Titan Company" or n == "Trent" or n == "Voltas"],
        "💻 Top IT":         [n for n in all_names if any(k in n for k in ["TCS","Infosys","Wipro","HCL","Tech Mahindra"])],
        "🏦 Top Banking":    [n for n in all_names if any(k in n for k in ["HDFC","ICICI","Kotak","State Bank","Axis"])],
        "🚗 Top Auto":       [n for n in all_names if any(k in n for k in ["Maruti","Bajaj Auto","Hero","Tata Motors","Mahindra","Eicher"])],
        "💊 Pharma":         [n for n in all_names if any(k in n for k in ["Sun Pharma","Dr. Reddy","Cipla","Divi","Lupin","Aurobindo"])],
        "⚡ Energy":         [n for n in all_names if any(k in n for k in ["Reliance","ONGC","NTPC","Coal India","BPCL","Power Grid"])],
    }
    preset = st.selectbox("Preset Basket", list(PRESETS.keys()))
    preset_vals = [v for v in PRESETS.get(preset, []) if v in all_names]
    default_sel = preset_vals if preset_vals else [n for n in all_names if "Tata" in n][:5]

    selected_names = st.multiselect(
        "Select Companies",
        options=all_names,
        default=default_sel[:8],
        help="Pick 2–15 companies",
    )

    st.markdown('<hr style="border-color:#2D3748;margin:12px 0;">', unsafe_allow_html=True)

    lookback    = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "3y", "5y"], index=1)
    opt_strategy = st.selectbox("Optimization Strategy", STRATEGIES)
    rf_rate     = st.number_input("Risk-Free Rate (%)", value=RISK_FREE_RATE * 100, step=0.1, format="%.2f") / 100

    st.markdown('<hr style="border-color:#2D3748;margin:12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#9CA3AF;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Price Prediction</div>', unsafe_allow_html=True)
    pred_company = st.selectbox("Stock to Predict", options=all_names,
                                index=all_names.index(selected_names[0]) if selected_names else 0)
    pred_model   = st.selectbox("Model", list(PREDICTION_MODELS.keys()))
    pred_horizon = st.slider("Forecast Horizon (Days)", 5, 180, 30)

    st.markdown('<hr style="border-color:#2D3748;margin:12px 0;">', unsafe_allow_html=True)
    run_btn  = st.button("▶  Run Analysis", use_container_width=True)
    pred_btn = st.button("🔮  Run Prediction", use_container_width=True)

    st.markdown(
        f'<div style="font-size:0.68rem;color:#4B5563;margin-top:12px;text-align:center;">'
        f'{datetime.now().strftime("%d %b %Y, %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )

# ── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="gw_refresh")

# ── Welcome screen ────────────────────────────────────────────────────────────
if len(selected_names) < 2:
    st.markdown("""
    <div class="gw-welcome">
        <div style="font-size:2.5rem;margin-bottom:12px;">📈</div>
        <div style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;">Portfolio Optimizer</div>
        <div style="color:#9CA3AF;margin-top:8px;font-size:0.9rem;">
            Select at least 2 companies from the sidebar to get started
        </div>
        <div style="display:flex;gap:12px;justify-content:center;margin-top:24px;flex-wrap:wrap;">
            <span class="gw-pill-green">✓ Live Prices</span>
            <span class="gw-pill-purple">✓ Risk Analytics</span>
            <span class="gw-pill-green">✓ Portfolio Optimization</span>
            <span class="gw-pill-purple">✓ Price Prediction</span>
        </div>
        <div style="color:#6B7280;font-size:0.78rem;margin-top:20px;">
            130+ NSE & BSE stocks · 5 optimization strategies · 5 prediction models
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("data_loaded", False), ("prices", None), ("quotes", {}),
             ("pred_result", None), ("last_selection", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Data fetch ────────────────────────────────────────────────────────────────
tickers      = {name: universe[name] for name in selected_names}
ticker_list  = list(tickers.values())
company_list = list(tickers.keys())

# Auto-reset if user changes the stock selection
if sorted(company_list) != sorted(st.session_state.last_selection):
    st.session_state.data_loaded    = False
    st.session_state.prices         = None
    st.session_state.last_selection = sorted(company_list)

if run_btn or not st.session_state.data_loaded:
    with st.spinner("Fetching market data..."):
        prices_raw = fetch_price_data(ticker_list, period=lookback)
        prices_raw.columns = [
            next((n for n, t in tickers.items() if t == c), c)
            for c in prices_raw.columns
        ]
        prices_raw = prices_raw[[n for n in company_list if n in prices_raw.columns]].dropna(how="all")
        st.session_state.prices = prices_raw
        st.session_state.data_loaded = True

if st.session_state.data_loaded and company_list:
    quotes_raw = fetch_all_live_quotes(company_list, exchange)
    st.session_state.quotes = quotes_raw

prices: pd.DataFrame = st.session_state.prices
quotes: dict         = st.session_state.quotes

if prices is None or prices.empty:
    st.error("No price data retrieved — check your network connection.")
    st.stop()

valid_cols = prices.dropna(axis=1, how="all").columns.tolist()
prices = prices[valid_cols].ffill().bfill().dropna(axis=1, thresh=10)
if prices.shape[1] < 2:
    st.error("Insufficient data — select more companies.")
    st.stop()

returns = compute_returns(prices)

# ── Page header ───────────────────────────────────────────────────────────────
n_up   = sum(1 for n in prices.columns if (quotes.get(n, {}).get("change_pct") or 0) > 0)
n_down = len(prices.columns) - n_up
now_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")

st.markdown(f"""
<div class="gw-header">
    <div>
        <div class="gw-header-title">📈 Portfolio Analytics</div>
        <div class="gw-header-sub">
            {exchange} &nbsp;·&nbsp; {len(prices.columns)} securities &nbsp;·&nbsp; {lookback}
            &nbsp;·&nbsp;
            <span style="color:{GW_GREEN};">▲ {n_up}</span>
            &nbsp;
            <span style="color:{GW_RED};">▼ {n_down}</span>
        </div>
    </div>
    <div style="text-align:right;">
        <div class="gw-live-badge">
            <span class="gw-dot"></span> LIVE
        </div>
        <div style="color:#6B7280;font-size:0.7rem;margin-top:6px;">{now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Ticker strip — uses full selected list, not just loaded prices ─────────────
parts = []
for name in company_list:
    q     = quotes.get(name, {})
    price = q.get("price")
    chg   = q.get("change_pct")
    sym   = universe.get(name, name).replace(".NS", "").replace(".BO", "")
    if price is not None:
        c     = chg or 0
        arrow = "▲" if c > 0 else ("▼" if c < 0 else "─")
        css   = "gw-tick-up" if c > 0 else ("gw-tick-down" if c < 0 else "gw-tick-flat")
        parts.append(f'<span class="{css}"><b>{sym}</b> ₹{price:,.1f} {arrow}{abs(c):.2f}%</span>')
    else:
        parts.append(f'<span class="gw-tick-flat"><b>{sym}</b> ─</span>')

st.markdown(
    '<div class="gw-ticker-bar">'
    + ' &nbsp;&nbsp;<span style="color:#E5E7EB;">|</span>&nbsp;&nbsp; '.join(parts)
    + '</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_charts, tab_risk, tab_opt, tab_score, tab_pred = st.tabs([
    "📊 Live Feed", "📉 Charts", "⚠️ Risk", "⚙️ Optimizer", "🎯 Scorecard", "🔮 Predict",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown(f'<div class="gw-section-title">Live Market Feed &nbsp;<span style="font-size:0.72rem;color:#9CA3AF;font-weight:400;">Updated {datetime.now().strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)

    cols_per_row = 3
    name_list    = company_list
    for row_start in range(0, len(name_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, name in enumerate(name_list[row_start:row_start + cols_per_row]):
            q      = quotes.get(name, {})
            price  = q.get("price")
            chg    = q.get("change_pct")
            hi52   = q.get("52w_high")
            lo52   = q.get("52w_low")
            volume = q.get("volume")
            mktcap = q.get("market_cap")
            sym    = universe.get(name, name).replace(".NS", "").replace(".BO", "")
            c_val  = chg or 0

            price_str = f"₹{price:,.2f}" if price else "─"
            chg_str   = f"{'▲' if c_val>0 else '▼' if c_val<0 else '─'} {abs(c_val):.2f}%" if price else "─"
            chg_cls   = "gw-change-up" if c_val > 0 else ("gw-change-down" if c_val < 0 else "gw-change-flat")

            meta_html = ""
            if hi52:   meta_html += f"<div><span>52W High</span> <b>₹{hi52:,.1f}</b></div>"
            if lo52:   meta_html += f"<div><span>52W Low</span>  <b>₹{lo52:,.1f}</b></div>"
            if volume: meta_html += f"<div><span>Volume</span>   <b>{volume/1e5:.1f}L</b></div>"
            if mktcap: meta_html += f"<div><span>Mkt Cap</span> <b>₹{mktcap/1e9:.0f}B</b></div>"

            with cols[ci]:
                st.markdown(f"""
                <div class="gw-stock-card">
                    <div class="gw-stock-name">{name}</div>
                    <div class="gw-stock-ticker">{sym} · {exchange}</div>
                    <div class="gw-price">{price_str}</div>
                    <span class="{chg_cls}">{chg_str}</span>
                    <div class="gw-meta">{meta_html}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Market breadth + daily change bar ──
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="gw-section-title">Market Breadth</div>', unsafe_allow_html=True)
        gainers = [n for n in prices.columns if (quotes.get(n, {}).get("change_pct") or 0) > 0]
        losers  = [n for n in prices.columns if (quotes.get(n, {}).get("change_pct") or 0) < 0]
        flat    = [n for n in prices.columns if (quotes.get(n, {}).get("change_pct") or 0) == 0]
        total   = len(prices.columns)

        for label, lst, color in [("Gainers", gainers, GW_GREEN), ("Losers", losers, GW_RED), ("Flat", flat, "#9CA3AF")]:
            pct = int(len(lst) / total * 100) if total else 0
            st.markdown(f"""
            <div class="gw-card" style="padding:12px 16px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;color:{color};font-size:0.85rem;">{label}</span>
                    <span style="font-weight:700;color:{color};font-size:1rem;">{len(lst)}</span>
                </div>
                <div class="gw-score-bar-wrap" style="margin-top:8px;">
                    <div class="gw-score-bar" style="width:{pct}%;background:{color};"></div>
                </div>
                <div style="font-size:0.68rem;color:#9CA3AF;margin-top:4px;">{', '.join(n.split()[0] for n in lst[:3])}{' ...' if len(lst)>3 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="gw-section-title">Daily Change %</div>', unsafe_allow_html=True)
        chg_data = {n: (quotes.get(n, {}).get("change_pct") or 0) for n in prices.columns}
        chg_series = pd.Series(chg_data).sort_values()
        fig_bar = go.Figure(go.Bar(
            x=chg_series.index.tolist(),
            y=chg_series.values.tolist(),
            marker_color=[GW_GREEN if v >= 0 else GW_RED for v in chg_series.values],
            marker_line_width=0,
            text=[f"{v:+.2f}%" for v in chg_series.values],
            textposition="outside",
            textfont=dict(family="Inter", size=9, color="#6B7280"),
        ))
        groww_fig(fig_bar, 280, "")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Benchmark ──
    try:
        bench = fetch_benchmark(period=lookback)
        if bench is not None and not bench.empty:
            bench_ret = bench.pct_change().dropna()
            st.markdown('<div class="gw-section-title">Benchmark Performance</div>', unsafe_allow_html=True)
            fig_bench = go.Figure()
            for col, colour in zip(bench_ret.columns, [GW_PURPLE, "#F59E0B"]):
                cumret = (1 + bench_ret[col]).cumprod() - 1
                fig_bench.add_trace(go.Scatter(
                    x=cumret.index, y=cumret.values * 100,
                    mode="lines", name=col,
                    line=dict(width=2, color=colour),
                    fill="tozeroy",
                    fillcolor=f"rgba({','.join(str(int(colour.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.05)",
                ))
            groww_fig(fig_bench, 220, "")
            st.plotly_chart(fig_bench, use_container_width=True)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown('<div class="gw-section-title">Price Charts & Technicals</div>', unsafe_allow_html=True)
    chart_name = st.selectbox("Select Stock", list(prices.columns), key="chart_sel")
    ticker_sym = universe.get(chart_name, chart_name)

    with st.spinner(f"Loading OHLCV for {chart_name}..."):
        try:
            ohlcv     = fetch_ohlcv(ticker_sym, period=lookback)
            has_ohlcv = ohlcv is not None and not ohlcv.empty
        except Exception:
            has_ohlcv = False

    if has_ohlcv:
        ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz else ohlcv.index

        fig_candle = go.Figure(go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"],
            increasing_line_color=GW_GREEN, decreasing_line_color=GW_RED,
            increasing_fillcolor=f"rgba(0,208,156,0.3)", decreasing_fillcolor=f"rgba(255,83,112,0.3)",
            name="OHLC",
        ))
        for win, colour in [(20, "#F59E0B"), (50, GW_PURPLE), (200, "#EC4899")]:
            if len(ohlcv) > win:
                ma = ohlcv["Close"].rolling(win).mean()
                fig_candle.add_trace(go.Scatter(
                    x=ohlcv.index, y=ma, mode="lines", name=f"MA{win}",
                    line=dict(width=1.2, color=colour, dash="dot"),
                ))
        if len(ohlcv) > 20:
            mid = ohlcv["Close"].rolling(20).mean()
            std = ohlcv["Close"].rolling(20).std()
            fig_candle.add_trace(go.Scatter(x=ohlcv.index, y=mid + 2*std, mode="lines",
                name="BB+", line=dict(width=1, color="rgba(83,103,255,0.3)"), fill=None))
            fig_candle.add_trace(go.Scatter(x=ohlcv.index, y=mid - 2*std, mode="lines",
                name="BB-", line=dict(width=1, color="rgba(83,103,255,0.3)"),
                fill="tonexty", fillcolor="rgba(83,103,255,0.04)"))

        groww_fig(fig_candle, 440, f"{chart_name}")
        fig_candle.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

        fig_vol = go.Figure(go.Bar(
            x=ohlcv.index, y=ohlcv.get("Volume", pd.Series()),
            marker_color=[f"rgba(0,208,156,0.5)" if c >= o else f"rgba(255,83,112,0.5)"
                          for c, o in zip(ohlcv["Close"], ohlcv["Open"])],
            marker_line_width=0,
            name="Volume",
        ))
        groww_fig(fig_vol, 160, "Volume")
        st.plotly_chart(fig_vol, use_container_width=True)

        try:
            ind = compute_technical_indicators(ohlcv)
            c1, c2 = st.columns(2)
            with c1:
                if "RSI_14" in ind.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=ind.index, y=ind["RSI_14"], mode="lines",
                        line=dict(color=GW_PURPLE, width=2), name="RSI(14)"))
                    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(255,83,112,0.06)", line_width=0)
                    fig_rsi.add_hrect(y0=0, y1=30,  fillcolor="rgba(0,208,156,0.06)",   line_width=0)
                    fig_rsi.add_hline(y=70, line_color=GW_RED,   line_dash="dot", line_width=1)
                    fig_rsi.add_hline(y=30, line_color=GW_GREEN, line_dash="dot", line_width=1)
                    groww_fig(fig_rsi, 220, "RSI (14)")
                    fig_rsi.update_layout(yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_rsi, use_container_width=True)
            with c2:
                if "MACD" in ind.columns and "MACD_Signal" in ind.columns:
                    hist = ind["MACD"] - ind["MACD_Signal"]
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Bar(x=ind.index, y=hist,
                        marker_color=[f"rgba(0,208,156,0.5)" if v >= 0 else f"rgba(255,83,112,0.5)" for v in hist],
                        marker_line_width=0, name="Histogram"))
                    fig_macd.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], mode="lines",
                        name="MACD", line=dict(color=GW_PURPLE, width=1.5)))
                    fig_macd.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], mode="lines",
                        name="Signal", line=dict(color="#F59E0B", width=1.5)))
                    groww_fig(fig_macd, 220, "MACD")
                    st.plotly_chart(fig_macd, use_container_width=True)
        except Exception:
            pass
    else:
        fig_line = go.Figure(go.Scatter(
            x=prices.index, y=prices[chart_name], mode="lines", name=chart_name,
            line=dict(color=GW_PURPLE, width=2),
            fill="tozeroy", fillcolor="rgba(83,103,255,0.06)",
        ))
        groww_fig(fig_line, 400, chart_name)
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="gw-section-title">Normalised Performance (Base 100)</div>', unsafe_allow_html=True)
    normed = prices / prices.iloc[0] * 100
    fig_norm = go.Figure()
    for i, col in enumerate(normed.columns):
        fig_norm.add_trace(go.Scatter(x=normed.index, y=normed[col], mode="lines", name=col,
            line=dict(width=1.8, color=PALETTE[i % len(PALETTE)])))
    groww_fig(fig_norm, 360, "")
    st.plotly_chart(fig_norm, use_container_width=True)

    st.markdown('<div class="gw-section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = correlation_matrix(returns)
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=[[0, GW_RED], [0.5, "#F9FAFB"], [1, GW_GREEN]],
        zmin=-1, zmax=1)
    fig_heat.update_traces(textfont=dict(family="Inter", size=10, color="#1B2236"))
    groww_fig(fig_heat, 420, "")
    fig_heat.update_layout(coloraxis_colorbar=dict(
        tickfont=dict(family="Inter", size=9),
        title=dict(text="Corr", font=dict(family="Inter", size=10)),
    ))
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="gw-section-title">Risk Analytics</div>', unsafe_allow_html=True)

    risk_rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 5:
            continue
        vs = var_summary(r)
        risk_rows.append({
            "Company":    col,
            "Ann Ret %":  round(annualised_return(r) * 100, 2),
            "Ann Vol %":  round(annualised_volatility(r) * 100, 2),
            "Sharpe":     round(sharpe_ratio(r, rf_rate), 2),
            "Max DD %":   round(drawdown_series(r).min() * 100, 2),
            "VaR 95%":    round(vs.get("hist_95", 0) * 100, 3),
            "CVaR 95%":   round(vs.get("cvar_95", 0) * 100, 3),
        })

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows).set_index("Company")
        st.dataframe(
            risk_df.style
                .format("{:.2f}")
                .background_gradient(subset=["Ann Ret %"], cmap="RdYlGn")
                .background_gradient(subset=["Ann Vol %"], cmap="RdYlGn_r")
                .background_gradient(subset=["Sharpe"],    cmap="RdYlGn"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        roll_vol = rolling_volatility(returns, window=21) * np.sqrt(TRADING_DAYS) * 100
        fig_rvol = go.Figure()
        for i, col in enumerate(roll_vol.columns):
            fig_rvol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol[col], mode="lines",
                name=col, line=dict(width=1.5, color=PALETTE[i % len(PALETTE)])))
        groww_fig(fig_rvol, 300, "Rolling 21-Day Volatility (%)")
        st.plotly_chart(fig_rvol, use_container_width=True)

    with c2:
        st.markdown('<div class="gw-section-title" style="font-size:0.82rem;">Drawdown Analysis</div>', unsafe_allow_html=True)
        dd_name = st.selectbox("Stock", list(returns.columns), key="dd_sel")
        dd_s = drawdown_series(returns[dd_name].dropna()) * 100
        fig_dd = go.Figure(go.Scatter(
            x=dd_s.index, y=dd_s.values, mode="lines",
            fill="tozeroy", fillcolor="rgba(255,83,112,0.1)",
            line=dict(color=GW_RED, width=1.5), name="Drawdown",
        ))
        groww_fig(fig_dd, 300, f"{dd_name} — Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="gw-section-title">Portfolio Optimizer</div>', unsafe_allow_html=True)

    with st.spinner(f"Optimizing [{opt_strategy}]..."):
        try:
            weights = run_strategy(opt_strategy, returns)
            summary = all_strategies_summary(returns)
        except Exception as e:
            st.error(f"Optimizer error: {e}")
            weights, summary = None, None

    if weights is not None:
        w_series = pd.Series(weights) * 100
        w_series = w_series[w_series > 0.01].sort_values(ascending=False)

        port_rets = portfolio_returns(returns, weights)
        ann_r = annualised_return(port_rets)
        ann_v = annualised_volatility(port_rets)
        sh    = sharpe_ratio(port_rets, rf_rate)
        mdd   = drawdown_series(port_rets).min()

        # ── Key stats ──
        m_cols = st.columns(4)
        for col, label, val, sub, sub_color in [
            (m_cols[0], "Annual Return",    f"{ann_r*100:.1f}%",  "Compounded",          GW_GREEN if ann_r>0 else GW_RED),
            (m_cols[1], "Annual Volatility",f"{ann_v*100:.1f}%",  "1 Std Dev",           GW_PURPLE),
            (m_cols[2], "Sharpe Ratio",     f"{sh:.2f}",          "Risk-adj Return",     GW_GREEN if sh>1 else "#F59E0B"),
            (m_cols[3], "Max Drawdown",     f"{mdd*100:.1f}%",    "Peak-to-Trough",      GW_RED),
        ]:
            with col:
                st.markdown(f"""
                <div class="gw-stat-card">
                    <div class="gw-stat-label">{label}</div>
                    <div class="gw-stat-value">{val}</div>
                    <div class="gw-stat-sub" style="color:{sub_color};">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=w_series.index.tolist(),
                values=w_series.values.tolist(),
                texttemplate="%{label}<br><b>%{value:.1f}%</b>",
                textfont=dict(family="Inter", size=10),
                marker=dict(colors=PALETTE[:len(w_series)],
                            line=dict(color="#FFFFFF", width=2)),
                hole=0.45,
            ))
            fig_pie.update_layout(
                annotations=[dict(text=f"<b>{opt_strategy[:8]}</b>", showarrow=False,
                                  font=dict(family="Inter", size=11, color=GW_NAVY))],
            )
            groww_fig(fig_pie, 380, f"Allocation — {opt_strategy}")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_wbar = go.Figure(go.Bar(
                x=w_series.values.tolist(),
                y=w_series.index.tolist(),
                orientation="h",
                marker=dict(
                    color=w_series.values.tolist(),
                    colorscale=[[0, "#E0E7FF"], [1, GW_PURPLE]],
                    line_width=0,
                ),
                text=[f"{v:.1f}%" for v in w_series.values],
                textposition="outside",
                textfont=dict(family="Inter", size=10),
            ))
            groww_fig(fig_wbar, 380, "Weight Allocation")
            fig_wbar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_wbar, use_container_width=True)

    if summary is not None and not summary.empty:
        st.markdown('<div class="gw-section-title">Strategy Comparison</div>', unsafe_allow_html=True)
        st.dataframe(
            summary.style.format("{:.4f}")
                .highlight_max(subset=[c for c in ["Ann. Return (%)", "Sharpe Ratio"] if c in summary.columns], color="#D1FAE5")
                .highlight_min(subset=[c for c in ["Ann. Volatility (%)"] if c in summary.columns], color="#D1FAE5"),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.markdown('<div class="gw-section-title">Risk Scorecard</div>', unsafe_allow_html=True)
    try:
        try:
            _bench = fetch_benchmark(period=lookback)
            _bench_ret = _bench.pct_change().dropna().iloc[:, 0] if _bench is not None and not _bench.empty else returns.mean(axis=1)
        except Exception:
            _bench_ret = returns.mean(axis=1)
        _bench_ret = _bench_ret.reindex(returns.index).ffill().bfill()
        sc = risk_scorecard(returns, _bench_ret)

        if sc is not None and not sc.empty:
            # Score cards grid
            cols_sc = st.columns(min(3, len(sc)))
            for i, (_, row) in enumerate(sc.iterrows()):
                score = row.get("Risk Score", 50)
                level = row.get("Risk Level", "Moderate")
                color = _risk_color(level)
                with cols_sc[i % 3]:
                    st.markdown(f"""
                    <div class="gw-card" style="text-align:center;padding:20px;">
                        <div style="font-size:0.8rem;font-weight:600;color:#1B2236;margin-bottom:8px;">{row.name}</div>
                        <div style="font-size:2rem;font-weight:700;color:{color};">{score:.0f}</div>
                        <div style="font-size:0.7rem;color:#9CA3AF;margin-bottom:10px;">/ 100</div>
                        <span style="background:{'rgba(0,208,156,0.1)' if level=='Low' else 'rgba(255,83,112,0.1)' if level in ('High','Very High') else 'rgba(245,158,11,0.1)'};
                               color:{color};border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;">{level}</span>
                        <div class="gw-score-bar-wrap" style="margin-top:12px;">
                            <div class="gw-score-bar" style="width:{score}%;background:{color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Radar chart ──
            cols_radar    = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Hist VaR 95% (%)"]
            cols_present  = [c for c in cols_radar if c in sc.columns]
            if cols_present:
                fig_radar = go.Figure()
                def _hex_rgba(h, alpha=0.1):
                    h = h.lstrip("#")
                    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                    return f"rgba({r},{g},{b},{alpha})"
                for i, name in enumerate(sc.index[:6]):
                    vals  = sc.loc[name, cols_present].values.tolist()
                    color = PALETTE[i % len(PALETTE)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=cols_present + [cols_present[0]],
                        name=name, line=dict(color=color, width=2),
                        fill="toself", fillcolor=_hex_rgba(color),
                    ))
                groww_fig(fig_radar, 440, "Risk Radar")
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#FAFAFA",
                        radialaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB",
                                        tickfont=dict(family="Inter", size=8, color="#9CA3AF")),
                        angularaxis=dict(gridcolor="#E5E7EB",
                                         tickfont=dict(family="Inter", size=10, color="#374151")),
                    )
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            num_cols = sc.select_dtypes(include="number").columns.tolist()
            st.dataframe(sc.style.format({c: "{:.4f}" for c in num_cols}, na_rep="N/A"), use_container_width=True)
    except Exception as e:
        st.error(f"Scorecard error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="gw-section-title">Price Prediction Engine</div>', unsafe_allow_html=True)

    model_info = {
        "ARIMA":             "Auto-regressive Integrated Moving Average — classical time-series model",
        "Linear Regression": "Ridge regression with lag + RSI/MACD/BB technical signals",
        "Random Forest":     "Ensemble of decision trees with walk-forward validation",
        "Monte Carlo":       "Geometric Brownian Motion simulation (10,000 paths)",
        "EMA Trend":         "Exponential moving average trend extrapolation",
    }
    st.markdown(f"""
    <div class="gw-info-box">
        <b>Model:</b> {pred_model} &nbsp;·&nbsp;
        <b>Stock:</b> {pred_company} &nbsp;·&nbsp;
        <b>Horizon:</b> {pred_horizon} trading days<br>
        <span style="color:#9CA3AF;font-size:0.75rem;">{model_info.get(pred_model,'')}</span>
    </div>
    """, unsafe_allow_html=True)

    if pred_btn or (st.session_state.pred_result and
                   st.session_state.pred_result.get("company") == pred_company and
                   st.session_state.pred_result.get("model") == pred_model):

        if pred_btn:
            with st.spinner(f"Running {pred_model} on {pred_company}..."):
                try:
                    if pred_company in prices.columns:
                        price_series = prices[pred_company].dropna()
                    else:
                        pred_ticker = universe.get(pred_company, pred_company)
                        _raw = fetch_price_data([pred_ticker], period=lookback)
                        if _raw.empty:
                            raise ValueError(f"No data for {pred_company}")
                        price_series = _raw.iloc[:, 0].dropna()
                        price_series.name = pred_company
                    if len(price_series) < 30:
                        raise ValueError(f"Not enough data ({len(price_series)} points)")
                    result = run_prediction(pred_model, price_series, horizon=pred_horizon)
                    st.session_state.pred_result = {
                        "company": pred_company, "model": pred_model,
                        "result": result, "series": price_series,
                    }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.session_state.pred_result = None

        pr = st.session_state.pred_result
        if pr and pr.get("result") is not None:
            result       = pr["result"]
            price_series = pr["series"]

            if result.error:
                st.error(f"Model error: {result.error}")
            else:
                forecast = result.forecast
                upper    = result.upper_bound
                lower    = result.lower_bound

                fig_pred = go.Figure()
                hist     = price_series.tail(180)
                hist_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in hist.index]

                fig_pred.add_trace(go.Scatter(
                    x=hist_idx, y=hist.values, mode="lines", name="Historical",
                    line=dict(color=GW_NAVY, width=2),
                ))
                if forecast is not None and not forecast.empty:
                    fore_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in forecast.index]
                    if upper is not None and lower is not None:
                        up_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in upper.index]
                        lo_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in lower.index]
                        fig_pred.add_trace(go.Scatter(
                            x=up_idx, y=upper.values, mode="lines",
                            name="Upper Band", line=dict(color="rgba(83,103,255,0.3)", width=1),
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=lo_idx, y=lower.values, mode="lines",
                            name="Lower Band", line=dict(color="rgba(83,103,255,0.3)", width=1),
                            fill="tonexty", fillcolor="rgba(83,103,255,0.08)",
                        ))
                    fig_pred.add_trace(go.Scatter(
                        x=fore_idx, y=forecast.values, mode="lines",
                        name=f"{pred_model} Forecast",
                        line=dict(color=GW_PURPLE, width=2.5, dash="dash"),
                    ))
                _vline(fig_pred, price_series.index[-1])
                groww_fig(fig_pred, 460, f"{pred_company} — {pred_model} Forecast ({pred_horizon}d)")
                st.plotly_chart(fig_pred, use_container_width=True)

                # ── Key metrics ──
                metrics    = result.metrics or {}
                last_price = float(price_series.iloc[-1])
                last_fore  = float(forecast.iloc[-1]) if forecast is not None and not forecast.empty else last_price
                pred_chg   = (last_fore - last_price) / last_price * 100

                m_cols = st.columns(4)
                with m_cols[0]:
                    st.markdown(f"""
                    <div class="gw-stat-card">
                        <div class="gw-stat-label">Current Price</div>
                        <div class="gw-stat-value">₹{last_price:,.2f}</div>
                    </div>""", unsafe_allow_html=True)
                with m_cols[1]:
                    color = GW_GREEN if pred_chg > 0 else GW_RED
                    st.markdown(f"""
                    <div class="gw-stat-card">
                        <div class="gw-stat-label">In {pred_horizon} Days</div>
                        <div class="gw-stat-value">₹{last_fore:,.2f}</div>
                        <div class="gw-stat-sub" style="color:{color};">{pred_chg:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                if "rmse" in metrics:
                    with m_cols[2]:
                        st.markdown(f"""
                        <div class="gw-stat-card">
                            <div class="gw-stat-label">RMSE</div>
                            <div class="gw-stat-value">₹{metrics['rmse']:.2f}</div>
                        </div>""", unsafe_allow_html=True)
                if "mape" in metrics:
                    with m_cols[3]:
                        st.markdown(f"""
                        <div class="gw-stat-card">
                            <div class="gw-stat-label">MAPE</div>
                            <div class="gw-stat-value">{metrics['mape']:.2f}%</div>
                        </div>""", unsafe_allow_html=True)

                # ── Scenario table ──
                if forecast is not None and not forecast.empty:
                    st.markdown('<div class="gw-section-title" style="margin-top:20px;">Scenario Analysis</div>', unsafe_allow_html=True)
                    cur  = float(price_series.iloc[-1])
                    f1w  = float(forecast.iloc[min(4,  len(forecast)-1)])
                    f1m  = float(forecast.iloc[min(19, len(forecast)-1)])
                    fend = float(forecast.iloc[-1])
                    sc_df = pd.DataFrame({
                        "Horizon":  ["1 Week", "1 Month", f"{pred_horizon} Days"],
                        "Forecast": [f"₹{f1w:,.2f}",  f"₹{f1m:,.2f}",  f"₹{fend:,.2f}"],
                        "Change":   [f"{(f1w-cur)/cur*100:+.2f}%",  f"{(f1m-cur)/cur*100:+.2f}%",
                                     f"{(fend-cur)/cur*100:+.2f}%"],
                        "Signal":   [("🟢 BUY" if f1w>cur else "🔴 SELL"),
                                     ("🟢 BUY" if f1m>cur else "🔴 SELL"),
                                     ("🟢 BUY" if fend>cur else "🔴 SELL")],
                    })
                    st.dataframe(sc_df.set_index("Horizon"), use_container_width=True)
    else:
        st.markdown("""
        <div class="gw-card" style="text-align:center;padding:32px;">
            <div style="font-size:2rem;margin-bottom:8px;">🔮</div>
            <div style="font-weight:600;color:#1B2236;">Configure and run a prediction</div>
            <div style="color:#9CA3AF;font-size:0.82rem;margin-top:6px;">
                Select a stock, model and horizon in the sidebar, then click <b>Run Prediction</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px;color:#9CA3AF;font-size:0.72rem;">
    Portfolio Optimizer &nbsp;·&nbsp; {exchange} &nbsp;·&nbsp;
    Data via Yahoo Finance (15-min delay) &nbsp;·&nbsp;
    {datetime.now().strftime("%d %b %Y")}
</div>
""", unsafe_allow_html=True)
