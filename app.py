"""
Groww-Style NSE/BSE Portfolio Optimizer & Risk Dashboard
=========================================================
• Clean, modern UI inspired by Groww
• White cards, purple/teal accents, Inter font
• Live price data, risk analysis, portfolio optimisation
• Stock Movement Predictor — next-day up/down direction classifiers
  (Logistic Regression · Random Forest · XGBoost) backtested vs baselines

Run:  streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

from data_fetcher import (
    fetch_price_data, fetch_ohlcv,
    fetch_all_live_quotes, fetch_benchmark, compute_returns, RISK_FREE_RATE,
)
from risk_engine import (
    annualised_return, annualised_volatility, sharpe_ratio,
    drawdown_series, var_summary,
    correlation_matrix, rolling_volatility,
    portfolio_returns, TRADING_DAYS,
)
from optimizer import run_strategy, all_strategies_summary, STRATEGIES
from predictor import run_all_predictions, compute_technical_indicators, MODEL_ORDER
from movement_predictor import (
    run_movement_analysis, monte_carlo_paths, MOVEMENT_MODELS,
)
from price_forecast import forecast_price
from fundamentals import NIFTY_50, fundamental_context
from insights_tab import render_insights_tab
from terminal_tab import render_terminal_tab

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

    # ── Whole-app universe: Nifty 50 only (NSE) ──────────────────────────────
    exchange  = "NSE"
    universe  = dict(NIFTY_50)
    all_names = sorted(universe.keys())

    PRESETS = {
        "── Select Preset ──": [],
        "💻 IT":       [n for n in all_names if any(k in n for k in ["TCS","Infosys","Wipro","HCL","Tech Mahindra"])],
        "🏦 Banking":  [n for n in all_names if any(k in n for k in ["HDFC","ICICI","Kotak","State Bank","Axis","IndusInd"])],
        "🚗 Auto":     [n for n in all_names if any(k in n for k in ["Maruti","Bajaj Auto","Hero","Tata Motors","Mahindra","Eicher"])],
        "💊 Pharma":   [n for n in all_names if any(k in n for k in ["Sun Pharma","Dr Reddy","Cipla","Apollo"])],
        "⚡ Energy":   [n for n in all_names if any(k in n for k in ["Reliance","ONGC","NTPC","Coal India","BPCL","Power Grid"])],
        "🛒 FMCG":     [n for n in all_names if any(k in n for k in ["Hindustan Unilever","ITC","Nestle","Tata Consumer","Titan"])],
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
    st.caption("🎯 The Movement Predictor is self-contained — open the **Predict** tab.")
    run_btn      = st.button("▶  Run Analysis",       use_container_width=True)

    st.markdown(
        f'<div style="font-size:0.68rem;color:#4B5563;margin-top:12px;text-align:center;">'
        f'{datetime.now().strftime("%d %b %Y, %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )


# ── Active tab tracking (controls autorefresh scope) ─────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Live Feed"

# Auto-refresh ONLY when on the Live Feed tab
if auto_refresh and st.session_state.active_tab == "Live Feed":
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
            <span class="gw-pill-purple">✓ Movement Predictor</span>
        </div>
        <div style="color:#6B7280;font-size:0.78rem;margin-top:20px;">
            130+ NSE & BSE stocks · 5 optimization strategies · up/down direction classifiers
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("data_loaded", False), ("prices", None), ("quotes", {}),
             ("mv_result", None), ("compare_results", None), ("last_selection", [])]:
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

# ── Custom tab bar ────────────────────────────────────────────────────────────
TAB_NAMES = ["📊 Live Feed", "🖥️ Terminal", "📉 Charts", "⚠️ Risk", "⚙️ Optimizer", "🔮 Predict", "💡 Insights"]

# Inject CSS to style active/inactive tab buttons
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] button {
    width: 100% !important;
    border-radius: 8px !important;
    font-family: Inter, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 8px 4px !important;
    border: 1px solid #E5E7EB !important;
    background: #ffffff !important;
    color: #6B7280 !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] button:hover {
    background: #F3F4F6 !important;
    color: #1B2236 !important;
}
</style>
""", unsafe_allow_html=True)

tab_cols = st.columns(len(TAB_NAMES))
for i, tname in enumerate(TAB_NAMES):
    short = tname.split(" ", 1)[1]
    with tab_cols[i]:
        if st.button(tname, key=f"tab_btn_{i}", use_container_width=True):
            st.session_state.active_tab = short
            st.rerun()

active = st.session_state.active_tab

# Highlight the active tab button with CSS override using its index
active_idx = next((i for i, t in enumerate(TAB_NAMES) if t.split(" ", 1)[1] == active), 0)
st.markdown(f"""
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child({active_idx + 1}) button {{
    background: linear-gradient(135deg, #5367FF, #8B5CF6) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(83,103,255,0.35) !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="border-top:2px solid #E5E7EB;margin:8px 0 16px;"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
if active == "Live Feed":
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
if active == "Charts":
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
if active == "Risk":
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
if active == "Optimizer":
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
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if active == "Predict":
    # Self-contained tab — hide the sidebar (controls live inline below).
    st.markdown("""
    <style>
    [data-testid="stSidebar"]                 { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="collapsedControl"]          { display: none !important; }
    .block-container { max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gw-section-title">🎯 Stock Movement Predictor '
                '<span style="font-size:0.72rem;color:#9CA3AF;font-weight:400;">· Nifty 50</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#6B7280;font-size:0.86rem;margin:-6px 0 14px;">'
        'Next-day <b>direction</b> (up / down) from price, <b>earnings</b> &amp; <b>market-regime</b> features — '
        'walk-forward backtested — then adjusted by a <b>live context</b> read of news, fundamentals &amp; analysts.</div>',
        unsafe_allow_html=True)

    MV_COLORS = {
        "Ensemble":            "#5367FF",
        "LightGBM":            "#00D09C",
        "XGBoost":             "#F59E0B",
        "Logistic Regression": "#8B5CF6",
    }

    # ── Inline controls (Nifty-50 focused) ────────────────────────────────────
    _nifty_names = list(NIFTY_50.keys())
    c_stock, c_hist, c_test, c_run = st.columns([2.4, 1, 1.2, 1.3])
    with c_stock:
        pred_company = st.selectbox("Stock (Nifty 50)", _nifty_names,
                                    index=_nifty_names.index("Reliance Industries"))
    with c_hist:
        mv_history = st.selectbox("History", ["2y", "3y", "5y", "10y"], index=2)
    with c_test:
        mv_test_pct = st.slider("Backtest %", 10, 40, 25, step=5)
    with c_run:
        st.write("")
        pred_btn = st.button("🎯  Run Predictor", use_container_width=True, type="primary")
    with st.expander("⚙️ Classifiers & sources", expanded=False):
        mv_models = st.multiselect("Classifiers", MOVEMENT_MODELS, default=MOVEMENT_MODELS)
        st.caption("Model features (backtested): price/technical · earnings-cycle · market regime (Nifty + India VIX). "
                   "Live context (not backtested): per-stock news, Screener.in fundamentals, analyst view.")

    mv_ticker     = NIFTY_50.get(pred_company, pred_company)
    _mv_test_frac = mv_test_pct / 100.0
    _mv_cache_key = f"{pred_company}|{mv_history}|{sorted(mv_models)}|{mv_test_pct}"
    _mv_cached    = st.session_state.get("mv_result")
    _mv_valid     = _mv_cached and _mv_cached.get("cache_key") == _mv_cache_key

    if pred_btn or not _mv_valid:
        try:
            if not mv_models:
                raise ValueError("Select at least one classifier (⚙️ Classifiers & sources).")
            with st.spinner(f"Pulling {mv_history} of data for {pred_company} & engineering features…"):
                _ohlcv = fetch_ohlcv(mv_ticker, period=mv_history)
            if _ohlcv is None or _ohlcv.empty or "Close" not in _ohlcv.columns:
                raise ValueError(f"No OHLCV data for {pred_company}.")

            with st.spinner("Training on price + earnings + market-regime features & backtesting…"):
                mv_out = run_movement_analysis(
                    _ohlcv, models=mv_models, test_frac=_mv_test_frac, ticker=mv_ticker)
            with st.spinner("Reading live context — news, Screener.in fundamentals, analysts…"):
                mv_ctx = fundamental_context(mv_ticker)
            with st.spinner("Projecting long-term price (1–4 months)…"):
                _target = (mv_ctx.get("analyst") or {}).get("target_mean")
                mv_fc = forecast_price(_ohlcv["Close"], analyst_target=_target)

            st.session_state.mv_result = {
                "cache_key": _mv_cache_key,
                "company":   pred_company,
                "out":       mv_out,
                "ctx":       mv_ctx,
                "fc":        mv_fc,
                "colors":    MV_COLORS,
            }
        except Exception as e:
            st.error(f"Movement predictor error: {e}")
            st.session_state.mv_result = None

    mvr = st.session_state.get("mv_result")
    if mvr and mvr.get("out"):
        out        = mvr["out"]
        results    = out["results"]
        best_name  = out["best_name"]
        base       = out["baselines"]
        MV_COLORS  = mvr["colors"]
        close      = out["close"]
        best       = results[best_name]
        ctx        = mvr.get("ctx") or {}

        # ── Tomorrow's headline call (from the best classifier) ───────────────
        _sig       = best.next_signal
        _prob      = best.next_up_prob
        _sig_prob  = _prob if _sig == "UP" else (1 - _prob)
        _sig_color = GW_GREEN if _sig == "UP" else GW_RED
        _last_dt   = close.index[-1]
        _last_str  = _last_dt.strftime("%d %b %Y") if hasattr(_last_dt, "strftime") else str(_last_dt)

        # ── Context blend: nudge the backtested prob by the live-overlay bias ──
        _bias      = float(ctx.get("bias", 0.0))          # −1..+1
        _blend_prob = float(min(max(_prob + 0.10 * _bias, 0.0), 1.0))
        _comb_sig  = "UP" if _blend_prob >= 0.5 else "DOWN"
        _comb_prob = _blend_prob if _comb_sig == "UP" else (1 - _blend_prob)
        _comb_color = GW_GREEN if _comb_sig == "UP" else GW_RED
        _ctx_label = ctx.get("label", "Neutral")
        _ctx_color = GW_GREEN if _bias > 0.15 else GW_RED if _bias < -0.15 else "#F59E0B"

        fc = mvr.get("fc")
        _acc      = best.accuracy * 100
        _acc_col  = GW_GREEN if _acc >= 55 else "#F59E0B" if _acc >= 50 else GW_RED
        _base_acc = base["majority_acc"]

        st.markdown(f'<div style="font-size:0.70rem;color:#9CA3AF;letter-spacing:1px;margin-bottom:8px;">'
                    f'{mvr["company"]} &nbsp;·&nbsp; last close <b>₹{best.last_close:,.2f}</b> on {_last_str}'
                    f' &nbsp;·&nbsp; model: {best_name} &nbsp;·&nbsp; {out.get("n_features",0)} features</div>',
                    unsafe_allow_html=True)

        # ── THREE HEADLINE OUTPUTS: Movement · Long-term price · Accuracy ─────
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(f"""
            <div class="gw-card" style="padding:16px 18px;border:1.5px solid {_comb_color}55;
                 background:linear-gradient(135deg,{_comb_color}12,#5367FF08);height:100%;">
                <div style="font-size:0.68rem;color:#9CA3AF;letter-spacing:1px;">① MOVEMENT · short-term</div>
                <div style="font-size:1.9rem;font-weight:800;color:{_comb_color};margin:4px 0;">
                    {'▲' if _comb_sig=='UP' else '▼'} {_comb_sig}</div>
                <div style="font-size:0.82rem;color:#6B7280;">conviction
                    <b style="color:{_comb_color};">{_comb_prob*100:.0f}%</b></div>
                <div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">
                    model {_sig} ({_sig_prob*100:.0f}%) + context {_ctx_label} ({_bias:+.2f})</div>
            </div>""", unsafe_allow_html=True)
        with h2:
            if fc is not None:
                _mL   = fc.months[-1]
                _up   = fc.upside[_mL]
                _upc  = GW_GREEN if _up > 0 else GW_RED
                st.markdown(f"""
                <div class="gw-card" style="padding:16px 18px;border:1.5px solid #5367FF33;height:100%;">
                    <div style="font-size:0.68rem;color:#9CA3AF;letter-spacing:1px;">② FUTURE PRICE · {_mL} months</div>
                    <div style="font-size:1.9rem;font-weight:800;color:{GW_NAVY};margin:4px 0;">
                        ₹{fc.median[_mL]:,.0f}</div>
                    <div style="font-size:0.82rem;color:#6B7280;">median ·
                        <b style="color:{_upc};">{_up:+.1f}%</b></div>
                    <div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">
                        50% range ₹{fc.p25[_mL]:,.0f}–{fc.p75[_mL]:,.0f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="gw-card" style="padding:16px 18px;height:100%;">Forecast unavailable</div>', unsafe_allow_html=True)
        with h3:
            st.markdown(f"""
            <div class="gw-card" style="padding:16px 18px;border:1.5px solid {_acc_col}44;height:100%;">
                <div style="font-size:0.68rem;color:#9CA3AF;letter-spacing:1px;">③ MODEL ACCURACY · backtested</div>
                <div style="font-size:1.9rem;font-weight:800;color:{_acc_col};margin:4px 0;">
                    {_acc:.1f}%</div>
                <div style="font-size:0.82rem;color:#6B7280;">vs {_base_acc:.0f}% baseline ·
                    <b style="color:{GW_GREEN if _acc>_base_acc else GW_RED};">{_acc-_base_acc:+.1f} pts</b></div>
                <div style="font-size:0.72rem;color:#9CA3AF;margin-top:4px;">
                    {out['n_test']} unseen days · direction hit-rate</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div style="font-size:0.70rem;color:#9CA3AF;margin:10px 0 4px;">'
                    f'① is walk-forward backtested; the context nudge (news/fundamentals/analysts) is a live read, '
                    f'<b>not backtested</b>. ② is a Monte-Carlo range anchored to analyst targets — a distribution, not a promise.</div>',
                    unsafe_allow_html=True)

        # ── LONG-TERM PRICE FORECAST — chart + table ─────────────────────────
        if fc is not None:
            st.markdown('<div class="gw-section-title" style="margin-top:14px;font-size:0.9rem;">🔮 Long-Term Price Forecast '
                        '<span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">Monte-Carlo · drift dampened & anchored to analyst target</span></div>',
                        unsafe_allow_html=True)
            _hist = close.tail(120)
            _hi   = [str(i.date()) if hasattr(i, "date") else str(i) for i in _hist.index]
            _fdates = pd.bdate_range(start=close.index[-1], periods=len(fc.path_median))
            _fi   = [str(d.date()) for d in _fdates]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=_hi, y=_hist.values, mode="lines",
                             name="Historical", line=dict(color=GW_NAVY, width=2)))
            fig_fc.add_trace(go.Scatter(x=_fi, y=fc.path_p95, mode="lines",
                             name="95th pct", line=dict(color="rgba(83,103,255,0.15)", width=1)))
            fig_fc.add_trace(go.Scatter(x=_fi, y=fc.path_p5, mode="lines", name="5th–95th",
                             line=dict(color="rgba(83,103,255,0.15)", width=1),
                             fill="tonexty", fillcolor="rgba(83,103,255,0.08)"))
            fig_fc.add_trace(go.Scatter(x=_fi, y=fc.path_p75, mode="lines", name="75th pct",
                             line=dict(color="rgba(0,208,156,0.20)", width=1), showlegend=False))
            fig_fc.add_trace(go.Scatter(x=_fi, y=fc.path_p25, mode="lines", name="25th–75th",
                             line=dict(color="rgba(0,208,156,0.20)", width=1),
                             fill="tonexty", fillcolor="rgba(0,208,156,0.12)"))
            fig_fc.add_trace(go.Scatter(x=_fi, y=fc.path_median, mode="lines", name="Median",
                             line=dict(color=GW_PURPLE, width=3)))
            if fc.analyst_target:
                fig_fc.add_trace(go.Scatter(x=[_fi[-1]], y=[fc.analyst_target], mode="markers",
                                 name="Analyst target", marker=dict(color="#F59E0B", size=10, symbol="star")))
            _vline(fig_fc, close.index[-1])
            groww_fig(fig_fc, 400, f"{mvr['company']} — {fc.months[-1]}-month projection "
                      f"(drift {fc.drift_annual*100:+.0f}%/yr · vol {fc.vol_annual*100:.0f}%/yr)")
            st.plotly_chart(fig_fc, use_container_width=True)

            _fc_rows = []
            for m in fc.months:
                _fc_rows.append({
                    "Horizon":      f"{m} month{'s' if m > 1 else ''}",
                    "Expected (median)": f"₹{fc.median[m]:,.0f}",
                    "Upside":       f"{fc.upside[m]:+.1f}%",
                    "50% range":    f"₹{fc.p25[m]:,.0f} – {fc.p75[m]:,.0f}",
                    "90% range":    f"₹{fc.p5[m]:,.0f} – {fc.p95[m]:,.0f}",
                })
            st.dataframe(pd.DataFrame(_fc_rows).set_index("Horizon"), use_container_width=True)
            if fc.analyst_target:
                st.caption(f"⭐ Analyst mean target (≈12-mo): ₹{fc.analyst_target:,.0f} "
                           f"({(fc.analyst_target/fc.last_close-1)*100:+.1f}% vs last close) — used to anchor the drift.")

        # ── Evaluation metric cards (best model) ──────────────────────────────
        _acc_color = GW_GREEN if best.accuracy >= 0.55 else ("#F59E0B" if best.accuracy >= 0.5 else GW_RED)
        _edge      = best.accuracy * 100 - base["majority_acc"]
        _edge_col  = GW_GREEN if _edge > 0 else GW_RED
        m_cols = st.columns(4)
        for _c, (_lbl, _val, _sub, _col) in zip(m_cols, [
            ("Accuracy",  f"{best.accuracy*100:.1f}%",  f"vs {base['majority_acc']:.0f}% majority", _acc_color),
            ("Precision", f"{best.precision*100:.1f}%", "of 'up' calls correct", GW_NAVY),
            ("Recall",    f"{best.recall*100:.1f}%",    "of up-days caught", GW_NAVY),
            ("F1-Score",  f"{best.f1*100:.1f}%",        f"edge {_edge:+.1f} pts", _edge_col),
        ]):
            with _c:
                st.markdown(f"""
                <div class="gw-stat-card">
                    <div class="gw-stat-label">{_lbl}</div>
                    <div class="gw-stat-value" style="color:{_col};">{_val}</div>
                    <div class="gw-stat-sub" style="color:#9CA3AF;">{_sub}</div>
                </div>""", unsafe_allow_html=True)

        # ── Live context panel (news · fundamentals · analysts · earnings) ────
        st.markdown('<div class="gw-section-title" style="margin-top:22px;">🌐 Live Context '
                    '<span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">news · Screener.in fundamentals · analysts · earnings — a real-time read, not backtested</span></div>',
                    unsafe_allow_html=True)
        _news = ctx.get("news", {}) or {}
        _scr  = ctx.get("screener", {}) or {}
        _anal = ctx.get("analyst", {}) or {}
        _earn = ctx.get("earnings", {}) or {}
        _contrib = ctx.get("contrib", {}) or {}

        cc1, cc2, cc3 = st.columns(3)
        # 1) Context bias breakdown
        with cc1:
            _rows = "".join(
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
                f'padding:2px 0;"><span style="color:#6B7280;">{k}</span>'
                f'<b style="color:{GW_GREEN if v>0.05 else GW_RED if v<-0.05 else "#9CA3AF"};">{v:+.2f}</b></div>'
                for k, v in sorted(_contrib.items(), key=lambda x: -abs(x[1]))
            ) or '<div style="color:#9CA3AF;font-size:0.8rem;">No context signals available.</div>'
            st.markdown(f"""
            <div class="gw-card" style="padding:14px 16px;height:100%;">
                <div style="font-size:0.72rem;color:#9CA3AF;letter-spacing:0.05em;margin-bottom:8px;">
                    CONTEXT BIAS &nbsp;·&nbsp; <b style="color:{_ctx_color};">{_ctx_label} ({_bias:+.2f})</b></div>
                {_rows}
            </div>""", unsafe_allow_html=True)
        # 2) Fundamentals (Screener) + analyst
        with cc2:
            def _fmt(v, suf=""):
                return f"{v:g}{suf}" if isinstance(v, (int, float)) else "—"
            _reco = _anal.get("recommendation") or "—"
            _up   = _anal.get("target_upside")
            _up_s = (f'<span style="color:{GW_GREEN if _up>0 else GW_RED};">{_up:+.1f}%</span>'
                     if _up is not None else "—")
            st.markdown(f"""
            <div class="gw-card" style="padding:14px 16px;height:100%;">
                <div style="font-size:0.72rem;color:#9CA3AF;letter-spacing:0.05em;margin-bottom:8px;">FUNDAMENTALS · ANALYSTS</div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Stock P/E</span><b>{_fmt(_scr.get('pe'))}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">ROCE / ROE</span><b>{_fmt(_scr.get('roce'),'%')} / {_fmt(_scr.get('roe'),'%')}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Div. yield</span><b>{_fmt(_scr.get('div_yield'),'%')}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Analyst view</span><b>{_reco}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Target upside</span><b>{_up_s}</b></div>
            </div>""", unsafe_allow_html=True)
        # 3) Earnings cycle
        with cc3:
            _dt  = _earn.get("days_to")
            _ds  = _earn.get("days_since")
            _ls  = _earn.get("last_surprise")
            _nd  = _earn.get("next_date")
            _nd_s = _nd.strftime("%d %b %Y") if hasattr(_nd, "strftime") else "—"
            _ls_s = (f'<span style="color:{GW_GREEN if _ls>0 else GW_RED};">{_ls:+.1f}%</span>'
                     if _ls is not None else "—")
            _warn = ('<div style="color:#F59E0B;font-size:0.74rem;margin-top:6px;">⚠️ Earnings within 5 days — expect elevated volatility.</div>'
                     if isinstance(_dt, int) and _dt <= 5 else "")
            st.markdown(f"""
            <div class="gw-card" style="padding:14px 16px;height:100%;">
                <div style="font-size:0.72rem;color:#9CA3AF;letter-spacing:0.05em;margin-bottom:8px;">EARNINGS CYCLE</div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Next report</span><b>{_nd_s}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Days to / since</span><b>{_dt if _dt is not None else '—'} / {_ds if _ds is not None else '—'}</b></div>
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span style="color:#6B7280;">Last EPS surprise</span><b>{_ls_s}</b></div>
                {_warn}
            </div>""", unsafe_allow_html=True)

        # News headlines
        if _news.get("items"):
            with st.expander(f"📰 Recent news for {mvr['company']} — sentiment {_news.get('label','')} "
                             f"({_news.get('score',0):+.2f})", expanded=False):
                for it in _news["items"][:8]:
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #F3F4F6;">'
                        f'<span style="background:{it["color"]}22;color:{it["color"]};font-size:0.66rem;'
                        f'font-weight:700;padding:2px 7px;border-radius:6px;">{it["polarity"]:+.2f}</span>'
                        f'<span style="font-size:0.85rem;color:#1B2236;margin-left:8px;">{it["title"]}</span>'
                        f'<span style="font-size:0.72rem;color:#9CA3AF;margin-left:8px;">· {it["source"]} {it["time"]}</span>'
                        f'</div>', unsafe_allow_html=True)

        # ── Backtest equity curve: strategy vs buy & hold ─────────────────────
        st.markdown('<div class="gw-section-title" style="margin-top:20px;font-size:0.86rem;">📈 Strategy Backtest — Signal vs Baselines <span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">(₹1 grown over the held-out test window)</span></div>', unsafe_allow_html=True)
        fig_eq = go.Figure()
        bh_eq  = base["buyhold_equity"]
        bh_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in bh_eq.index]
        fig_eq.add_trace(go.Scatter(
            x=bh_idx, y=bh_eq.values, mode="lines", name="Buy & Hold",
            line=dict(color=GW_NAVY, width=2, dash="dot")))
        for mname, res in results.items():
            if res.error or res.equity.empty:
                continue
            eq_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in res.equity.index]
            fig_eq.add_trace(go.Scatter(
                x=eq_idx, y=res.equity.values, mode="lines",
                name=f"{'🏆 ' if mname == best_name else ''}{mname}",
                line=dict(color=MV_COLORS.get(mname, "#9CA3AF"),
                          width=3 if mname == best_name else 1.5)))
        groww_fig(fig_eq, 380, f"{mvr['company']} — strategy equity vs buy & hold")
        st.plotly_chart(fig_eq, use_container_width=True)

        _bh_col = GW_GREEN if base["buyhold_return"] >= 0 else GW_RED
        _st_col = GW_GREEN if best.strat_return >= base["buyhold_return"] else GW_RED
        st.markdown(
            f'<div style="color:#6B7280;font-size:0.82rem;margin:-4px 0 10px;">'
            f'Best strategy return <b style="color:{_st_col};">{best.strat_return*100:+.1f}%</b> '
            f'&nbsp;·&nbsp; buy &amp; hold <b style="color:{_bh_col};">{base["buyhold_return"]*100:+.1f}%</b> '
            f'&nbsp;·&nbsp; win rate <b>{best.win_rate:.0f}%</b> '
            f'&nbsp;·&nbsp; strategy Sharpe <b>{best.strat_sharpe:.2f}</b> '
            f'&nbsp;·&nbsp; random-guess accuracy ≈ <b>{base["random_acc"]:.0f}%</b></div>',
            unsafe_allow_html=True)

        # ── Model leaderboard ─────────────────────────────────────────────────
        st.markdown('<div class="gw-section-title" style="margin-top:16px;">🏆 Classifier Leaderboard &nbsp;<span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">ranked by F1 · 🏆 = best</span></div>', unsafe_allow_html=True)
        lb_rows = []
        for mname, res in results.items():
            if res.error:
                lb_rows.append({"_f1": -1, "Model": f"⚠️ {mname}", "Accuracy": "—",
                                "Precision": "—", "Recall": "—", "F1": "—",
                                "Next Day": "—", "Pred. Close": "—",
                                "Strategy Return": res.error[:40]})
                continue
            lb_rows.append({
                "_f1":       res.f1,
                "Model":     ("🏆 " if mname == best_name else "") + mname,
                "Accuracy":  f"{res.accuracy*100:.1f}%",
                "Precision": f"{res.precision*100:.1f}%",
                "Recall":    f"{res.recall*100:.1f}%",
                "F1":        f"{res.f1*100:.1f}%",
                "Next Day":  f"{'▲' if res.next_signal=='UP' else '▼'} {res.next_signal} "
                             f"({(res.next_up_prob if res.next_signal=='UP' else 1-res.next_up_prob)*100:.0f}%)",
                "Pred. Close": f"₹{res.next_price:,.2f} ({res.next_exp_return*100:+.2f}%)",
                "Strategy Return": f"{res.strat_return*100:+.1f}%",
            })
        lb_rows.sort(key=lambda r: r["_f1"], reverse=True)
        for r in lb_rows:
            r.pop("_f1", None)
        st.dataframe(pd.DataFrame(lb_rows).set_index("Model"), use_container_width=True)

        # ── Feature importance + confusion matrix ─────────────────────────────
        fi_col, cm_col = st.columns([1.4, 1])
        with fi_col:
            st.markdown('<div class="gw-section-title" style="font-size:0.82rem;">🔧 What drives the call <span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">(top features · best model)</span></div>', unsafe_allow_html=True)
            fi = best.feat_importance or {}
            if fi:
                items = list(fi.items())[:8][::-1]
                fig_fi = go.Figure(go.Bar(
                    x=[v*100 for _, v in items], y=[k for k, _ in items],
                    orientation="h",
                    marker=dict(color=MV_COLORS.get(best_name, GW_PURPLE))))
                groww_fig(fig_fi, 300, "Relative importance (%)")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.caption("Feature importance unavailable for this model.")
        with cm_col:
            st.markdown('<div class="gw-section-title" style="font-size:0.82rem;">🎛️ Confusion Matrix <span style="font-size:0.7rem;color:#9CA3AF;font-weight:400;">(test window)</span></div>', unsafe_allow_html=True)
            cm = best.confusion
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred Down", "Pred Up"], y=["Actual Down", "Actual Up"],
                text=cm, texttemplate="%{text}", textfont=dict(size=16),
                colorscale=[[0, "#EEF2FF"], [1, MV_COLORS.get(best_name, GW_PURPLE)]],
                showscale=False))
            groww_fig(fig_cm, 300, f"{best_name} · acc {best.accuracy*100:.0f}%")
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Monte Carlo simulation of the stock (hero) ────────────────────────
        with st.expander("🎲 Monte Carlo simulation of this stock (100 paths · GBM)", expanded=False):
            mc = monte_carlo_paths(close, n_paths=100, n_steps=100)
            fig_mc = go.Figure()
            _pal = ["#5367FF", "#00D09C", "#F59E0B", "#8B5CF6", "#06B6D4",
                    "#EC4899", "#10B981", "#F97316", "#EF4444", "#6366F1"]
            for j in range(mc.shape[1]):
                fig_mc.add_trace(go.Scatter(
                    y=mc[:, j], mode="lines", showlegend=False,
                    line=dict(color=_pal[j % len(_pal)], width=0.7), opacity=0.5,
                    hoverinfo="skip"))
            fig_mc.add_trace(go.Scatter(
                y=mc.mean(axis=1), mode="lines", name="Mean path",
                line=dict(color=GW_NAVY, width=2.5)))
            groww_fig(fig_mc, 420,
                      f"{mvr['company']} — 100 simulated paths over 100 trading days "
                      f"(start ₹{float(close.iloc[-1]):,.0f})")
            fig_mc.update_xaxes(title_text="Trading days ahead")
            fig_mc.update_yaxes(title_text="Simulated price")
            st.plotly_chart(fig_mc, use_container_width=True)
            _terminal = mc[-1]
            st.caption(
                f"Across 100 GBM paths calibrated to {mvr['company']}'s own drift & volatility: "
                f"median ₹{np.median(_terminal):,.0f} · "
                f"5th–95th pct ₹{np.percentile(_terminal,5):,.0f}–₹{np.percentile(_terminal,95):,.0f} "
                f"after 100 trading days.")
    else:
        st.markdown("""
        <div class="gw-card" style="text-align:center;padding:32px;">
            <div style="font-size:2rem;margin-bottom:8px;">🎯</div>
            <div style="font-weight:600;color:#1B2236;">Pick a Nifty-50 stock above, then click Run Predictor</div>
            <div style="color:#9CA3AF;font-size:0.82rem;margin-top:6px;">
                Trains on price, earnings-cycle & market-regime features to call tomorrow's direction,
                backtests it against buy&amp;hold and a random guess, then overlays a live read of
                news, Screener.in fundamentals & analyst views.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════
if active == "Terminal":
    # Hide sidebar and expand main content to full width on Terminal tab
    st.markdown("""
    <style>
    [data-testid="stSidebar"]                { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="collapsedControl"]          { display: none !important; }
    .block-container { max-width: 100% !important; padding-left: 1rem !important;
                       padding-right: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)
    render_terminal_tab(universe, exchange)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
if active == "Insights":
    render_insights_tab()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px;color:#9CA3AF;font-size:0.72rem;">
    Portfolio Optimizer &nbsp;·&nbsp; {exchange} &nbsp;·&nbsp;
    Data via Yahoo Finance (15-min delay) &nbsp;·&nbsp;
    {datetime.now().strftime("%d %b %Y")}
</div>
""", unsafe_allow_html=True)
