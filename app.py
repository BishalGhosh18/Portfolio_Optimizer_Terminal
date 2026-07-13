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
from price_forecast import forecast_price
from fundamentals import NIFTY_50, fundamental_context
from cross_sectional import run_universe_model
from insights_tab import render_insights_tab
from terminal_tab import render_terminal_tab

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer | Groww Style",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Zerodha (Kite) style CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══ ZERODHA / KITE THEME — clean light, blue primary ═══
   bg #FFFFFF · panel #F5F6F8 · border #E9ECEF · text #3C3C3C · muted #9AA0A6
   blue #387ED1 · green #4CA64C · red #E64A3B */

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #FFFFFF !important;
    color: #3C3C3C !important;
}
.main .block-container { background-color: transparent !important; padding-top: 1rem; max-width: 100%; }
h1, h2, h3, h4, h5 { color: #3C3C3C; }

/* ── Sidebar (clean light) ── */
[data-testid="stSidebar"] {
    background-color: #FBFBFC !important;
    border-right: 1px solid #E9ECEF !important;
}
[data-testid="stSidebar"] * { color: #3C3C3C !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #1A1A1A !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #FFFFFF !important; border: 1px solid #E9ECEF !important; color: #3C3C3C !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #387ED1 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 4px !important; font-weight: 600 !important; font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important; transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(56,126,209,0.25) !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #2F6FB8 !important; }
[data-testid="stSidebar"] label {
    color: #9AA0A6 !important; font-size: 0.75rem !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: 0.04em !important;
}

/* ── Widgets ── */
.stSelectbox > div > div, .stMultiSelect > div > div,
.stNumberInput > div > div, .stDateInput > div > div, .stTextInput > div > div {
    background-color: #FFFFFF !important; border: 1px solid #E9ECEF !important; color: #3C3C3C !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #387ED1 !important; }

/* ── Main buttons ── */
.stButton > button {
    background: #387ED1 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 4px !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    transition: all 0.15s ease !important; box-shadow: 0 1px 2px rgba(56,126,209,0.25) !important;
}
.stButton > button:hover { background: #2F6FB8 !important; box-shadow: 0 2px 6px rgba(56,126,209,0.35) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] {
    background: #FFFFFF; border: 1px solid #E9ECEF; border-radius: 6px; padding: 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04); margin-bottom: 16px;
}
[data-testid="stTabs"] button {
    background: transparent !important; color: #6B7280 !important; border: none !important;
    border-radius: 4px !important; font-family: 'Inter', sans-serif !important; font-size: 0.8rem !important;
    font-weight: 500 !important; padding: 6px 14px !important; transition: all 0.15s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: #387ED1 !important; color: #FFFFFF !important;
}
[data-testid="stTabs"] button:hover:not([aria-selected="true"]) { background: #F0F3F7 !important; color: #387ED1 !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #3C3C3C !important; font-family: 'Inter', sans-serif !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #9AA0A6 !important; font-family: 'Inter', sans-serif !important; font-size: 0.72rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] { font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; font-weight: 600 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 6px !important; overflow: hidden !important; border: 1px solid #E9ECEF !important; box-shadow: none !important; }

/* ── Divider / Spinner / Progress ── */
hr { border-color: #E9ECEF !important; }
.stSpinner > div { border-color: #387ED1 transparent transparent transparent !important; }
.stProgress > div > div { background: #387ED1 !important; border-radius: 4px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F5F6F8; }
::-webkit-scrollbar-thumb { background: #D5D9DE; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #387ED1; }

/* ── Custom components ── */
.gw-header {
    background: #FFFFFF; border: 1px solid #E9ECEF; border-radius: 8px; padding: 18px 22px;
    margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.gw-header-title { color: #1A1A1A; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
.gw-header-sub  { color: #9AA0A6; font-size: 0.78rem; margin-top: 2px; }
.gw-live-badge {
    background: rgba(76,166,76,0.10); border: 1px solid rgba(76,166,76,0.35); color: #4CA64C;
    font-size: 0.7rem; font-weight: 600; padding: 4px 10px; border-radius: 4px;
    display: inline-flex; align-items: center; gap: 5px;
}
.gw-dot { width: 7px; height: 7px; background: #4CA64C; border-radius: 50%; animation: pulse 1.5s infinite; display: inline-block; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.8); } }

.gw-card {
    background: #FFFFFF; border-radius: 6px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 12px; border: 1px solid #E9ECEF;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.gw-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-color: #DCE0E5; }

.gw-stock-card {
    background: #FFFFFF; border-radius: 6px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #E9ECEF; margin-bottom: 8px; transition: all 0.15s ease;
}
.gw-stock-card:hover { box-shadow: 0 2px 8px rgba(56,126,209,0.12); border-color: rgba(56,126,209,0.35); }
.gw-stock-name { font-size: 0.85rem; font-weight: 600; color: #3C3C3C; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gw-stock-ticker { font-size: 0.7rem; color: #9AA0A6; font-weight: 400; }
.gw-price { font-size: 1.15rem; font-weight: 700; color: #3C3C3C; margin-top: 6px; }
.gw-change-up { font-size: 0.78rem; font-weight: 600; color: #4CA64C; background: rgba(76,166,76,0.10); padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 4px; }
.gw-change-down { font-size: 0.78rem; font-weight: 600; color: #E64A3B; background: rgba(230,74,59,0.10); padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 4px; }
.gw-change-flat { font-size: 0.78rem; font-weight: 600; color: #6B7280; background: #F0F3F7; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 4px; }
.gw-meta { font-size: 0.68rem; color: #9AA0A6; margin-top: 8px; display: grid; grid-template-columns: 1fr 1fr; gap: 2px; }
.gw-meta span { color: #9AA0A6; }
.gw-meta b { color: #4B5563; }

.gw-section-title { font-size: 0.95rem; font-weight: 700; color: #3C3C3C; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.gw-section-title::before { content: ''; width: 3px; height: 18px; background: #387ED1; border-radius: 2px; display: inline-block; }

.gw-ticker-bar {
    background: #FFFFFF; border-radius: 6px; padding: 10px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #E9ECEF; overflow-x: auto; white-space: nowrap; margin-bottom: 16px;
    font-size: 0.78rem; font-weight: 500;
}
.gw-tick-up { color: #4CA64C; }
.gw-tick-down { color: #E64A3B; }
.gw-tick-flat { color: #9AA0A6; }

.gw-stat-card { background: #FFFFFF; border-radius: 6px; padding: 18px 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #E9ECEF; }
.gw-stat-label { font-size: 0.72rem; font-weight: 500; color: #9AA0A6; text-transform: uppercase; letter-spacing: 0.05em; }
.gw-stat-value { font-size: 1.4rem; font-weight: 700; color: #3C3C3C; margin-top: 4px; }
.gw-stat-sub { font-size: 0.75rem; margin-top: 2px; font-weight: 600; }

.gw-pill-green  { background: rgba(76,166,76,0.10); color: #4CA64C; border-radius: 4px; padding: 3px 10px; font-size: 0.72rem; font-weight: 600; }
.gw-pill-red    { background: rgba(230,74,59,0.10); color: #E64A3B; border-radius: 4px; padding: 3px 10px; font-size: 0.72rem; font-weight: 600; }
.gw-pill-purple { background: rgba(56,126,209,0.10); color: #387ED1; border-radius: 4px; padding: 3px 10px; font-size: 0.72rem; font-weight: 600; }

.gw-score-bar-wrap { background: #EEF0F2; border-radius: 20px; height: 8px; margin-top: 6px; overflow: hidden; }
.gw-score-bar { height: 8px; border-radius: 20px; }

.gw-welcome {
    background: #FFFFFF; border: 1px solid #E9ECEF; border-radius: 8px; padding: 40px 32px;
    text-align: center; color: #3C3C3C; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.gw-info-box { background: rgba(56,126,209,0.05); border: 1px solid rgba(56,126,209,0.20); border-radius: 6px; padding: 14px 18px; font-size: 0.8rem; color: #4B5563; margin-bottom: 16px; }
.gw-info-box b { color: #387ED1; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

# ── Zerodha (Kite) palette ──
GW_GREEN  = "#4CA64C"   # gain / up (Kite green)
GW_RED    = "#E64A3B"   # loss / down (Kite red)
GW_PURPLE = "#387ED1"   # primary accent → Zerodha blue
GW_NAVY   = "#3C3C3C"   # primary dark text / chart lines
GW_GRAY   = "#F5F6F8"   # light panel surface

PALETTE = [GW_PURPLE, "#4CA64C", "#F59E0B", GW_RED,
           "#06B6D4", "#8B5CF6", "#EC4899", "#10B981", "#F97316", "#6366F1"]

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
    """Apply the Zerodha (Kite) light theme to a plotly figure."""
    fig.update_layout(
        height=height,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter", size=13, color="#3C3C3C"),
            x=0.01, y=0.99,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBFCFD",
        font=dict(family="Inter", color="#6B7280", size=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E9ECEF",
            borderwidth=1,
            font=dict(family="Inter", color="#3C3C3C", size=10),
        ),
        xaxis=dict(
            gridcolor="#EEF0F2", linecolor="#E9ECEF",
            tickfont=dict(family="Inter", color="#9AA0A6", size=9),
            title_font=dict(family="Inter", color="#6B7280"),
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#EEF0F2", linecolor="#E9ECEF",
            tickfont=dict(family="Inter", color="#9AA0A6", size=9),
            title_font=dict(family="Inter", color="#6B7280"),
            showgrid=True, zeroline=False,
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(
            bgcolor="#3C3C3C",
            font=dict(family="Inter", color="#FFFFFF", size=11),
            bordercolor="#387ED1",
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
            Nifty 50 · 5 optimization strategies · up/down movement predictor + long-term forecast
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
    border: 1px solid #E9ECEF !important;
    background: #FFFFFF !important;
    color: #6B7280 !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] button:hover {
    background: #F0F3F7 !important;
    color: #387ED1 !important;
    border-color: #C9D6E5 !important;
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
    background: #387ED1 !important;
    color: #FFFFFF !important;
    border: 1px solid #387ED1 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(56,126,209,0.3) !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="border-top:1px solid #E9ECEF;margin:8px 0 16px;"></div>', unsafe_allow_html=True)


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
            increasing_fillcolor=f"rgba(76,166,76,0.3)", decreasing_fillcolor=f"rgba(255,83,112,0.3)",
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
                name="BB+", line=dict(width=1, color="rgba(56,126,209,0.3)"), fill=None))
            fig_candle.add_trace(go.Scatter(x=ohlcv.index, y=mid - 2*std, mode="lines",
                name="BB-", line=dict(width=1, color="rgba(56,126,209,0.3)"),
                fill="tonexty", fillcolor="rgba(56,126,209,0.04)"))

        groww_fig(fig_candle, 440, f"{chart_name}")
        fig_candle.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

        fig_vol = go.Figure(go.Bar(
            x=ohlcv.index, y=ohlcv.get("Volume", pd.Series()),
            marker_color=[f"rgba(76,166,76,0.5)" if c >= o else f"rgba(255,83,112,0.5)"
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
                    fig_rsi.add_hrect(y0=0, y1=30,  fillcolor="rgba(76,166,76,0.06)",   line_width=0)
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
                        marker_color=[f"rgba(76,166,76,0.5)" if v >= 0 else f"rgba(255,83,112,0.5)" for v in hist],
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
            fill="tozeroy", fillcolor="rgba(56,126,209,0.06)",
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
    # Self-contained tab — hide the sidebar (controls inline below).
    st.markdown("""
    <style>
    [data-testid="stSidebar"]                 { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="collapsedControl"]          { display: none !important; }
    .block-container { max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gw-section-title">🎯 Trading Signal Model '
                '<span style="font-size:0.72rem;color:#6B7280;font-weight:400;">· Nifty 50 · walk-forward validated · cost-aware</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#6B7280;font-size:0.86rem;margin:-6px 0 14px;">'
        'One <b>pooled cross-sectional model</b> trained on the whole Nifty-50 ranks every stock by calibrated '
        'probability of rising over the horizon. Evaluated by <b>walk-forward CV</b> with an embargo, '
        '<b>net-of-cost</b> long/short backtest, and a significance test — then a clear <b>deploy verdict</b>.</div>',
        unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    cv, ch, cs, ct, cc, cr = st.columns([1.7, 1, 1.2, 1.1, 1.1, 1.1])
    with cv:
        mv_view = st.radio("View", ["🌐 Universe Ranking", "🔎 Single Stock"], horizontal=True)
    with ch:
        mv_h = st.selectbox("Horizon (days)", [5, 10, 20], index=1)
    with cs:
        mv_ls = st.selectbox("Strategy", ["Long / Short", "Long only"])
    with ct:
        mv_top = st.slider("Top/bottom %", 10, 40, 20, step=5)
    with cc:
        mv_cost = st.slider("Cost (bps/side)", 0, 30, 10, step=5)
    with cr:
        st.write("")
        mv_run = st.button("⚙️  Run Model", use_container_width=True, type="primary")

    _key = f"{mv_h}|{mv_ls}|{mv_top}|{mv_cost}"
    _cached = st.session_state.get("univ_result")
    _valid  = _cached and _cached.get("key") == _key

    if mv_run or not _valid:
        try:
            _prog = st.progress(0.0, text="Building the Nifty-50 panel…")
            def _pcb(i, tot, name):
                _prog.progress(min((i + 1) / tot * 0.55, 0.55),
                               text=f"Fetching & engineering features — {name} ({i+1}/{tot})")
            with st.spinner("Training pooled model, walk-forward backtesting & validating… (~60–90s)"):
                res = run_universe_model(
                    NIFTY_50, period="6y", horizon=mv_h,
                    top_frac=mv_top / 100.0, long_short=(mv_ls == "Long / Short"),
                    cost_bps=float(mv_cost), slippage_bps=float(mv_cost) * 0.5,
                    n_folds=4, progress_cb=_pcb)
            _prog.empty()
            st.session_state.univ_result = {"key": _key, "res": res}
        except Exception as e:
            st.error(f"Model error: {e}")
            st.session_state.univ_result = None

    _ur = st.session_state.get("univ_result")
    if _ur and _ur.get("res"):
        res = _ur["res"]
        _vc = {"GREEN": GW_GREEN, "AMBER": "#F59E0B", "RED": GW_RED}[res.verdict]
        _vemoji = {"GREEN": "✅", "AMBER": "⚠️", "RED": "⛔"}[res.verdict]

        # ── Shared verdict banner ─────────────────────────────────────────────
        _lo, _hi = res.auc_ci
        st.markdown(f"""
        <div class="gw-card" style="border:1.5px solid {_vc}66;
             background:radial-gradient(420px 120px at 15% 0%,{_vc}1A,transparent 70%),#FFFFFF;
             box-shadow:0 0 26px {_vc}22;padding:16px 22px;margin-bottom:14px;">
            <div style="font-size:0.70rem;color:#6B7280;letter-spacing:1px;">
                DEPLOY VERDICT &nbsp;·&nbsp; {res.n_stocks} stocks · {res.n_rows:,} rows · {res.horizon}-day horizon · {res.n_features} features
            </div>
            <span style="font-size:1.7rem;font-weight:800;color:{_vc};text-shadow:0 0 16px {_vc}55;">
                {_vemoji} {res.verdict}</span>
            <span style="font-size:0.9rem;color:#4B5563;margin-left:12px;">{res.verdict_reason}</span>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # UNIVERSE RANKING VIEW
        # =====================================================================
        if mv_view.startswith("🌐"):
            # model-quality + economic cards
            _auc_col = GW_GREEN if _lo > 0.5 else GW_RED
            _sh_col  = GW_GREEN if res.sharpe > 0.3 else ("#F59E0B" if res.sharpe > 0 else GW_RED)
            _beat    = res.ann_return > res.bench_ann
            m = st.columns(4)
            for _c, (_l, _v, _s, _col) in zip(m, [
                ("OOS ROC-AUC", f"{res.auc:.3f}", f"95% CI {_lo:.3f}–{_hi:.3f}", _auc_col),
                ("Net Sharpe",  f"{res.sharpe:.2f}", f"p={res.sharpe_pvalue:.2f} vs random", _sh_col),
                ("Strategy Ann. Return", f"{res.ann_return*100:+.1f}%", f"index {res.bench_ann*100:+.1f}%",
                 GW_GREEN if _beat else GW_RED),
                ("Max Drawdown", f"{res.max_dd*100:.0f}%", f"hit {res.hit_rate:.0f}% · {res.n_periods} periods", GW_NAVY),
            ]):
                with _c:
                    st.markdown(f"""<div class="gw-stat-card">
                        <div class="gw-stat-label">{_l}</div>
                        <div class="gw-stat-value" style="color:{_col};">{_v}</div>
                        <div class="gw-stat-sub" style="color:#6B7280;">{_s}</div></div>""",
                        unsafe_allow_html=True)

            # backtest equity vs benchmark
            st.markdown('<div class="gw-section-title" style="margin-top:18px;font-size:0.88rem;">📈 Long/Short Basket — Net of Costs vs Equal-Weight Index</div>', unsafe_allow_html=True)
            eq, be = res.equity, res.bench_equity
            xi = [str(d.date()) if hasattr(d, "date") else str(d) for d in eq.index]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xi, y=be.values, mode="lines", name="Equal-weight index",
                          line=dict(color=GW_NAVY, width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=xi, y=eq.values, mode="lines", name=f"{mv_ls} strategy (net)",
                          line=dict(color=_vc, width=3)))
            groww_fig(fig, 360, f"₹1 grown over {res.n_periods} out-of-sample {res.horizon}-day periods · {res.cost_bps:.0f} bps round-trip")
            st.plotly_chart(fig, use_container_width=True)

            colL, colR = st.columns([1.5, 1])
            # ranking leaderboard
            with colL:
                st.markdown('<div class="gw-section-title" style="font-size:0.86rem;">🏅 Today\'s Ranking '
                            '<span style="font-size:0.7rem;color:#6B7280;font-weight:400;">calibrated P(up) · LONG = strongest, SHORT = weakest</span></div>', unsafe_allow_html=True)
                rk = res.ranking.copy()
                rk["P(up)"] = (rk["proba"] * 100).map(lambda v: f"{v:.1f}%")
                rk["Signal"] = rk["bucket"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "Hold": "⚪ Hold"})
                show = pd.concat([rk.head(8), rk[rk.bucket == "SHORT"].tail(5)])
                show = show[["name", "P(up)", "Signal"]].rename(columns={"name": "Stock"})
                st.dataframe(show.set_index("Stock"), use_container_width=True, height=460)
            # feature importance + reliability
            with colR:
                st.markdown('<div class="gw-section-title" style="font-size:0.86rem;">🔧 Signal Drivers</div>', unsafe_allow_html=True)
                fi = list((res.feat_importance or {}).items())[:8][::-1]
                if fi:
                    fig_fi = go.Figure(go.Bar(x=[v*100 for _, v in fi], y=[k for k, _ in fi],
                              orientation="h", marker=dict(color=GW_GREEN)))
                    groww_fig(fig_fi, 220, "Relative importance (%)")
                    st.plotly_chart(fig_fi, use_container_width=True)
                # reliability
                st.markdown('<div style="font-size:0.74rem;color:#6B7280;margin-top:4px;">Calibration (reliability)</div>', unsafe_allow_html=True)
                rel = res.reliability
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                          line=dict(color="#6B7280", dash="dash", width=1), showlegend=False))
                if rel.get("pred"):
                    fig_rel.add_trace(go.Scatter(x=rel["pred"], y=rel["obs"], mode="lines+markers",
                              line=dict(color=GW_GREEN, width=2), showlegend=False))
                groww_fig(fig_rel, 190, f"Brier {res.brier:.3f} · predicted vs observed")
                st.plotly_chart(fig_rel, use_container_width=True)

            st.caption("Honest read: with free price/volume/earnings/market data the directional edge is tiny and "
                       "regime-dependent. Treat this as a research/validation tool — the verdict tells you whether "
                       "the signal is tradeable *today*, net of costs. It is not investment advice.")

        # =====================================================================
        # SINGLE-STOCK DEEP-DIVE VIEW
        # =====================================================================
        else:
            _names = list(NIFTY_50.keys())
            pick = st.selectbox("Stock", _names, index=_names.index("Reliance Industries"))
            tk = NIFTY_50[pick]
            row = res.ranking[res.ranking.ticker == tk]
            ps  = res.per_stock.get(tk)

            _proba = float(row["proba"].iloc[0]) if len(row) else 0.5
            _bucket = row["bucket"].iloc[0] if len(row) else "Hold"
            _sig = "UP" if _proba >= 0.5 else "DOWN"
            _sig_c = GW_GREEN if _sig == "UP" else GW_RED
            _last = float(row["last_close"].iloc[0]) if len(row) else 0.0

            # forecast + context (cached per stock)
            _sc = st.session_state.get("mv_stock_cache") or {}
            if tk not in _sc:
                with st.spinner(f"Long-term forecast & live context for {pick}…"):
                    _ohlcv = fetch_ohlcv(tk, period="6y")
                    _ctx = fundamental_context(tk)
                    _has = _ohlcv is not None and not _ohlcv.empty
                    _fc  = forecast_price(_ohlcv["Close"], analyst_target=(_ctx.get("analyst") or {}).get("target_mean")) if _has else None
                    _start = _ohlcv.index[-1] if _has else pd.Timestamp.today()
                    _sc[tk] = {"ctx": _ctx, "fc": _fc, "start": _start}
                    st.session_state.mv_stock_cache = _sc
            ctx = _sc[tk]["ctx"]; fc = _sc[tk]["fc"]; _fc_start = _sc[tk]["start"]

            h1, h2, h3 = st.columns(3)
            with h1:
                st.markdown(f"""<div class="gw-card" style="padding:18px 20px;border:1.5px solid {_sig_c}66;
                     background:radial-gradient(320px 120px at 20% 0%,{_sig_c}1F,transparent 70%),#FFFFFF;
                     box-shadow:0 0 26px {_sig_c}22;height:100%;">
                    <div style="font-size:0.68rem;color:#6B7280;letter-spacing:1px;">① {res.horizon}-DAY DIRECTION · pooled model</div>
                    <div style="font-size:2.2rem;font-weight:800;color:{_sig_c};text-shadow:0 0 16px {_sig_c}55;margin:2px 0;">
                        {'▲' if _sig=='UP' else '▼'} {_sig}</div>
                    <div style="font-size:0.82rem;color:#6B7280;">calibrated P(up)
                        <b style="color:{_sig_c};">{_proba*100:.1f}%</b> · rank <b>{_bucket}</b></div>
                </div>""", unsafe_allow_html=True)
            with h2:
                if fc is not None:
                    _mL = fc.months[-1]; _up = fc.upside[_mL]
                    _uc = GW_GREEN if _up > 0 else GW_RED
                    st.markdown(f"""<div class="gw-card" style="padding:18px 20px;border:1.5px solid {GW_PURPLE}55;
                         box-shadow:0 0 22px {GW_PURPLE}1F;height:100%;">
                        <div style="font-size:0.68rem;color:#6B7280;letter-spacing:1px;">② FUTURE PRICE · {_mL} months</div>
                        <div style="font-size:2.2rem;font-weight:800;color:{GW_NAVY};margin:2px 0;">₹{fc.median[_mL]:,.0f}</div>
                        <div style="font-size:0.82rem;color:#6B7280;">median · <b style="color:{_uc};">{_up:+.1f}%</b>
                            · range ₹{fc.p25[_mL]:,.0f}–{fc.p75[_mL]:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="gw-card" style="height:100%;">Forecast unavailable</div>', unsafe_allow_html=True)
            with h3:
                _pauc = ps["auc"] if ps else 0.5
                _pc = GW_GREEN if _pauc > 0.53 else ("#F59E0B" if _pauc > 0.5 else GW_RED)
                st.markdown(f"""<div class="gw-card" style="padding:18px 20px;border:1.5px solid {_pc}55;
                     box-shadow:0 0 22px {_pc}1F;height:100%;">
                    <div style="font-size:0.68rem;color:#6B7280;letter-spacing:1px;">③ THIS STOCK · out-of-sample</div>
                    <div style="font-size:2.2rem;font-weight:800;color:{_pc};margin:2px 0;">
                        {_pauc:.3f}</div>
                    <div style="font-size:0.82rem;color:#6B7280;">ROC-AUC · hit {ps['hit']:.0f}%
                        · {ps['n'] if ps else 0} OOS preds</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="font-size:0.72rem;color:#6B7280;margin:10px 0 2px;">'
                        f'Last close ₹{_last:,.2f} · signal from the pooled Nifty-50 model (verdict <b style="color:{_vc};">{res.verdict}</b> universe-wide). '
                        f'② is a Monte-Carlo range anchored to analyst targets. Per-stock edge is weaker than the pooled model — use the ranking, not one name in isolation.</div>',
                        unsafe_allow_html=True)

            # long-term forecast chart
            if fc is not None:
                st.markdown('<div class="gw-section-title" style="margin-top:14px;font-size:0.86rem;">🔮 Long-Term Price Forecast (1–4 mo)</div>', unsafe_allow_html=True)
                _fd = pd.bdate_range(start=_fc_start, periods=len(fc.path_median))
                _fi = [str(d.date()) for d in _fd]
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=_fi, y=fc.path_p95, mode="lines", name="95th",
                          line=dict(color="rgba(56,126,209,0.15)", width=1), showlegend=False))
                figf.add_trace(go.Scatter(x=_fi, y=fc.path_p5, mode="lines", name="5th–95th",
                          line=dict(color="rgba(56,126,209,0.15)", width=1), fill="tonexty",
                          fillcolor="rgba(56,126,209,0.10)"))
                figf.add_trace(go.Scatter(x=_fi, y=fc.path_p75, mode="lines",
                          line=dict(color="rgba(76,166,76,0.2)", width=1), showlegend=False))
                figf.add_trace(go.Scatter(x=_fi, y=fc.path_p25, mode="lines", name="25th–75th",
                          line=dict(color="rgba(76,166,76,0.2)", width=1), fill="tonexty",
                          fillcolor="rgba(76,166,76,0.12)"))
                figf.add_trace(go.Scatter(x=_fi, y=fc.path_median, mode="lines", name="Median",
                          line=dict(color=GW_GREEN, width=3)))
                if fc.analyst_target:
                    figf.add_trace(go.Scatter(x=[_fi[-1]], y=[fc.analyst_target], mode="markers",
                              name="Analyst target", marker=dict(color="#F59E0B", size=10, symbol="star")))
                groww_fig(figf, 340, f"{pick} — {fc.months[-1]}-month projection")
                st.plotly_chart(figf, use_container_width=True)
                _rows = [{"Horizon": f"{mm} mo", "Median": f"₹{fc.median[mm]:,.0f}",
                          "Upside": f"{fc.upside[mm]:+.1f}%",
                          "50% range": f"₹{fc.p25[mm]:,.0f}–{fc.p75[mm]:,.0f}"} for mm in fc.months]
                st.dataframe(pd.DataFrame(_rows).set_index("Horizon"), use_container_width=True)

            # live context
            _ctx_bias = float(ctx.get("bias", 0.0))
            _ctx_l = ctx.get("label", "Neutral")
            _ctx_c = GW_GREEN if _ctx_bias > 0.15 else GW_RED if _ctx_bias < -0.15 else "#F59E0B"
            _an = ctx.get("analyst", {}) or {}; _scr = ctx.get("screener", {}) or {}; _ne = ctx.get("news", {}) or {}
            st.markdown(f'<div class="gw-section-title" style="margin-top:12px;font-size:0.86rem;">🌐 Live Context '
                        f'<span style="font-size:0.7rem;color:#6B7280;font-weight:400;">not backtested — {_ctx_l} ({_ctx_bias:+.2f})</span></div>', unsafe_allow_html=True)
            cA, cB, cC = st.columns(3)
            def _g(v, s=""):
                return f"{v:g}{s}" if isinstance(v, (int, float)) else "—"
            _up = _an.get("target_upside")
            cA.markdown(f"""<div class="gw-card"><div class="gw-stat-label">FUNDAMENTALS</div>
                <div style="font-size:0.82rem;color:#4B5563;margin-top:6px;">P/E {_g(_scr.get('pe'))} · ROCE {_g(_scr.get('roce'),'%')} · ROE {_g(_scr.get('roe'),'%')}</div></div>""", unsafe_allow_html=True)
            cB.markdown(f"""<div class="gw-card"><div class="gw-stat-label">ANALYSTS</div>
                <div style="font-size:0.82rem;color:#4B5563;margin-top:6px;">{_an.get('recommendation') or '—'} · target upside {f'{_up:+.1f}%' if _up is not None else '—'}</div></div>""", unsafe_allow_html=True)
            cC.markdown(f"""<div class="gw-card"><div class="gw-stat-label">NEWS SENTIMENT</div>
                <div style="font-size:0.82rem;color:#4B5563;margin-top:6px;">{_ne.get('label','—')} ({_ne.get('score',0):+.2f}) · {_ne.get('n',0)} headlines</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="gw-card" style="text-align:center;padding:32px;">
            <div style="font-size:2rem;margin-bottom:8px;">🎯</div>
            <div style="font-weight:700;color:#3C3C3C;">Set the horizon & costs, then Run Model</div>
            <div style="color:#6B7280;font-size:0.82rem;margin-top:6px;">
                Trains one pooled model across the whole Nifty-50, walk-forward validates it, backtests a
                cost-aware long/short basket, and gives an honest deploy verdict (~60–90s, then cached).
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
