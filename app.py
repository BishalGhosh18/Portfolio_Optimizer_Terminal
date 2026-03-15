"""
Terminal-Style NSE/BSE Portfolio Optimizer & Risk Dashboard
===========================================================
• Bloomberg/hacker terminal aesthetic — dark bg, green-on-black, monospace
• Select ANY company listed on NSE or BSE
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

# ── Local modules ───────────────────────────────────────────────────────────────
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
from optimizer import (
    run_strategy, all_strategies_summary, STRATEGIES,
)
from predictor import (
    PREDICTION_MODELS, run_prediction, compute_technical_indicators,
)

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TERMINAL | Portfolio Optimizer",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Terminal CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Share+Tech+Mono&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Share Tech Mono', 'Courier New', monospace !important;
    background-color: #0a0e1a !important;
    color: #00ff88 !important;
}

/* ── Main container ── */
.main .block-container {
    background-color: #0a0e1a !important;
    padding-top: 1rem;
    max-width: 100%;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #060810 !important;
    border-right: 1px solid #00ff8833 !important;
}
[data-testid="stSidebar"] * {
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Headings ── */
h1, h2, h3 {
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Inputs & selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div,
.stNumberInput > div > div {
    background-color: #0d1120 !important;
    border: 1px solid rgba(0,255,136,0.27) !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput input, .stNumberInput input {
    background-color: #0d1120 !important;
    color: #00ff88 !important;
    border: 1px solid rgba(0,255,136,0.27) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #001a0d !important;
    color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #00ff88 !important;
    color: #0a0e1a !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: #00aa55 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── DataFrames ── */
.dataframe, [data-testid="stDataFrame"] {
    background-color: #0d1120 !important;
    color: #00ff88 !important;
    border: 1px solid rgba(0,255,136,0.13) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}
thead tr th {
    background-color: #001a0d !important;
    color: #00ff88 !important;
    border-bottom: 1px solid rgba(0,255,136,0.27) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
tbody tr td {
    border-bottom: 1px solid #00ff8811 !important;
}
tbody tr:hover td {
    background-color: #001a0d !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    background-color: #060810 !important;
    color: #00aa55 !important;
    border: 1px solid rgba(0,255,136,0.13) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #001a0d !important;
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(0,255,136,0.13) !important;
}

/* ── Alerts ── */
.stAlert {
    background-color: #0d1120 !important;
    border: 1px solid #00ff8833 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #0d1120 !important;
    color: #00ff88 !important;
    border: 1px solid rgba(0,255,136,0.13) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.streamlit-expanderContent {
    background-color: #060810 !important;
    border: 1px solid #00ff8811 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: rgba(0,255,136,0.27); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00ff88; }

/* ── Ticker strip ── */
.ticker-strip {
    background: #060810;
    border: 1px solid #00ff8833;
    border-radius: 4px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    overflow-x: auto;
    white-space: nowrap;
    margin-bottom: 12px;
}
.tick-up   { color: #00ff88; }
.tick-down { color: #ff4444; }
.tick-flat { color: #aaaaaa; }

/* ── Terminal box ── */
.term-box {
    background: #060810;
    border: 1px solid #00ff8833;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #00ff88;
    margin-bottom: 8px;
}
.term-header {
    color: #00ff88;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid rgba(0,255,136,0.13);
    padding-bottom: 4px;
    margin-bottom: 8px;
}
.term-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    border-bottom: 1px solid #00ff8811;
}
.term-label { color: #00aa55; }
.term-value { color: #00ff88; font-weight: 700; }
.term-warn  { color: #ffaa00; }
.term-crit  { color: #ff4444; }

/* ── Blinking cursor ── */
.blink {
    animation: blink-cursor 1s step-start infinite;
    color: #00ff88;
}
@keyframes blink-cursor {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ── Status bar ── */
.status-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #060810;
    border-top: 1px solid #00ff8833;
    padding: 4px 16px;
    font-size: 0.65rem;
    color: #00aa55;
    font-family: 'JetBrains Mono', monospace;
    z-index: 9999;
    display: flex;
    justify-content: space-between;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #00ff88 transparent transparent transparent !important;
}

/* ── Radio ── */
.stRadio label { color: #00ff88 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Checkbox ── */
.stCheckbox label { color: #00ff88 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Progress bar ── */
.stProgress > div > div { background-color: #00ff88 !important; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { background: #0a0e1a !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _vline(fig: go.Figure, x_val, label: str = "▶ TODAY"):
    """Add a vertical dashed line without triggering the plotly 6.x add_vline bug."""
    xs = str(x_val.date()) if hasattr(x_val, "date") else str(x_val)
    fig.add_shape(
        type="line", x0=xs, x1=xs, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#00ff88", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=xs, y=0.97, yref="paper",
        text=label, showarrow=False,
        font=dict(color="#00ff88", size=9, family="JetBrains Mono"),
        xanchor="left",
    )


def term_fig(fig: go.Figure, height: int = 380, title: str = "") -> go.Figure:
    """Apply terminal dark theme to a plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(
            text=title.upper(),
            font=dict(family="JetBrains Mono", size=11, color="#00ff88"),
            x=0.01, y=0.99,
        ),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1120",
        font=dict(family="JetBrains Mono", color="#00aa55", size=10),
        legend=dict(
            bgcolor="#060810", bordercolor="rgba(0,255,136,0.2)", borderwidth=1,
            font=dict(family="JetBrains Mono", color="#00ff88", size=9),
        ),
        xaxis=dict(
            gridcolor="rgba(0,255,136,0.07)", linecolor="rgba(0,255,136,0.2)",
            tickfont=dict(family="JetBrains Mono", color="#00aa55", size=9),
            title_font=dict(family="JetBrains Mono", color="#00aa55"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,255,136,0.07)", linecolor="rgba(0,255,136,0.2)",
            tickfont=dict(family="JetBrains Mono", color="#00aa55", size=9),
            title_font=dict(family="JetBrains Mono", color="#00aa55"),
        ),
        margin=dict(l=50, r=20, t=35, b=40),
    )
    return fig


def _colour_risk(level: str) -> str:
    m = {"Low": "tick-up", "Moderate": "tick-flat", "High": "term-warn", "Very High": "term-crit"}
    return m.get(level, "tick-flat")


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="term-header">■ SYSTEM CONFIG</div>', unsafe_allow_html=True)
    st.markdown("```\nPORTFOLIO TERMINAL v3.0\nNSE/BSE ANALYTICS ENGINE\n```")

    auto_refresh = st.toggle("AUTO REFRESH", value=True)
    refresh_secs = st.select_slider(
        "REFRESH INTERVAL",
        options=[5, 10, 15, 30, 60],
        value=5,
        format_func=lambda x: f"{x}s",
        disabled=not auto_refresh,
    )

    exchange = st.selectbox("EXCHANGE", ["NSE", "BSE"], index=0)
    universe = get_stock_universe(exchange)
    all_names = sorted(universe.keys())

    st.markdown('<div class="term-header">■ PRESET BASKETS</div>', unsafe_allow_html=True)
    PRESETS = {
        "── Select ──": [],
        "TATA GROUP":     [n for n in all_names if "Tata" in n or "TCS" in n],
        "TOP IT":         [n for n in all_names if any(k in n for k in ["TCS","Infosys","Wipro","HCL","Tech Mahindra"])],
        "TOP BANKING":    [n for n in all_names if any(k in n for k in ["HDFC","ICICI","Kotak","SBI","Axis"])],
        "TOP AUTO":       [n for n in all_names if any(k in n for k in ["Maruti","Bajaj Auto","Hero","Tata Motors","M&M","Eicher"])],
        "PHARMA":         [n for n in all_names if any(k in n for k in ["Sun Pharma","Dr Reddy","Cipla","Divi","Lupin","Aurobindo"])],
        "ENERGY":         [n for n in all_names if any(k in n for k in ["Reliance","ONGC","NTPC","Coal India","BPCL","IOC","Power Grid"])],
    }
    preset = st.selectbox("LOAD PRESET", list(PRESETS.keys()))
    preset_vals = PRESETS.get(preset, [])
    preset_vals = [v for v in preset_vals if v in all_names]

    default_sel = preset_vals if preset_vals else (
        [n for n in all_names if "Tata" in n or "TCS" in n][:5]
    )
    selected_names = st.multiselect(
        "SELECT COMPANIES",
        options=all_names,
        default=default_sel[:8],
        help="Pick 2–15 companies",
    )

    st.markdown('<div class="term-header">■ ANALYSIS PARAMS</div>', unsafe_allow_html=True)
    lookback = st.selectbox("LOOKBACK PERIOD", ["6mo", "1y", "2y", "3y", "5y"], index=1)
    opt_strategy = st.selectbox("OPT STRATEGY", STRATEGIES)
    rf_rate = st.number_input("RISK-FREE RATE (%)", value=RISK_FREE_RATE * 100, step=0.1, format="%.2f") / 100

    st.markdown('<div class="term-header">■ PREDICTION</div>', unsafe_allow_html=True)
    pred_company = st.selectbox("PREDICT STOCK", options=all_names, index=all_names.index(selected_names[0]) if selected_names else 0)
    pred_model   = st.selectbox("MODEL", list(PREDICTION_MODELS.keys()))
    pred_horizon = st.slider("HORIZON (DAYS)", 5, 180, 30)

    run_btn = st.button("► RUN ANALYSIS", use_container_width=True)
    pred_btn = st.button("► RUN PREDICTION", use_container_width=True)

    st.markdown("---")
    refresh_status = "ON" if auto_refresh else "OFF"
    st.markdown(f'<div style="font-size:0.65rem;color:#00aa55;">SYS: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>MODE: LIVE FEED &nbsp;|&nbsp; REFRESH: {refresh_status}<br>ENV: PORTFOLIO_OPTIMIZER</div>', unsafe_allow_html=True)

# ── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="live_refresh")

# ── Guard ───────────────────────────────────────────────────────────────────────
if len(selected_names) < 2:
    st.markdown("""
    <div class="term-box">
    <div class="term-header">■ TERMINAL READY</div>
    <pre style="color:#00ff88;font-size:0.85rem;">
    ╔══════════════════════════════════════════════════════════════╗
    ║         NSE/BSE PORTFOLIO ANALYTICS TERMINAL v3.0           ║
    ║                  Bloomberg-Style Interface                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  SELECT ≥ 2 COMPANIES FROM THE SIDEBAR TO BEGIN             ║
    ║                                                              ║
    ║  FEATURES:                                                   ║
    ║   • Live quote feed with real-time price data                ║
    ║   • Interactive OHLCV charts with technical indicators       ║
    ║   • Risk metrics: VaR, CVaR, Beta, Sharpe, Sortino          ║
    ║   • Portfolio optimisation (5 strategies)                    ║
    ║   • Efficient frontier via Monte Carlo                       ║
    ║   • Risk scorecard with composite scoring                    ║
    ║   • Price prediction: ARIMA, LinReg, RF, MC, EMA            ║
    ╚══════════════════════════════════════════════════════════════╝
    </pre>
    <span class="blink">█</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="term-box" style="border-color:rgba(0,255,136,0.4);margin-bottom:8px;">
<span style="font-size:1.1rem;font-weight:700;color:#00ff88;letter-spacing:0.15em;">
■ PORTFOLIO ANALYTICS TERMINAL
</span>
<span style="float:right;font-size:0.7rem;color:#00aa55;">
{exchange} &nbsp;|&nbsp; {len(selected_names)} SECURITIES &nbsp;|&nbsp; {lookback.upper()} &nbsp;|&nbsp;
<span class="blink">●</span> LIVE
</span>
</div>
""", unsafe_allow_html=True)

# ── Initialise session state ─────────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "prices" not in st.session_state:
    st.session_state.prices = None
if "quotes" not in st.session_state:
    st.session_state.quotes = {}
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None

# ── Data fetch ───────────────────────────────────────────────────────────────────
tickers = {name: universe[name] for name in selected_names}
ticker_list = list(tickers.values())
company_list = list(tickers.keys())

if run_btn or not st.session_state.data_loaded:
    with st.spinner("FETCHING MARKET DATA..."):
        prices_raw = fetch_price_data(ticker_list, period=lookback)
        prices_raw.columns = [
            next((n for n, t in tickers.items() if t == c), c)
            for c in prices_raw.columns
        ]
        prices_raw = prices_raw[[n for n in company_list if n in prices_raw.columns]].dropna(how="all")
        st.session_state.prices = prices_raw

        quotes_raw = fetch_all_live_quotes(company_list, exchange)
        st.session_state.quotes = quotes_raw
        st.session_state.data_loaded = True

prices: pd.DataFrame = st.session_state.prices
quotes: dict = st.session_state.quotes

if prices is None or prices.empty:
    st.error("NO PRICE DATA RETRIEVED — CHECK TICKERS / NETWORK")
    st.stop()

valid_cols = prices.dropna(axis=1, how="all").columns.tolist()
prices = prices[valid_cols].ffill().bfill().dropna(axis=1, thresh=10)
if prices.shape[1] < 2:
    st.error("INSUFFICIENT DATA — SELECT MORE COMPANIES")
    st.stop()

returns = compute_returns(prices)

# ── Live ticker strip ────────────────────────────────────────────────────────────
strip_parts = []
for name in prices.columns:
    q = quotes.get(name, {})
    price = q.get("price")
    chg   = q.get("change_pct")
    sym   = universe.get(name, name).replace(".NS", "").replace(".BO", "")
    if price is not None:
        css = "tick-up" if (chg or 0) > 0 else ("tick-down" if (chg or 0) < 0 else "tick-flat")
        arrow = "▲" if (chg or 0) > 0 else ("▼" if (chg or 0) < 0 else "─")
        strip_parts.append(
            f'<span class="{css}">{sym} ₹{price:,.1f} {arrow}{abs(chg or 0):.2f}%</span>'
        )
    else:
        strip_parts.append(f'<span class="tick-flat">{sym} ─</span>')

st.markdown(
    '<div class="ticker-strip">'
    + ' &nbsp;&nbsp;|&nbsp;&nbsp; '.join(strip_parts)
    + ' &nbsp;&nbsp;<span class="blink">|</span></div>',
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab_live, tab_charts, tab_risk, tab_opt, tab_score, tab_pred = st.tabs([
    "[ LIVE FEED ]",
    "[ CHARTS ]",
    "[ RISK ]",
    "[ OPTIMIZER ]",
    "[ SCORECARD ]",
    "[ PREDICT ]",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ════════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="term-header">■ LIVE MARKET FEED</div>', unsafe_allow_html=True)

    cols_per_row = 3
    name_list = list(prices.columns)
    for row_start in range(0, len(name_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, name in enumerate(name_list[row_start:row_start + cols_per_row]):
            q = quotes.get(name, {})
            with cols[ci]:
                price  = q.get("price")
                chg    = q.get("change_pct")
                volume = q.get("volume")
                hi52   = q.get("52w_high")
                lo52   = q.get("52w_low")
                mktcap = q.get("market_cap")
                sym    = universe.get(name, name).replace(".NS","").replace(".BO","")

                chg_val = chg if chg is not None else 0.0
                delta_str = f"{chg_val:+.2f}%" if price else "N/A"
                price_str = f"₹{price:,.2f}" if price else "─"

                st.metric(
                    label=f"{sym}",
                    value=price_str,
                    delta=delta_str,
                )
                detail_lines = []
                if hi52:  detail_lines.append(f"52W HI: ₹{hi52:,.1f}")
                if lo52:  detail_lines.append(f"52W LO: ₹{lo52:,.1f}")
                if volume: detail_lines.append(f"VOL: {volume:,.0f}")
                if mktcap: detail_lines.append(f"MCAP: ₹{mktcap/1e9:.1f}B")
                if detail_lines:
                    st.markdown(
                        '<div class="term-box" style="padding:6px 10px;font-size:0.68rem;">'
                        + "<br>".join(detail_lines)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Sector pie ──
    sectors: dict[str, list] = {}
    for name in prices.columns:
        meta = get_stock_universe(exchange).get(name)
        # get_stock_universe returns ticker strings; get sector from universe dict metadata
        # data_fetcher stores universe as {name: ticker}; no sector metadata exposed here
        # use change_pct grouping fallback
        q = quotes.get(name, {})
        chg_val = q.get("change_pct", 0) or 0
        bucket = "GAINERS" if chg_val > 0 else ("LOSERS" if chg_val < 0 else "FLAT")
        sectors.setdefault(bucket, []).append(name)

    sector_df = pd.DataFrame([
        {"Bucket": k, "Count": len(v), "Companies": ", ".join(v[:4]) + ("…" if len(v) > 4 else "")}
        for k, v in sectors.items()
    ])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="term-header">■ MARKET BREADTH</div>', unsafe_allow_html=True)
        for _, row in sector_df.iterrows():
            bar_len = int(row["Count"] / len(prices.columns) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            css = "tick-up" if row["Bucket"] == "GAINERS" else ("tick-down" if row["Bucket"] == "LOSERS" else "tick-flat")
            st.markdown(
                f'<div class="term-box" style="padding:5px 10px;font-size:0.72rem;">'
                f'<span class="{css}">{row["Bucket"]}</span> ({row["Count"]})<br>'
                f'<span style="font-size:0.6rem;color:#004422;">{bar}</span></div>',
                unsafe_allow_html=True,
            )

    with c2:
        chg_data = {
            name: (quotes.get(name, {}).get("change_pct") or 0)
            for name in prices.columns
        }
        chg_series = pd.Series(chg_data).sort_values()
        fig_bar = go.Figure(go.Bar(
            x=chg_series.index.tolist(),
            y=chg_series.values.tolist(),
            marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in chg_series.values],
            text=[f"{v:+.2f}%" for v in chg_series.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=8, color="#00aa55"),
        ))
        term_fig(fig_bar, 260, "DAILY CHANGE %")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Benchmark strip ──
    with st.spinner("LOADING BENCHMARK..."):
        try:
            bench = fetch_benchmark(period=lookback)
            if bench is not None and not bench.empty:
                bench_ret = bench.pct_change().dropna()
                fig_bench = go.Figure()
                for col, colour in zip(bench_ret.columns, ["#00ff88", "#ffaa00"]):
                    cumret = (1 + bench_ret[col]).cumprod() - 1
                    fig_bench.add_trace(go.Scatter(
                        x=cumret.index, y=cumret.values * 100,
                        mode="lines", name=col,
                        line=dict(width=1.5, color=colour),
                    ))
                term_fig(fig_bench, 220, "BENCHMARK — NIFTY50 vs SENSEX")
                st.plotly_chart(fig_bench, use_container_width=True)
        except Exception:
            pass

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown('<div class="term-header">■ PRICE CHARTS & TECHNICALS</div>', unsafe_allow_html=True)

    chart_name = st.selectbox("SELECT SECURITY", list(prices.columns), key="chart_sel")
    ticker_sym = universe.get(chart_name, chart_name)

    with st.spinner(f"LOADING OHLCV FOR {chart_name}..."):
        try:
            ohlcv = fetch_ohlcv(ticker_sym, period=lookback)
            has_ohlcv = ohlcv is not None and not ohlcv.empty
        except Exception:
            has_ohlcv = False

    if has_ohlcv:
        ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz else ohlcv.index

        # ── Candlestick ──
        fig_candle = go.Figure(go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"],
            increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
            increasing_fillcolor="rgba(0,255,136,0.27)", decreasing_fillcolor="rgba(255,68,68,0.27)",
            name="OHLC",
        ))
        # MA overlays
        for win, colour in [(20, "#ffaa00"), (50, "#4488ff"), (200, "#ff44ff")]:
            if len(ohlcv) > win:
                ma = ohlcv["Close"].rolling(win).mean()
                fig_candle.add_trace(go.Scatter(
                    x=ohlcv.index, y=ma, mode="lines", name=f"MA{win}",
                    line=dict(width=1, color=colour, dash="dot"),
                ))
        # Bollinger Bands
        if len(ohlcv) > 20:
            mid = ohlcv["Close"].rolling(20).mean()
            std = ohlcv["Close"].rolling(20).std()
            fig_candle.add_trace(go.Scatter(x=ohlcv.index, y=mid + 2 * std, mode="lines",
                name="BB+2σ", line=dict(width=1, color="rgba(255,255,255,0.13)"), fill=None))
            fig_candle.add_trace(go.Scatter(x=ohlcv.index, y=mid - 2 * std, mode="lines",
                name="BB-2σ", line=dict(width=1, color="rgba(255,255,255,0.13)"),
                fill="tonexty", fillcolor="rgba(255,255,255,0.03)"))
        term_fig(fig_candle, 400, f"{chart_name} — CANDLESTICK")
        fig_candle.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

        # ── Volume ──
        fig_vol = go.Figure(go.Bar(
            x=ohlcv.index, y=ohlcv.get("Volume", pd.Series()),
            marker_color=["rgba(0,255,136,0.4)" if c >= o else "rgba(255,68,68,0.4)"
                          for c, o in zip(ohlcv["Close"], ohlcv["Open"])],
            name="VOLUME",
        ))
        term_fig(fig_vol, 150, "VOLUME")
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Technical indicators ──
        try:
            ind = compute_technical_indicators(ohlcv)
            c1, c2 = st.columns(2)

            with c1:
                # RSI
                if "RSI_14" in ind.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=ind.index, y=ind["RSI_14"], mode="lines",
                        line=dict(color="#ffaa00", width=1.5), name="RSI(14)",
                    ))
                    fig_rsi.add_hline(y=70, line_color="#ff4444", line_dash="dot", line_width=1)
                    fig_rsi.add_hline(y=30, line_color="#00ff88", line_dash="dot", line_width=1)
                    term_fig(fig_rsi, 200, "RSI (14)")
                    fig_rsi.update_layout(yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_rsi, use_container_width=True)

            with c2:
                # MACD
                if "MACD" in ind.columns and "MACD_Signal" in ind.columns:
                    fig_macd = go.Figure()
                    hist = ind["MACD"] - ind["MACD_Signal"]
                    fig_macd.add_trace(go.Bar(
                        x=ind.index, y=hist,
                        marker_color=["rgba(0,255,136,0.4)" if v >= 0 else "rgba(255,68,68,0.4)" for v in hist],
                        name="HIST",
                    ))
                    fig_macd.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], mode="lines",
                        name="MACD", line=dict(color="#00ff88", width=1.5)))
                    fig_macd.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], mode="lines",
                        name="SIGNAL", line=dict(color="#ffaa00", width=1.5)))
                    term_fig(fig_macd, 200, "MACD")
                    st.plotly_chart(fig_macd, use_container_width=True)
        except Exception:
            pass

    else:
        # Fallback to line chart from prices
        fig_line = go.Figure(go.Scatter(
            x=prices.index, y=prices[chart_name],
            mode="lines", name=chart_name,
            line=dict(color="#00ff88", width=1.5),
        ))
        term_fig(fig_line, 380, f"{chart_name} — PRICE")
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Normalised comparison ──
    st.markdown('<div class="term-header">■ NORMALISED PRICE COMPARISON (BASE 100)</div>', unsafe_allow_html=True)
    normed = (prices / prices.iloc[0] * 100)
    fig_norm = go.Figure()
    palette = ["#00ff88","#ffaa00","#4488ff","#ff44ff","#00ccff","#ff8844","#44ff88","#ffff44","#ff4488","#88ff44"]
    for i, col in enumerate(normed.columns):
        fig_norm.add_trace(go.Scatter(
            x=normed.index, y=normed[col],
            mode="lines", name=col,
            line=dict(width=1.5, color=palette[i % len(palette)]),
        ))
    term_fig(fig_norm, 350, "PRICE PERFORMANCE (BASE 100)")
    st.plotly_chart(fig_norm, use_container_width=True)

    # ── Correlation heatmap ──
    corr = correlation_matrix(returns)
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=[[0,"#ff4444"],[0.5,"#0a0e1a"],[1,"#00ff88"]],
        zmin=-1, zmax=1,
    )
    term_fig(fig_heat, 400, "CORRELATION MATRIX")
    fig_heat.update_traces(textfont=dict(family="JetBrains Mono", size=9, color="#ffffff"))
    st.plotly_chart(fig_heat, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK
# ════════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="term-header">■ RISK ANALYTICS</div>', unsafe_allow_html=True)

    # ── Per-stock risk metrics ──
    risk_rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 5:
            continue
        ann_ret  = annualised_return(r)
        ann_vol  = annualised_volatility(r)
        sh       = sharpe_ratio(r, rf_rate)
        mdd      = drawdown_series(r).min()
        vs       = var_summary(r)
        risk_rows.append({
            "COMPANY":     col,
            "ANN RET%":    f"{ann_ret*100:.1f}",
            "ANN VOL%":    f"{ann_vol*100:.1f}",
            "SHARPE":      f"{sh:.2f}",
            "MAX DD%":     f"{mdd*100:.1f}",
            "VaR 95%":     f"{vs.get('hist_95',0)*100:.2f}",
            "CVaR 95%":    f"{vs.get('cvar_95',0)*100:.2f}",
        })

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows).set_index("COMPANY")
        st.dataframe(risk_df, use_container_width=True)

    # ── Rolling volatility ──
    roll_vol = rolling_volatility(returns, window=21) * np.sqrt(TRADING_DAYS) * 100
    fig_rvol = go.Figure()
    for i, col in enumerate(roll_vol.columns):
        fig_rvol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[col], mode="lines", name=col,
            line=dict(width=1.2, color=palette[i % len(palette)]),
        ))
    term_fig(fig_rvol, 320, "ROLLING 21-DAY ANNUALISED VOLATILITY (%)")
    st.plotly_chart(fig_rvol, use_container_width=True)

    # ── Drawdowns ──
    st.markdown('<div class="term-header">■ DRAWDOWN ANALYSIS</div>', unsafe_allow_html=True)
    dd_name = st.selectbox("SELECT STOCK", list(returns.columns), key="dd_sel")
    dd_series = drawdown_series(returns[dd_name].dropna()) * 100
    fig_dd = go.Figure(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines", name="DRAWDOWN",
        fill="tozeroy", fillcolor="rgba(255,68,68,0.12)",
        line=dict(color="#ff4444", width=1.2),
    ))
    term_fig(fig_dd, 260, f"{dd_name} — DRAWDOWN (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── VaR breakdown ──
    st.markdown('<div class="term-header">■ VaR / CVaR SUMMARY</div>', unsafe_allow_html=True)
    var_rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 10:
            continue
        vs = var_summary(r)
        var_rows.append({
            "COMPANY": col,
            "HIST VaR 95": f"{vs.get('hist_95',0)*100:.2f}%",
            "HIST VaR 99": f"{vs.get('hist_99',0)*100:.2f}%",
            "PARAM VaR 95": f"{vs.get('param_95',0)*100:.2f}%",
            "CVaR 95": f"{vs.get('cvar_95',0)*100:.2f}%",
            "CVaR 99": f"{vs.get('cvar_99',0)*100:.2f}%",
        })
    if var_rows:
        st.dataframe(pd.DataFrame(var_rows).set_index("COMPANY"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="term-header">■ PORTFOLIO OPTIMISER</div>', unsafe_allow_html=True)

    with st.spinner(f"OPTIMISING [{opt_strategy.upper()}]..."):
        try:
            weights = run_strategy(opt_strategy, returns)
            summary = all_strategies_summary(returns)
        except Exception as e:
            st.error(f"OPTIMISER ERROR: {e}")
            weights = None
            summary = None

    if weights is not None:
        # ── Weights pie ──
        w_series = pd.Series(weights) * 100
        w_series = w_series[w_series > 0.01].sort_values(ascending=False)
        fig_pie = go.Figure(go.Pie(
            labels=w_series.index.tolist(),
            values=w_series.values.tolist(),
            texttemplate="%{label}<br>%{value:.1f}%",
            textfont=dict(family="JetBrains Mono", size=9),
            marker=dict(colors=palette[:len(w_series)],
                        line=dict(color="#0a0e1a", width=2)),
            hole=0.4,
        ))
        term_fig(fig_pie, 380, f"OPTIMAL WEIGHTS — {opt_strategy}")
        st.plotly_chart(fig_pie, use_container_width=True)

        # ── Bar chart ──
        fig_wbar = go.Figure(go.Bar(
            x=w_series.index.tolist(), y=w_series.values.tolist(),
            marker_color="#00ff88",
            text=[f"{v:.1f}%" for v in w_series.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=9),
        ))
        term_fig(fig_wbar, 260, "WEIGHT ALLOCATION (%)")
        st.plotly_chart(fig_wbar, use_container_width=True)

        # ── Portfolio metrics ──
        port_rets = portfolio_returns(returns, weights)
        ann_r = annualised_return(port_rets)
        ann_v = annualised_volatility(port_rets)
        sh    = sharpe_ratio(port_rets, rf_rate)
        mdd   = drawdown_series(port_rets).min()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ANN RETURN", f"{ann_r*100:.1f}%")
        c2.metric("ANN VOLAT",  f"{ann_v*100:.1f}%")
        c3.metric("SHARPE",     f"{sh:.2f}")
        c4.metric("MAX DD",     f"{mdd*100:.1f}%")

    # ── Strategy comparison ──
    if summary is not None and not summary.empty:
        st.markdown('<div class="term-header">■ STRATEGY COMPARISON</div>', unsafe_allow_html=True)
        st.dataframe(
            summary.style.format("{:.4f}").highlight_max(
                subset=["Ann. Return (%)", "Sharpe Ratio"],
                color="#002200"
            ).highlight_min(
                subset=["Ann. Volatility (%)"],
                color="#002200"
            ),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.markdown('<div class="term-header">■ RISK SCORECARD</div>', unsafe_allow_html=True)

    try:
        try:
            _bench = fetch_benchmark(period=lookback)
            _bench_ret = _bench.pct_change().dropna().iloc[:, 0] if _bench is not None and not _bench.empty else returns.mean(axis=1)
        except Exception:
            _bench_ret = returns.mean(axis=1)
        _bench_ret = _bench_ret.reindex(returns.index).ffill().bfill()
        sc = risk_scorecard(returns, _bench_ret)
        if sc is not None and not sc.empty:
            # ASCII-style table
            for _, row in sc.iterrows():
                score = row.get("Risk Score", 50)
                level = row.get("Risk Level", "Moderate")
                bar_len = int(score / 100 * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                css = _colour_risk(level)
                st.markdown(
                    f'<div class="term-box" style="padding:6px 12px;">'
                    f'<span style="color:#00aa55;font-size:0.7rem;">{row.name:<25}</span> '
                    f'<span class="{css}">[{bar}] {score:.0f}/100 — {level}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Radar chart ──
            cols_radar = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio",
                          "Max Drawdown (%)", "Hist VaR 95% (%)"]
            cols_present = [c for c in cols_radar if c in sc.columns]
            if cols_present:
                fig_radar = go.Figure()
                def _hex_rgba(h, alpha=0.13):
                    h = h.lstrip("#")
                    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                    return f"rgba({r},{g},{b},{alpha})"

                for i, name in enumerate(sc.index[:6]):
                    vals = sc.loc[name, cols_present].values.tolist()
                    color = palette[i % len(palette)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=cols_present + [cols_present[0]],
                        name=name,
                        line=dict(color=color, width=1.5),
                        fill="toself",
                        fillcolor=_hex_rgba(color),
                    ))
                term_fig(fig_radar, 420, "RISK RADAR")
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#0d1120",
                        radialaxis=dict(
                            gridcolor="rgba(0,255,136,0.13)", linecolor="rgba(0,255,136,0.13)",
                            tickfont=dict(family="JetBrains Mono", size=8, color="#00aa55"),
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(0,255,136,0.13)",
                            tickfont=dict(family="JetBrains Mono", size=9, color="#00ff88"),
                        ),
                    )
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ── Full table ──
            num_cols = sc.select_dtypes(include="number").columns.tolist()
            fmt = {c: "{:.4f}" for c in num_cols}
            st.dataframe(
                sc.style.format(fmt, na_rep="N/A"),
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"SCORECARD ERROR: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="term-header">■ PRICE PREDICTION ENGINE</div>', unsafe_allow_html=True)

    # Show model info
    model_info = {
        "ARIMA":             "Auto-regressive Integrated Moving Average — classical time-series model",
        "Linear Regression": "Ridge regression with lag features + RSI/MACD/BB technical signals",
        "Random Forest":     "Ensemble of decision trees with walk-forward validation",
        "Monte Carlo":       "Geometric Brownian Motion simulation (10,000 paths)",
        "EMA Trend":         "Exponential moving average trend extrapolation",
    }
    st.markdown(
        f'<div class="term-box" style="font-size:0.72rem;">'
        f'<span class="term-label">MODEL:</span> {pred_model}<br>'
        f'<span class="term-label">DESC:</span>  {model_info.get(pred_model,"")}<br>'
        f'<span class="term-label">STOCK:</span> {pred_company} &nbsp;|&nbsp; '
        f'<span class="term-label">HORIZON:</span> {pred_horizon} trading days'
        f'</div>',
        unsafe_allow_html=True,
    )

    if pred_btn or (st.session_state.pred_result and
                   st.session_state.pred_result.get("company") == pred_company and
                   st.session_state.pred_result.get("model") == pred_model):

        if pred_btn:
            with st.spinner(f"RUNNING {pred_model.upper()} MODEL ON {pred_company}..."):
                try:
                    # Use already-fetched prices if available, else fetch separately
                    if pred_company in prices.columns:
                        price_series = prices[pred_company].dropna()
                    else:
                        pred_ticker = universe.get(pred_company, pred_company)
                        _raw = fetch_price_data([pred_ticker], period=lookback)
                        if _raw.empty:
                            raise ValueError(f"No price data found for {pred_company}")
                        price_series = _raw.iloc[:, 0].dropna()
                        price_series.name = pred_company
                    if len(price_series) < 30:
                        raise ValueError(f"Not enough data for {pred_company} ({len(price_series)} points)")
                    result = run_prediction(pred_model, price_series, horizon=pred_horizon)
                    st.session_state.pred_result = {
                        "company": pred_company,
                        "model":   pred_model,
                        "result":  result,
                        "series":  price_series,
                    }
                except Exception as e:
                    st.error(f"PREDICTION ERROR: {e}")
                    st.session_state.pred_result = None

        pr = st.session_state.pred_result
        if pr and pr.get("result") is not None:
            result = pr["result"]
            price_series = pr["series"]

            if result.error:
                st.error(f"MODEL ERROR: {result.error}")
            else:
                forecast = result.forecast
                upper    = result.upper_bound
                lower    = result.lower_bound

                # Prediction chart
                fig_pred = go.Figure()

                # Historical (last 180 days max)
                hist = price_series.tail(180)
                hist_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in hist.index]
                fig_pred.add_trace(go.Scatter(
                    x=hist_idx, y=hist.values,
                    mode="lines", name="HISTORICAL",
                    line=dict(color="#00ff88", width=1.5),
                ))

                # Forecast
                if forecast is not None and not forecast.empty:
                    fore_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in forecast.index]
                    fig_pred.add_trace(go.Scatter(
                        x=fore_idx, y=forecast.values,
                        mode="lines", name=f"{pred_model} FORECAST",
                        line=dict(color="#ffaa00", width=2, dash="dash"),
                    ))

                    # Confidence bands
                    if upper is not None and lower is not None:
                        up_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in upper.index]
                        lo_idx = [str(i.date()) if hasattr(i, "date") else str(i) for i in lower.index]
                        fig_pred.add_trace(go.Scatter(
                            x=up_idx, y=upper.values, mode="lines",
                            name="UPPER BAND", line=dict(color="rgba(255,170,0,0.27)", width=1),
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=lo_idx, y=lower.values, mode="lines",
                            name="LOWER BAND", line=dict(color="rgba(255,170,0,0.27)", width=1),
                            fill="tonexty", fillcolor="rgba(255,170,0,0.06)",
                        ))

                # Today line using _vline helper
                _vline(fig_pred, price_series.index[-1])

                term_fig(fig_pred, 450, f"{pred_company} — {pred_model} FORECAST ({pred_horizon}D)")
                st.plotly_chart(fig_pred, use_container_width=True)

                # ── Prediction metrics ──
                metrics = result.metrics or {}
                if metrics:
                    c1, c2, c3, c4 = st.columns(4)
                    last_price = float(price_series.iloc[-1])
                    last_fore  = float(forecast.iloc[-1]) if forecast is not None and not forecast.empty else last_price
                    pred_chg   = (last_fore - last_price) / last_price * 100

                    c1.metric("CURRENT PRICE",  f"₹{last_price:,.2f}")
                    c2.metric(f"PRICE IN {pred_horizon}D", f"₹{last_fore:,.2f}", f"{pred_chg:+.2f}%")
                    if "rmse" in metrics:
                        c3.metric("RMSE", f"₹{metrics['rmse']:.2f}")
                    if "mape" in metrics:
                        c4.metric("MAPE", f"{metrics['mape']:.2f}%")

                    # Model accuracy summary
                    metric_rows = {k.upper(): f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()}
                    if metric_rows:
                        st.markdown('<div class="term-header">■ MODEL METRICS</div>', unsafe_allow_html=True)
                        mc1, mc2 = st.columns(2)
                        items = list(metric_rows.items())
                        for i, (k, v) in enumerate(items):
                            (mc1 if i % 2 == 0 else mc2).markdown(
                                f'<div class="term-box" style="padding:5px 10px;font-size:0.72rem;">'
                                f'<span class="term-label">{k}:</span> <span class="term-value">{v}</span></div>',
                                unsafe_allow_html=True,
                            )

                # ── Scenario analysis ──
                if forecast is not None and not forecast.empty:
                    st.markdown('<div class="term-header">■ SCENARIO ANALYSIS</div>', unsafe_allow_html=True)
                    cur = float(price_series.iloc[-1])
                    f1w  = float(forecast.iloc[min(4, len(forecast)-1)])
                    f1m  = float(forecast.iloc[min(19, len(forecast)-1)])
                    fend = float(forecast.iloc[-1])

                    sc_data = {
                        "HORIZON": ["1 WEEK", "1 MONTH", f"{pred_horizon}D"],
                        "FORECAST": [f"₹{f1w:,.2f}", f"₹{f1m:,.2f}", f"₹{fend:,.2f}"],
                        "CHANGE":   [f"{(f1w-cur)/cur*100:+.2f}%", f"{(f1m-cur)/cur*100:+.2f}%",
                                     f"{(fend-cur)/cur*100:+.2f}%"],
                        "SIGNAL":   ["BUY" if f1w > cur else "SELL",
                                     "BUY" if f1m > cur else "SELL",
                                     "BUY" if fend > cur else "SELL"],
                    }
                    sc_df = pd.DataFrame(sc_data)
                    st.dataframe(sc_df.set_index("HORIZON"), use_container_width=True)
    else:
        st.markdown('<div class="term-box">CONFIGURE MODEL PARAMETERS IN SIDEBAR AND CLICK ► RUN PREDICTION</div>', unsafe_allow_html=True)

# ── Status bar ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="status-bar">'
    f'<span>■ PORTFOLIO TERMINAL v3.0 &nbsp;|&nbsp; {exchange} &nbsp;|&nbsp; {len(prices.columns)} SECURITIES</span>'
    f'<span>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp; <span class="blink">● LIVE</span></span>'
    f'</div>',
    unsafe_allow_html=True,
)
