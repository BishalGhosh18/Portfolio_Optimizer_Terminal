"""
terminal_tab.py — Renders the 🖥️ Terminal tab.
===============================================
7 panels in two rows:

  Top row (3 cols):
    P1 Market Overview  |  P2 TA Chart  |  P3 Order Book

  Bottom row (4 cols):
    P4 News Feed  |  P5 Geo Map  |  P6 Strategy Signals  |  P7 Sector Heatmap

Entry point: render_terminal_tab(universe, exchange)
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from data_fetcher import fetch_ohlcv, fetch_live_quote
from terminal_utils import (
    HEATMAP_STOCKS,
    compute_strategy_signals,
    fetch_gainers_losers,
    fetch_heatmap_data,
    fetch_index_quotes,
    fetch_news,
    simulate_order_book,
)

# ── Groww theme palette ───────────────────────────────────────────────────────
_GREEN  = "#00D09C"
_RED    = "#FF5370"
_PURPLE = "#5367FF"
_NAVY   = "#1B2236"
_GRAY   = "#F6F7F8"
_AMBER  = "#F59E0B"

# ── Timeframe → yfinance period (for fetch_ohlcv) ────────────────────────────
_TF_PERIODS = {
    "5D":  "5d",
    "1M":  "1mo",
    "3M":  "3mo",
    "6M":  "6mo",
    "1Y":  "1y",
    "2Y":  "2y",
}

# ── Geopolitical hotspot data ─────────────────────────────────────────────────
_GEO_EVENTS = [
    {"name": "Eastern Europe",    "lat": 49.0,  "lon":  31.0,  "category": "Active Conflict",
     "color": [255, 83, 112, 210], "detail": "Ongoing armed conflict; supply-chain disruptions"},
    {"name": "Middle East",       "lat": 31.5,  "lon":  35.5,  "category": "Active Conflict",
     "color": [255, 83, 112, 210], "detail": "Regional hostilities; oil price volatility"},
    {"name": "India — NSE / BSE", "lat": 19.08, "lon":  72.88, "category": "Home Market",
     "color": [0, 208, 156, 220],  "detail": "Primary market: Nifty 50 & Sensex"},
    {"name": "China",             "lat": 39.9,  "lon": 116.4,  "category": "Economic Event",
     "color": [83, 103, 255, 200], "detail": "GDP data, trade policy, yuan movement"},
    {"name": "Korean Peninsula",  "lat": 37.5,  "lon": 127.0,  "category": "Tensions",
     "color": [245, 158, 11, 200], "detail": "Geopolitical escalation risk"},
    {"name": "Gulf Region",       "lat": 25.3,  "lon":  51.5,  "category": "Tensions",
     "color": [245, 158, 11, 200], "detail": "Strait of Hormuz; crude oil flow sensitivity"},
    {"name": "Singapore",         "lat":  1.35, "lon": 103.8,  "category": "Economic Hub",
     "color": [83, 103, 255, 200], "detail": "ASEAN financial hub; SGX; regional FII flows"},
    {"name": "East Africa",       "lat": -1.3,  "lon":  36.8,  "category": "Tensions",
     "color": [245, 158, 11, 200], "detail": "Red Sea shipping & trade route risk"},
]

_GEO_LEGEND = {
    "Active Conflict": _RED,
    "Tensions":        _AMBER,
    "Economic Event":  _PURPLE,
    "Economic Hub":    _PURPLE,
    "Home Market":     _GREEN,
}



# ─────────────────────────────────────────────────────────────────────────────
# Tiny HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _panel_title(text: str, badge: str = "") -> None:
    badge_html = (
        f'<span style="background:#E5E7EB;border-radius:4px;padding:1px 7px;'
        f'font-size:0.60rem;color:#9CA3AF;margin-left:8px;">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="font-size:0.80rem;font-weight:700;color:{_NAVY};'
        f'letter-spacing:0.05em;margin-bottom:8px;">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{_GREEN};margin-right:6px;animation:blink 1.6s ease-in-out infinite;">'
        f'</span>{text}{badge_html}</div>',
        unsafe_allow_html=True,
    )


def _vline_shape(fig: go.Figure, x, color: str = "#9CA3AF", row: int = 1) -> None:
    """Add a vertical dashed line without annotation_position (avoids plotly bug)."""
    xs = str(x.date()) if hasattr(x, "date") else str(x)
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=xs, x1=xs, y0=0, y1=1,
        line=dict(color=color, width=1.2, dash="dash"),
        row=row, col=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Market Overview
# ─────────────────────────────────────────────────────────────────────────────

def _market_overview() -> None:
    _panel_title("Market Overview")
    quotes = fetch_index_quotes()

    for idx_name, q in quotes.items():
        price   = q.get("price", 0.0)
        chg_pct = q.get("change_pct", 0.0)
        hi52    = q.get("52w_high")
        lo52    = q.get("52w_low")
        clr     = _GREEN if chg_pct >= 0 else _RED
        arrow   = "▲" if chg_pct >= 0 else "▼"

        strength = 0.5
        if hi52 and lo52 and hi52 != lo52:
            strength = float(np.clip((price - lo52) / (hi52 - lo52), 0, 1))

        hi_s = f"₹{hi52:,.2f}" if hi52 else "—"
        lo_s = f"₹{lo52:,.2f}" if lo52 else "—"

        st.markdown(f"""
        <div style="background:#fff;border-radius:10px;padding:11px 13px;
                    margin-bottom:7px;box-shadow:0 1px 4px rgba(0,0,0,0.06);
                    border-left:4px solid {clr};">
            <div style="font-size:0.66rem;color:#9CA3AF;font-weight:600;
                        letter-spacing:0.07em;">{idx_name}</div>
            <div style="font-size:1.15rem;font-weight:700;color:{_NAVY};margin:2px 0;">
                ₹{price:,.2f}
                <span style="font-size:0.78rem;color:{clr};margin-left:6px;">
                    {arrow} {abs(chg_pct):.2f}%
                </span>
            </div>
            <div style="font-size:0.65rem;color:#9CA3AF;">
                52W H: {hi_s} &nbsp;·&nbsp; 52W L: {lo_s}
            </div>
            <div style="margin-top:5px;background:#E5E7EB;border-radius:4px;height:3px;">
                <div style="width:{strength*100:.1f}%;background:{clr};
                            height:3px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # FII / DII flow (indicative — yfinance does not provide real flow data)
    st.markdown(f"""
    <div style="background:#fff;border-radius:10px;padding:11px 13px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:0.66rem;font-weight:700;color:#9CA3AF;
                    letter-spacing:0.07em;margin-bottom:7px;">FII / DII FLOW</div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.76rem;color:{_NAVY};">
            <span>FII (Indicative)</span>
            <span style="color:{_RED};font-weight:600;">−₹1,240 Cr</span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.76rem;color:{_NAVY};margin-top:4px;">
            <span>DII (Indicative)</span>
            <span style="color:{_GREEN};font-weight:600;">+₹2,180 Cr</span>
        </div>
        <div style="font-size:0.60rem;color:#9CA3AF;margin-top:6px;">
            Indicative values · Updated daily post market close
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Technical Analysis Chart
# ─────────────────────────────────────────────────────────────────────────────

def _ta_chart(universe: dict[str, str]) -> None:
    _panel_title("Technical Analysis Chart")
    names = sorted(universe.keys())

    cc1, cc2 = st.columns([3, 1])
    with cc1:
        sel = st.selectbox(
            "Symbol", names,
            index=names.index("HDFC Bank") if "HDFC Bank" in names else 0,
            key="terminal_chart_sym",
        )
    with cc2:
        tf = st.selectbox(
            "Timeframe", list(_TF_PERIODS.keys()), index=4,
            key="terminal_chart_tf",
        )

    ticker = universe.get(sel, "")
    ohlcv  = fetch_ohlcv(ticker, period=_TF_PERIODS[tf])
    if ohlcv.empty:
        st.warning(f"No data for {sel}.")
        return

    c = ohlcv["Close"].astype(float)
    o = ohlcv["Open"].astype(float)
    h = ohlcv["High"].astype(float)
    l = ohlcv["Low"].astype(float)
    v = ohlcv["Volume"].astype(float) if "Volume" in ohlcv.columns else pd.Series(0.0, index=c.index)

    ema20  = c.ewm(span=20,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    sma200 = c.rolling(200).mean()
    bb_ma  = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_ma + 2 * bb_std
    bb_dn  = bb_ma - 2 * bb_std

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=list(ohlcv.index) + list(ohlcv.index[::-1]),
        y=list(bb_up.values) + list(bb_dn.values[::-1]),
        fill="toself", fillcolor="rgba(83,103,255,0.07)",
        line=dict(width=0), showlegend=False,
        hoverinfo="skip", name="BB Fill",
    ), row=1, col=1)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=o, high=h, low=l, close=c,
        name="Price",
        increasing_line_color=_GREEN, decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN,  decreasing_fillcolor=_RED,
    ), row=1, col=1)

    for series, name_, color, width, dash in [
        (ema20,  "EMA 20",  "#F59E0B", 1.4, "solid"),
        (ema50,  "EMA 50",  _PURPLE,   1.4, "solid"),
        (sma200, "SMA 200", "#9CA3AF", 1.1, "solid"),
        (bb_up,  "BB+2σ",   _PURPLE,   0.7, "dot"),
        (bb_dn,  "BB−2σ",   _PURPLE,   0.7, "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=series,
            name=name_, line=dict(color=color, width=width, dash=dash),
            opacity=0.85,
        ), row=1, col=1)

    # Volume bars
    vol_colors = [_GREEN if float(c.iloc[i]) >= float(o.iloc[i]) else _RED
                  for i in range(len(c))]
    fig.add_trace(go.Bar(
        x=ohlcv.index, y=v, marker_color=vol_colors,
        opacity=0.55, showlegend=False, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        height=430,
        paper_bgcolor="white", plot_bgcolor=_GRAY,
        margin=dict(l=6, r=6, t=6, b=6),
        font=dict(family="Inter", size=10, color=_NAVY),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis=dict(gridcolor="#E5E7EB", tickprefix="₹"),
        yaxis2=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — Order Book (Simulated)
# ─────────────────────────────────────────────────────────────────────────────

def _order_book(universe: dict[str, str]) -> None:
    _panel_title("Order Book", "Simulated")

    names    = sorted(universe.keys())
    ob_stock = st.selectbox(
        "Stock (Order Book)", names,
        index=names.index("ICICI Bank") if "ICICI Bank" in names else 0,
        key="terminal_ob_sym",
    )
    ticker = universe.get(ob_stock, "")
    q      = fetch_live_quote(ticker) or {}
    ltp    = q.get("price", 1000.0)

    # Seed shifts every 5 s to give "live" feel on auto-refresh
    seed = int(time.time() // 5) + sum(ord(ch) for ch in ticker)
    book = simulate_order_book(ltp, seed=seed)

    asks     = list(reversed(book["asks"]))   # worst ask first (top of book display)
    bids     = book["bids"]                   # best bid first
    max_qty  = max(
        max((a["qty"] for a in asks), default=1),
        max((b["qty"] for b in bids), default=1),
    )

    def _row(level: dict, side: str) -> str:
        bar_pct = int(level["qty"] / max_qty * 100)
        bar_clr = "rgba(255,83,112,0.10)" if side == "ask" else "rgba(0,208,156,0.10)"
        txt_clr = _RED if side == "ask" else _GREEN
        return (
            f'<div style="display:flex;align-items:center;padding:2px 0;'
            f'position:relative;border-radius:3px;overflow:hidden;">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:{bar_pct}%;'
            f'background:{bar_clr};z-index:0;"></div>'
            f'<span style="flex:1;font-size:0.68rem;color:#9CA3AF;z-index:1;'
            f'padding-left:3px;">{level["orders"]}</span>'
            f'<span style="flex:2;font-size:0.68rem;color:#6B7280;z-index:1;'
            f'text-align:right;">{level["qty"]:,}</span>'
            f'<span style="flex:2;font-size:0.70rem;font-weight:600;color:{txt_clr};'
            f'z-index:1;text-align:right;padding-right:3px;">₹{level["price"]:,.2f}</span>'
            f'</div>'
        )

    header = (
        '<div style="display:flex;font-size:0.60rem;color:#9CA3AF;'
        'padding:0 3px 4px;border-bottom:1px solid #E5E7EB;margin-bottom:3px;">'
        '<span style="flex:1;">ORD</span>'
        '<span style="flex:2;text-align:right;">QTY</span>'
        '<span style="flex:2;text-align:right;">PRICE</span>'
        '</div>'
    )
    ltp_clr = _GREEN if q.get("change_pct", 0) >= 0 else _RED
    ltp_row = (
        f'<div style="text-align:center;padding:5px;margin:4px 0;border-radius:6px;'
        f'background:linear-gradient(90deg,rgba(0,208,156,0.07),rgba(83,103,255,0.07));'
        f'border-top:1px solid #E5E7EB;border-bottom:1px solid #E5E7EB;">'
        f'<span style="font-size:0.95rem;font-weight:700;color:{ltp_clr};">₹{ltp:,.2f}</span>'
        f'<span style="font-size:0.64rem;color:#9CA3AF;margin-left:8px;">'
        f'Spread {book["spread_pct"]:.4f}% &nbsp;·&nbsp; Vol {book["total_volume"]:,}'
        f'</span></div>'
    )

    ask_rows = "".join(_row(a, "ask") for a in asks)
    bid_rows = "".join(_row(b, "bid") for b in bids)

    st.markdown(
        f'<div style="background:#fff;border-radius:10px;padding:12px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
        f'{header}{ask_rows}{ltp_row}{bid_rows}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Live News Feed
# ─────────────────────────────────────────────────────────────────────────────

def _news_feed() -> None:
    _panel_title("Live News Feed")
    api_key = os.getenv("NEWS_API_KEY", "")
    items   = fetch_news(api_key=api_key)

    if not items:
        st.markdown(
            f'<div style="color:#9CA3AF;font-size:0.78rem;padding:10px;">'
            f'No news available.<br>Set <code>NEWS_API_KEY</code> in .env '
            f'for live feeds, or ensure feedparser is installed for RSS fallback.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    for item in items[:8]:
        clr   = item.get("color", _AMBER)
        label = item.get("label", "NEUTRAL")
        st.markdown(f"""
        <div style="background:#fff;border-radius:9px;padding:9px 11px;
                    margin-bottom:5px;box-shadow:0 1px 3px rgba(0,0,0,0.05);
                    border-left:3px solid {clr};">
            <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;
                        flex-wrap:wrap;">
                <span style="font-size:0.60rem;color:#9CA3AF;">{item.get('time','')}</span>
                <span style="background:#F3F4F6;border-radius:3px;padding:0 5px;
                             font-size:0.60rem;color:#6B7280;">{item.get('source','')}</span>
                <span style="background:{clr};border-radius:3px;padding:0 6px;
                             font-size:0.60rem;color:#fff;font-weight:600;">{label}</span>
            </div>
            <div style="font-size:0.74rem;color:{_NAVY};line-height:1.4;
                        overflow:hidden;display:-webkit-box;
                        -webkit-line-clamp:2;-webkit-box-orient:vertical;">
                {item.get('title','')}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Geopolitical Intelligence Map
# ─────────────────────────────────────────────────────────────────────────────

def _geo_map() -> None:
    _panel_title("Geopolitical Intelligence")

    df = pd.DataFrame(_GEO_EVENTS)

    # Map category → marker colour (hex) for Plotly
    _CAT_COLOR = {
        "Active Conflict": _RED,
        "Tensions":        _AMBER,
        "Economic Event":  _PURPLE,
        "Economic Hub":    _PURPLE,
        "Home Market":     _GREEN,
    }
    # Map category → marker size
    _CAT_SIZE = {
        "Active Conflict": 18,
        "Tensions":        14,
        "Economic Event":  14,
        "Economic Hub":    14,
        "Home Market":     20,
    }
    df["color"]  = df["category"].map(_CAT_COLOR).fillna(_AMBER)
    df["size"]   = df["category"].map(_CAT_SIZE).fillna(14)
    df["label"]  = df["name"] + "<br><i>" + df["category"] + "</i><br>" + df["detail"]

    fig = go.Figure()

    # Draw one trace per category so the legend is clean
    for cat, clr in _CAT_COLOR.items():
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        sz = _CAT_SIZE.get(cat, 14)
        fig.add_trace(go.Scattergeo(
            lat=sub["lat"],
            lon=sub["lon"],
            mode="markers",
            name=cat,
            marker=dict(
                size=sz,
                color=clr,
                opacity=0.85,
                line=dict(width=1, color="rgba(255,255,255,0.5)"),
                symbol="circle",
            ),
            text=sub["label"],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=_NAVY,
                font=dict(color="white", size=11, family="Inter"),
                bordercolor=clr,
            ),
        ))

    fig.update_layout(
        height=340,
        paper_bgcolor=_NAVY,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="center",  x=0.5,
            font=dict(size=9, color="white"),
            bgcolor="rgba(27,34,54,0.7)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,       landcolor="#1e293b",
            showocean=True,      oceancolor="#0f172a",
            showcoastlines=True, coastlinecolor="rgba(255,255,255,0.15)",
            showframe=False,
            showcountries=True,  countrycolor="rgba(255,255,255,0.08)",
            bgcolor=_NAVY,
            lataxis=dict(range=[-50, 75]),
            lonaxis=dict(range=[-30, 150]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 6 — Trading Strategy Signals
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_signals(universe: dict[str, str]) -> None:
    _panel_title("Algo Strategy Signals")

    names    = sorted(universe.keys())
    sig_stock = st.selectbox(
        "Stock (Strategy)", names,
        index=names.index("Infosys") if "Infosys" in names else 0,
        key="terminal_sig_sym",
    )
    ticker    = universe.get(sig_stock, "")
    cache_key = f"terminal_signals_{ticker}"

    refresh   = st.button("↻ Refresh", key="terminal_sig_refresh",
                          use_container_width=True)

    # Compute only on tab activation or explicit refresh — not every rerun
    if cache_key not in st.session_state or refresh:
        with st.spinner("Computing signals…"):
            ohlcv = fetch_ohlcv(ticker, period="1y")
            st.session_state[cache_key] = compute_strategy_signals(ticker, ohlcv)

    signals = st.session_state.get(cache_key, [])

    if not signals:
        st.markdown(
            f'<div style="color:#9CA3AF;font-size:0.78rem;padding:10px;">'
            f'Signals unavailable. Ensure <code>pandas-ta</code> is installed.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    for sig in signals:
        clr  = _GREEN if sig["signal"] == "BUY" else _RED
        tags = "".join(
            f'<span style="background:#F3F4F6;border-radius:3px;padding:0 5px;'
            f'font-size:0.60rem;color:#6B7280;margin-right:3px;">{t}</span>'
            for t in sig.get("tags", [])
        )
        st.markdown(f"""
        <div style="background:#fff;border-radius:10px;padding:11px 13px;
                    margin-bottom:7px;box-shadow:0 1px 4px rgba(0,0,0,0.06);
                    border-top:3px solid {clr};">
            <div style="display:flex;align-items:center;
                        justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:0.78rem;font-weight:700;color:{_NAVY};">
                    {sig['icon']} {sig['name']}
                </span>
                <span style="font-size:0.78rem;font-weight:700;color:{_PURPLE};">
                    #{sig['ticker']}
                </span>
                <span style="background:{clr};color:#fff;border-radius:5px;
                             padding:1px 9px;font-size:0.72rem;font-weight:700;">
                    {sig['signal']}
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:4px;font-size:0.68rem;color:#6B7280;margin-bottom:7px;">
                <div>Entry<br/>
                     <b style="color:{_NAVY};">₹{sig['entry']:,.2f}</b></div>
                <div>Target<br/>
                     <b style="color:{_GREEN};">₹{sig['target']:,.2f}</b></div>
                <div>Stop Loss<br/>
                     <b style="color:{_RED};">₹{sig['stop_loss']:,.2f}</b></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-size:0.68rem;color:#6B7280;margin-bottom:6px;">
                <span>RSI <b style="color:{_NAVY};">{sig['rsi']}</b></span>
                <span>R:R <b style="color:{_NAVY};">{sig['risk_reward']}×</b></span>
                <span>Exp. Ret <b style="color:{_GREEN};">+{sig['exp_return']}%</b></span>
            </div>
            <div style="font-size:0.62rem;color:#9CA3AF;">Win probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(sig["win_prob"]),
                    text=f"{sig['win_prob']*100:.0f}%")
        st.markdown(
            f'<div style="margin-top:-6px;margin-bottom:2px;">{tags}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Panel 7 — Sector Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _sector_heatmap() -> None:
    _panel_title("Sector Heatmap")
    rows = fetch_heatmap_data()
    if not rows:
        st.warning("Could not fetch heatmap data.")
        return

    df = pd.DataFrame(rows)
    df["color_val"] = df["change_pct"].clip(-3.0, 3.0)
    df["val"]       = 1                 # equal area for each stock

    fig = px.treemap(
        df,
        path=["name"],
        values="val",
        color="color_val",
        color_continuous_scale=[
            [0.00, "#7F1D1D"],
            [0.17, "#FF5370"],
            [0.42, "#FECDD3"],
            [0.50, "#E5E7EB"],
            [0.58, "#D1FAE5"],
            [0.83, "#00D09C"],
            [1.00, "#065F46"],
        ],
        range_color=[-3, 3],
        custom_data=["ticker", "change_pct", "price"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Change: %{customdata[1]:+.2f}%<br>"
            "Price: ₹%{customdata[2]:,.2f}<extra></extra>"
        ),
        textfont=dict(size=10, color="white"),
        marker=dict(pad=dict(t=4, l=2, r=2, b=2)),
    )
    fig.update_layout(
        height=360,
        paper_bgcolor="white",
        margin=dict(l=4, r=4, t=4, b=4),
        font=dict(family="Inter"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 8 — Gainers & Losers
# ─────────────────────────────────────────────────────────────────────────────

def _gainers_losers() -> None:
    _panel_title("📈 Gainers & Losers of the Day", "Top 10 / Bottom 10")

    gainers, losers = fetch_gainers_losers(top_n=10)

    def _row_html(name: str, ticker: str, price: float, pct: float, is_gain: bool) -> str:
        color  = _GREEN if is_gain else _RED
        sign   = "+" if pct >= 0 else ""
        arrow  = "▲" if pct >= 0 else "▼"
        bar_w  = min(abs(pct) / 5 * 100, 100)   # scale: 5% = full bar
        bar_bg = "rgba(0,208,156,0.12)" if is_gain else "rgba(255,83,112,0.12)"
        return (
            f'<div style="position:relative;display:flex;align-items:center;'
            f'justify-content:space-between;padding:5px 8px;margin-bottom:3px;'
            f'border-radius:6px;background:white;overflow:hidden;">'
            f'<div style="position:absolute;left:0;top:0;height:100%;'
            f'width:{bar_w:.1f}%;background:{bar_bg};z-index:0;"></div>'
            f'<div style="position:relative;z-index:1;display:flex;flex-direction:column;">'
            f'<span style="font-size:0.72rem;font-weight:600;color:{_NAVY};">{name}</span>'
            f'<span style="font-size:0.60rem;color:#9CA3AF;">{ticker}</span>'
            f'</div>'
            f'<div style="position:relative;z-index:1;text-align:right;">'
            f'<div style="font-size:0.72rem;font-weight:600;color:{_NAVY};">₹{price:,.1f}</div>'
            f'<div style="font-size:0.68rem;font-weight:700;color:{color};">'
            f'{arrow} {sign}{pct:.2f}%</div>'
            f'</div>'
            f'</div>'
        )

    col_g, col_l = st.columns(2)

    with col_g:
        st.markdown(
            f'<div style="font-size:0.70rem;font-weight:700;color:{_GREEN};'
            f'margin-bottom:6px;letter-spacing:0.04em;">▲ TOP GAINERS</div>',
            unsafe_allow_html=True,
        )
        if gainers:
            html = "".join(_row_html(r["name"], r["ticker"], r["price"], r["change_pct"], True) for r in gainers)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No data available")

    with col_l:
        st.markdown(
            f'<div style="font-size:0.70rem;font-weight:700;color:{_RED};'
            f'margin-bottom:6px;letter-spacing:0.04em;">▼ TOP LOSERS</div>',
            unsafe_allow_html=True,
        )
        if losers:
            html = "".join(_row_html(r["name"], r["ticker"], r["price"], r["change_pct"], False) for r in losers)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No data available")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_terminal_tab(
    universe: dict[str, str],
    exchange: str = "NSE",
) -> None:
    """Render the full Terminal tab. Called from app.py tab6 block."""

    # Blinking dot animation
    st.markdown("""
    <style>
    @keyframes blink {
      0%,100% { opacity: 1; }
      50%      { opacity: 0.25; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="gw-section-title">🖥️ Terminal — Live Market Intelligence</div>',
        unsafe_allow_html=True,
    )

    # ── Top row ───────────────────────────────────────────────────────────────
    top_l, top_c, top_r = st.columns([1, 2.2, 1])
    with top_l:
        _market_overview()
    with top_c:
        _ta_chart(universe)
    with top_r:
        _order_book(universe)

    st.markdown(
        '<div style="margin:14px 0;border-top:1px solid #E5E7EB;"></div>',
        unsafe_allow_html=True,
    )

    # ── Bottom row ────────────────────────────────────────────────────────────
    bot_l, bot_cl, bot_cr, bot_r = st.columns([1, 1.3, 1, 1.3])
    with bot_l:
        _news_feed()
    with bot_cl:
        _geo_map()
    with bot_cr:
        _strategy_signals(universe)
    with bot_r:
        _sector_heatmap()

    st.markdown(
        '<div style="margin:14px 0;border-top:1px solid #E5E7EB;"></div>',
        unsafe_allow_html=True,
    )

    # ── Gainers & Losers row ──────────────────────────────────────────────────
    with st.container():
        _gainers_losers()

    st.markdown(
        '<div style="font-size:0.68rem;color:#9CA3AF;text-align:center;margin-top:10px;">'
        'Order book is <b>simulated</b>. FII/DII flows are indicative. '
        'All content is for educational purposes only — not financial advice.'
        '</div>',
        unsafe_allow_html=True,
    )
