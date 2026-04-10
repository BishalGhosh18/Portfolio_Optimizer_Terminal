"""
insights_tab.py — Stock Picks & Investment Insights tab.
=========================================================
Analyses 50+ NSE stocks across multiple timeframes and signals,
then ranks them into:
  • Short-term picks  (1–4 weeks)  — momentum, breakout, RSI, volume surge
  • Long-term picks   (6–24 months)— fundamentals proxy, trend strength,
                                     52-week position, consistent returns

All data via data_fetcher.py (fetch_ohlcv, fetch_live_quote).
No external APIs needed.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from data_fetcher import fetch_live_quote, fetch_ohlcv

# ── Groww palette ─────────────────────────────────────────────────────────────
_GREEN  = "#00D09C"
_RED    = "#FF5370"
_PURPLE = "#5367FF"
_NAVY   = "#1B2236"
_GRAY   = "#F6F7F8"
_AMBER  = "#F59E0B"
_WHITE  = "#FFFFFF"

# ── Universe to screen ────────────────────────────────────────────────────────
_SCREEN_STOCKS: dict[str, str] = {
    # Banking & Finance
    "HDFC Bank":        "HDFCBANK.NS",
    "ICICI Bank":       "ICICIBANK.NS",
    "SBI":              "SBIN.NS",
    "Axis Bank":        "AXISBANK.NS",
    "Kotak Bank":       "KOTAKBANK.NS",
    "Bajaj Finance":    "BAJFINANCE.NS",
    "PFC":              "PFC.NS",
    "REC":              "RECLTD.NS",
    "IndusInd Bank":    "INDUSINDBK.NS",
    # IT
    "TCS":              "TCS.NS",
    "Infosys":          "INFY.NS",
    "Wipro":            "WIPRO.NS",
    "HCL Tech":         "HCLTECH.NS",
    "Tech Mahindra":    "TECHM.NS",
    "LTIMindtree":      "LTIM.NS",
    # Energy / Infra
    "Reliance":         "RELIANCE.NS",
    "ONGC":             "ONGC.NS",
    "NTPC":             "NTPC.NS",
    "Power Grid":       "POWERGRID.NS",
    "Adani Ports":      "ADANIPORTS.NS",
    "Adani Green":      "ADANIGREEN.NS",
    "Coal India":       "COALINDIA.NS",
    "L&T":              "LT.NS",
    "Tata Power":       "TATAPOWER.NS",
    # Telecom / FMCG
    "Bharti Airtel":    "BHARTIARTL.NS",
    "HUL":              "HINDUNILVR.NS",
    "ITC":              "ITC.NS",
    "Nestle":           "NESTLEIND.NS",
    "Britannia":        "BRITANNIA.NS",
    "Dabur":            "DABUR.NS",
    # Auto
    "Maruti":           "MARUTI.NS",
    "M&M":              "M&M.NS",
    "Hero MotoCorp":    "HEROMOTOCO.NS",
    "Bajaj Auto":       "BAJAJ-AUTO.NS",
    "Tata Motors":      "TMCV.NS",
    "Eicher Motors":    "EICHERMOT.NS",
    # Pharma
    "Sun Pharma":       "SUNPHARMA.NS",
    "Dr Reddy's":       "DRREDDY.NS",
    "Cipla":            "CIPLA.NS",
    "Divi's Labs":      "DIVISLAB.NS",
    "Lupin":            "LUPIN.NS",
    # Metals
    "Tata Steel":       "TATASTEEL.NS",
    "JSW Steel":        "JSWSTEEL.NS",
    "Hindalco":         "HINDALCO.NS",
    "Vedanta":          "VEDL.NS",
    # Consumer / Misc
    "Titan":            "TITAN.NS",
    "Asian Paints":     "ASIANPAINT.NS",
    "DLF":              "DLF.NS",
    "Zomato":           "ZOMATO.NS",
    "Tata Consultancy": "TCS.NS",
    "Tata Consumer":    "TATACONSUM.NS",
}

_SECTOR_MAP: dict[str, str] = {
    "HDFC Bank": "Banking", "ICICI Bank": "Banking", "SBI": "Banking",
    "Axis Bank": "Banking", "Kotak Bank": "Banking", "Bajaj Finance": "Finance",
    "PFC": "Finance", "REC": "Finance", "IndusInd Bank": "Banking",
    "TCS": "IT", "Infosys": "IT", "Wipro": "IT", "HCL Tech": "IT",
    "Tech Mahindra": "IT", "LTIMindtree": "IT",
    "Reliance": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "Power Grid": "Energy", "Adani Ports": "Infra", "Adani Green": "Energy",
    "Coal India": "Energy", "L&T": "Infra", "Tata Power": "Energy",
    "Bharti Airtel": "Telecom", "HUL": "FMCG", "ITC": "FMCG",
    "Nestle": "FMCG", "Britannia": "FMCG", "Dabur": "FMCG",
    "Maruti": "Auto", "M&M": "Auto", "Hero MotoCorp": "Auto",
    "Bajaj Auto": "Auto", "Tata Motors": "Auto", "Eicher Motors": "Auto",
    "Sun Pharma": "Pharma", "Dr Reddy's": "Pharma", "Cipla": "Pharma",
    "Divi's Labs": "Pharma", "Lupin": "Pharma",
    "Tata Steel": "Metals", "JSW Steel": "Metals", "Hindalco": "Metals",
    "Vedanta": "Metals",
    "Titan": "Consumer", "Asian Paints": "Consumer", "DLF": "Realty",
    "Zomato": "Technology", "Tata Consultancy": "IT", "Tata Consumer": "FMCG",
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_signals(name: str, ticker: str) -> dict | None:
    """
    Fetch 1y OHLCV for ticker and compute all scoring signals.
    Returns None if data unavailable.
    """
    try:
        df = fetch_ohlcv(ticker, period="1y")
        if df is None or df.empty or len(df) < 50:
            return None

        c  = df["Close"].astype(float)
        v  = df["Volume"].astype(float)
        h  = df["High"].astype(float)
        l  = df["Low"].astype(float)

        # ── Trend indicators ──────────────────────────────────────────────────
        ema20  = c.ewm(span=20,  adjust=False).mean()
        ema50  = c.ewm(span=50,  adjust=False).mean()
        sma200 = c.rolling(200, min_periods=100).mean()

        price_now  = float(c.iloc[-1])
        ema20_now  = float(ema20.iloc[-1])
        ema50_now  = float(ema50.iloc[-1])
        sma200_now = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else price_now

        above_ema20  = price_now > ema20_now
        above_ema50  = price_now > ema50_now
        above_sma200 = price_now > sma200_now
        ema_bullish  = ema20_now > ema50_now   # golden cross proxy

        # ── RSI-14 ───────────────────────────────────────────────────────────
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - 100 / (1 + gain / (loss + 1e-8))).iloc[-1])

        # ── Momentum ─────────────────────────────────────────────────────────
        ret_1w  = (price_now / float(c.iloc[-5])  - 1) * 100 if len(c) >= 5  else 0
        ret_1m  = (price_now / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else 0
        ret_3m  = (price_now / float(c.iloc[-63]) - 1) * 100 if len(c) >= 63 else 0
        ret_6m  = (price_now / float(c.iloc[-126])- 1) * 100 if len(c) >= 126 else 0
        ret_1y  = (price_now / float(c.iloc[0])   - 1) * 100

        # ── Volatility ───────────────────────────────────────────────────────
        lr        = np.log(c / c.shift(1)).dropna()
        vol_21    = float(lr.tail(21).std() * np.sqrt(252) * 100)  # annualised %
        vol_63    = float(lr.tail(63).std() * np.sqrt(252) * 100)

        # ── Volume surge ─────────────────────────────────────────────────────
        avg_vol_20 = float(v.tail(21).iloc[:-1].mean())  # prev 20 days
        vol_today  = float(v.iloc[-1])
        vol_ratio  = vol_today / (avg_vol_20 + 1e-8)

        # ── Bollinger Band position ───────────────────────────────────────────
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_pct = float(((c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-8)).iloc[-1])  # 0=lower, 1=upper

        # ── 52-week position ─────────────────────────────────────────────────
        hi52 = float(c.max())
        lo52 = float(c.min())
        pos52 = (price_now - lo52) / (hi52 - lo52 + 1e-8)  # 0=52w low, 1=52w high

        # ── ATR (Average True Range) ──────────────────────────────────────────
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr14 / price_now * 100  # ATR as % of price

        # ── Consistency score: % of last 12 months weeks positive ────────────
        weekly = c.resample("W").last().pct_change().dropna()
        consistency = float((weekly > 0).mean() * 100)

        # ── MACD ─────────────────────────────────────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
        macd_now   = float(macd_line.iloc[-1])
        macd_cross = float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1])  # bullish cross

        return {
            "name":         name,
            "ticker":       ticker.replace(".NS", "").replace(".BO", ""),
            "sector":       _SECTOR_MAP.get(name, "Other"),
            "price":        price_now,
            "rsi":          rsi,
            "ret_1w":       ret_1w,
            "ret_1m":       ret_1m,
            "ret_3m":       ret_3m,
            "ret_6m":       ret_6m,
            "ret_1y":       ret_1y,
            "vol_21":       vol_21,
            "vol_ratio":    vol_ratio,
            "bb_pct":       bb_pct,
            "pos52":        pos52,
            "hi52":         hi52,
            "lo52":         lo52,
            "consistency":  consistency,
            "above_ema20":  above_ema20,
            "above_ema50":  above_ema50,
            "above_sma200": above_sma200,
            "ema_bullish":  ema_bullish,
            "macd_cross":   macd_cross,
            "atr_pct":      atr_pct,
        }
    except Exception:
        return None


def _short_term_score(s: dict) -> float:
    """
    Score 0–100 for short-term (1–4 week) potential.
    Weights: momentum (1w/1m), RSI sweet spot, volume surge, MACD cross, BB position.
    """
    score = 0.0

    # 1-week momentum (20 pts)
    score += min(max(s["ret_1w"] * 4, -20), 20)

    # 1-month momentum (15 pts)
    score += min(max(s["ret_1m"] * 1.5, -15), 15)

    # RSI sweet spot 45–65 for momentum, oversold bounce < 35 (15 pts)
    rsi = s["rsi"]
    if 45 <= rsi <= 65:
        score += 15
    elif rsi < 35:
        score += 12  # oversold bounce potential
    elif 35 <= rsi < 45:
        score += 6
    elif rsi > 75:
        score -= 10  # overbought risk

    # Volume surge > 1.5x average (15 pts)
    vr = s["vol_ratio"]
    if vr > 2.0:
        score += 15
    elif vr > 1.5:
        score += 10
    elif vr > 1.2:
        score += 5

    # MACD bullish cross (10 pts)
    if s["macd_cross"]:
        score += 10

    # Price above EMA20 + EMA50 (10 pts)
    score += (5 if s["above_ema20"] else -5)
    score += (5 if s["above_ema50"] else -3)

    # BB position — avoid extreme overbought (15 pts)
    bb = s["bb_pct"]
    if 0.3 <= bb <= 0.7:
        score += 15
    elif bb < 0.3:
        score += 8   # near lower band — potential bounce
    else:
        score -= 5   # near upper band — stretched

    return round(min(max(score, 0), 100), 1)


def _long_term_score(s: dict) -> float:
    """
    Score 0–100 for long-term (6–24 month) investment potential.
    Weights: annual return, trend structure, consistency, 52w position, low vol.
    """
    score = 0.0

    # 1-year return (25 pts)
    score += min(max(s["ret_1y"] * 0.5, -20), 25)

    # 6-month momentum (15 pts)
    score += min(max(s["ret_6m"] * 0.4, -10), 15)

    # 3-month trend (10 pts)
    score += min(max(s["ret_3m"] * 0.4, -8), 10)

    # Trend structure: above SMA200 (15 pts)
    if s["above_sma200"]:
        score += 10
    else:
        score -= 10  # bearish structure

    if s["ema_bullish"]:
        score += 5

    # Weekly consistency (15 pts)
    score += (s["consistency"] - 50) * 0.3  # neutral at 50%

    # 52-week position: prefer not at top (avoid buying peaks) (10 pts)
    pos = s["pos52"]
    if 0.30 <= pos <= 0.75:
        score += 10  # healthy range
    elif pos < 0.30:
        score += 5   # near 52w low — value zone
    else:
        score += 2   # near 52w high — momentum but stretched

    # Low volatility preferred for long term (10 pts)
    vol = s["vol_21"]
    if vol < 20:
        score += 10
    elif vol < 30:
        score += 5
    elif vol > 50:
        score -= 5

    return round(min(max(score, 0), 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Cached screener
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def run_screener() -> list[dict]:
    """Run full screener on all stocks. Cached for 5 minutes."""
    results = []
    for name, ticker in _SCREEN_STOCKS.items():
        sig = _compute_signals(name, ticker)
        if sig:
            sig["st_score"] = _short_term_score(sig)
            sig["lt_score"] = _long_term_score(sig)
            results.append(sig)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_bar(score: float, color: str) -> str:
    w = min(int(score), 100)
    return (
        f'<div style="background:#F3F4F6;border-radius:4px;height:6px;width:100%;margin-top:4px;">'
        f'<div style="background:{color};border-radius:4px;height:6px;width:{w}%;"></div>'
        f'</div>'
    )


def _grade(score: float) -> str:
    if score >= 75: return "A+"
    if score >= 65: return "A"
    if score >= 55: return "B+"
    if score >= 45: return "B"
    if score >= 35: return "C"
    return "D"


def _signal_badge(score: float) -> str:
    if score >= 70:
        return f'<span style="background:{_GREEN};color:white;border-radius:4px;padding:2px 8px;font-size:0.65rem;font-weight:700;">STRONG BUY</span>'
    if score >= 55:
        return f'<span style="background:#10B981;color:white;border-radius:4px;padding:2px 8px;font-size:0.65rem;font-weight:700;">BUY</span>'
    if score >= 40:
        return f'<span style="background:{_AMBER};color:white;border-radius:4px;padding:2px 8px;font-size:0.65rem;font-weight:700;">WATCH</span>'
    return f'<span style="background:#9CA3AF;color:white;border-radius:4px;padding:2px 8px;font-size:0.65rem;font-weight:700;">NEUTRAL</span>'


def _stock_card(s: dict, score: float, score_type: str) -> str:
    color      = _GREEN if score >= 55 else (_AMBER if score >= 40 else "#9CA3AF")
    ret_key    = "ret_1m" if score_type == "short" else "ret_1y"
    ret_label  = "1M Return" if score_type == "short" else "1Y Return"
    ret_val    = s[ret_key]
    ret_color  = _GREEN if ret_val > 0 else _RED
    badge      = _signal_badge(score)
    bar        = _score_bar(score, color)
    grade      = _grade(score)
    rsi_color  = _RED if s["rsi"] > 70 else (_GREEN if s["rsi"] < 35 else _NAVY)

    return f"""
    <div style="background:white;border-radius:12px;padding:14px 16px;margin-bottom:10px;
                box-shadow:0 1px 6px rgba(0,0,0,0.07);border-left:4px solid {color};">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div style="font-size:0.82rem;font-weight:700;color:{_NAVY};">{s['name']}</div>
                <div style="font-size:0.65rem;color:#9CA3AF;margin-top:1px;">
                    {s['ticker']} &nbsp;·&nbsp; {s['sector']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.90rem;font-weight:700;color:{_NAVY};">₹{s['price']:,.1f}</div>
                <div style="font-size:0.68rem;font-weight:700;color:{ret_color};">
                    {ret_val:+.1f}% {ret_label}
                </div>
            </div>
        </div>
        <div style="display:flex;gap:16px;margin-top:8px;font-size:0.70rem;color:#6B7280;">
            <span>RSI <b style="color:{rsi_color};">{s['rsi']:.0f}</b></span>
            <span>Vol×<b>{s['vol_ratio']:.1f}</b></span>
            <span>52w <b>{s['pos52']*100:.0f}%</b></span>
            <span>{'✅ >SMA200' if s['above_sma200'] else '⚠️ <SMA200'}</span>
        </div>
        <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;">
            <div style="flex:1;margin-right:12px;">
                <div style="font-size:0.65rem;color:#9CA3AF;">Score {score}/100 · Grade {grade}</div>
                {bar}
            </div>
            <div>{badge}</div>
        </div>
    </div>
    """


def _mini_chart(s: dict, ohlcv_data: pd.DataFrame | None) -> go.Figure:
    """Tiny sparkline chart for the stock."""
    fig = go.Figure()
    if ohlcv_data is not None and not ohlcv_data.empty:
        c = ohlcv_data["Close"].tail(60)
        color = _GREEN if float(c.iloc[-1]) >= float(c.iloc[0]) else _RED
        fig.add_trace(go.Scatter(
            x=list(range(len(c))), y=c.values,
            mode="lines", line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=f"{'rgba(0,208,156,0.08)' if color == _GREEN else 'rgba(255,83,112,0.08)'}",
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_insights_tab() -> None:
    st.markdown("""
    <style>
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="gw-section-title">💡 Smart Stock Picks — AI Screening Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.75rem;color:#9CA3AF;margin-bottom:16px;">'
        f'Screening {len(_SCREEN_STOCKS)} NSE stocks across momentum, trend, volume, RSI, MACD and consistency signals. '
        f'Refreshes every 5 minutes. &nbsp;·&nbsp; {datetime.now().strftime("%d %b %Y, %H:%M")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Sector filter ─────────────────────────────────────────────────────────
    all_sectors = sorted(set(_SECTOR_MAP.values()))
    sel_sectors = st.multiselect(
        "Filter by sector (leave blank for all)",
        options=all_sectors,
        default=[],
        key="insights_sectors",
    )
    top_n = st.slider("Show top N picks per category", 3, 10, 5, key="insights_topn")

    with st.spinner("Running screener across all stocks…"):
        all_stocks = run_screener()

    if not all_stocks:
        st.error("Could not fetch data for any stock. Check your connection.")
        return

    # Apply sector filter
    if sel_sectors:
        all_stocks = [s for s in all_stocks if s["sector"] in sel_sectors]

    if not all_stocks:
        st.warning("No stocks match the selected sectors.")
        return

    # Sort separately for ST and LT
    st_picks = sorted(all_stocks, key=lambda x: -x["st_score"])[:top_n]
    lt_picks = sorted(all_stocks, key=lambda x: -x["lt_score"])[:top_n]

    # ── Summary metrics ───────────────────────────────────────────────────────
    avg_st = np.mean([s["st_score"] for s in all_stocks])
    avg_lt = np.mean([s["lt_score"] for s in all_stocks])
    bullish_count  = sum(1 for s in all_stocks if s["above_sma200"])
    overbought     = sum(1 for s in all_stocks if s["rsi"] > 70)
    oversold       = sum(1 for s in all_stocks if s["rsi"] < 30)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, sub, color in [
        (m1, "Stocks Screened",   f"{len(all_stocks)}", f"{bullish_count} above SMA200", _GREEN),
        (m2, "Avg ST Score",      f"{avg_st:.0f}/100",  "Short-term market mood",        _PURPLE),
        (m3, "Avg LT Score",      f"{avg_lt:.0f}/100",  "Long-term trend health",        _PURPLE),
        (m4, "RSI Signals",       f"{overbought} / {oversold}", "Overbought / Oversold", _AMBER),
    ]:
        with col:
            st.markdown(f"""
            <div class="gw-stat-card">
                <div class="gw-stat-label">{label}</div>
                <div class="gw-stat-value" style="color:{color};">{val}</div>
                <div class="gw-stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin:20px 0;border-top:1px solid #E5E7EB;"></div>', unsafe_allow_html=True)

    # ── Two columns: Short Term | Long Term ───────────────────────────────────
    col_st, col_lt = st.columns(2)

    with col_st:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{_GREEN}18,{_GREEN}05);
             border:1px solid {_GREEN}44;border-radius:10px;padding:12px 16px;margin-bottom:14px;">
            <div style="font-size:0.65rem;color:{_GREEN};letter-spacing:1px;font-weight:700;">
                ⚡ SHORT-TERM PICKS &nbsp;·&nbsp; 1–4 WEEKS
            </div>
            <div style="font-size:0.75rem;color:#6B7280;margin-top:2px;">
                Based on momentum, RSI, volume surge and MACD signals
            </div>
        </div>
        """, unsafe_allow_html=True)

        cards_html = "".join(_stock_card(s, s["st_score"], "short") for s in st_picks)
        st.markdown(cards_html, unsafe_allow_html=True)

    with col_lt:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{_PURPLE}18,{_PURPLE}05);
             border:1px solid {_PURPLE}44;border-radius:10px;padding:12px 16px;margin-bottom:14px;">
            <div style="font-size:0.65rem;color:{_PURPLE};letter-spacing:1px;font-weight:700;">
                🏦 LONG-TERM PICKS &nbsp;·&nbsp; 6–24 MONTHS
            </div>
            <div style="font-size:0.75rem;color:#6B7280;margin-top:2px;">
                Based on annual return, SMA200 structure, consistency and volatility
            </div>
        </div>
        """, unsafe_allow_html=True)

        cards_html = "".join(_stock_card(s, s["lt_score"], "long") for s in lt_picks)
        st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div style="margin:20px 0;border-top:1px solid #E5E7EB;"></div>', unsafe_allow_html=True)

    # ── Full ranking table ────────────────────────────────────────────────────
    st.markdown('<div class="gw-section-title">📊 Full Stock Ranking</div>', unsafe_allow_html=True)

    sort_by = st.radio(
        "Sort by", ["Short-Term Score", "Long-Term Score", "1Y Return %", "RSI"],
        horizontal=True, key="insights_sort",
    )
    sort_map = {
        "Short-Term Score": ("st_score",  True),
        "Long-Term Score":  ("lt_score",  True),
        "1Y Return %":      ("ret_1y",    True),
        "RSI":              ("rsi",       False),
    }
    sk, asc = sort_map[sort_by]
    sorted_stocks = sorted(all_stocks, key=lambda x: x[sk], reverse=asc)

    medals = ["🥇", "🥈", "🥉"] + [""] * 100
    table_rows = []
    for i, s in enumerate(sorted_stocks):
        st_grade = _grade(s["st_score"])
        lt_grade = _grade(s["lt_score"])
        table_rows.append({
            "#":             f"{medals[i] if i < 3 else ''} {i+1}",
            "Stock":         s["name"],
            "Sector":        s["sector"],
            "Price (₹)":     f"₹{s['price']:,.1f}",
            "1W %":          f"{s['ret_1w']:+.1f}%",
            "1M %":          f"{s['ret_1m']:+.1f}%",
            "1Y %":          f"{s['ret_1y']:+.1f}%",
            "RSI":           f"{s['rsi']:.0f}",
            "ST Score":      f"{s['st_score']} ({st_grade})",
            "LT Score":      f"{s['lt_score']} ({lt_grade})",
            "ST Signal":     "STRONG BUY" if s["st_score"] >= 70 else ("BUY" if s["st_score"] >= 55 else ("WATCH" if s["st_score"] >= 40 else "NEUTRAL")),
            "LT Signal":     "STRONG BUY" if s["lt_score"] >= 70 else ("BUY" if s["lt_score"] >= 55 else ("WATCH" if s["lt_score"] >= 40 else "NEUTRAL")),
            ">SMA200":       "✅" if s["above_sma200"] else "❌",
        })

    df_table = pd.DataFrame(table_rows).set_index("#")
    st.dataframe(df_table, use_container_width=True, height=420)

    # ── Sector heatmap of scores ──────────────────────────────────────────────
    st.markdown('<div style="margin:20px 0 8px;border-top:1px solid #E5E7EB;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="gw-section-title">🗺️ Sector Opportunity Map</div>', unsafe_allow_html=True)

    sector_df = pd.DataFrame(all_stocks)
    sector_agg = sector_df.groupby("sector").agg(
        st_avg=("st_score", "mean"),
        lt_avg=("lt_score", "mean"),
        count=("name", "count"),
        ret_1y=("ret_1y", "mean"),
    ).reset_index()

    col_hm1, col_hm2 = st.columns(2)

    with col_hm1:
        fig_st = go.Figure(go.Bar(
            x=sector_agg.sort_values("st_avg", ascending=True)["st_avg"],
            y=sector_agg.sort_values("st_avg", ascending=True)["sector"],
            orientation="h",
            marker=dict(
                color=sector_agg.sort_values("st_avg", ascending=True)["st_avg"],
                colorscale=[[0, "#FFE4E6"], [0.5, _AMBER], [1, _GREEN]],
                cmin=0, cmax=100,
            ),
            text=[f"{v:.0f}" for v in sector_agg.sort_values("st_avg", ascending=True)["st_avg"]],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_st.update_layout(
            height=300, title=dict(text="<b>Short-Term Score by Sector</b>",
                                   font=dict(size=12, color=_NAVY), x=0.01),
            paper_bgcolor="white", plot_bgcolor=_GRAY,
            margin=dict(l=6, r=40, t=36, b=6),
            xaxis=dict(range=[0, 105], showgrid=False),
            yaxis=dict(showgrid=False),
            font=dict(family="Inter", size=10),
        )
        st.plotly_chart(fig_st, use_container_width=True)

    with col_hm2:
        fig_lt = go.Figure(go.Bar(
            x=sector_agg.sort_values("lt_avg", ascending=True)["lt_avg"],
            y=sector_agg.sort_values("lt_avg", ascending=True)["sector"],
            orientation="h",
            marker=dict(
                color=sector_agg.sort_values("lt_avg", ascending=True)["lt_avg"],
                colorscale=[[0, "#FFE4E6"], [0.5, _AMBER], [1, _PURPLE]],
                cmin=0, cmax=100,
            ),
            text=[f"{v:.0f}" for v in sector_agg.sort_values("lt_avg", ascending=True)["lt_avg"]],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_lt.update_layout(
            height=300, title=dict(text="<b>Long-Term Score by Sector</b>",
                                   font=dict(size=12, color=_NAVY), x=0.01),
            paper_bgcolor="white", plot_bgcolor=_GRAY,
            margin=dict(l=6, r=40, t=36, b=6),
            xaxis=dict(range=[0, 105], showgrid=False),
            yaxis=dict(showgrid=False),
            font=dict(family="Inter", size=10),
        )
        st.plotly_chart(fig_lt, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#FFF9E6;border:1px solid #F59E0B;border-radius:8px;
                padding:10px 14px;margin-top:8px;font-size:0.72rem;color:#92400E;">
        <b>⚠️ Disclaimer:</b> These picks are generated by a quantitative screening algorithm
        using technical signals only. They do <b>not</b> constitute financial advice.
        Always do your own research (DYOR) and consult a SEBI-registered advisor before investing.
        Past performance of signals is not a guarantee of future returns.
    </div>
    """, unsafe_allow_html=True)
