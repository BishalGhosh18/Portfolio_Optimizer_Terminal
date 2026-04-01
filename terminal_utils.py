"""
terminal_utils.py — Back-end helpers for the Terminal tab.
==========================================================
Provides:
  1. fetch_index_quotes()     — NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT
  2. simulate_order_book()    — realistic bid/ask depth around LTP
  3. fetch_news()             — NewsAPI → RSS fallback + TextBlob sentiment
  4. compute_strategy_signals()— EMA Crossover, BB Squeeze, RSI Divergence
  5. fetch_heatmap_data()     — live % change for top-20 NSE stocks

All market data goes through data_fetcher.py — no direct yfinance calls.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from data_fetcher import fetch_live_quote, fetch_ohlcv

# ── Index tickers ─────────────────────────────────────────────────────────────
_INDEX_TICKERS = {
    "NIFTY 50":   "^NSEI",
    "SENSEX":     "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT":   "^CNXIT",
}

# ── Top-20 NSE stocks for sector heatmap ─────────────────────────────────────
HEATMAP_STOCKS: dict[str, str] = {
    "HDFC Bank":           "HDFCBANK.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Reliance":            "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "Infosys":             "INFY.NS",
    "Wipro":               "WIPRO.NS",
    "SBI":                 "SBIN.NS",
    "Kotak Bank":          "KOTAKBANK.NS",
    "Axis Bank":           "AXISBANK.NS",
    "Bharti Airtel":       "BHARTIARTL.NS",
    "HUL":                 "HINDUNILVR.NS",
    "L&T":                 "LT.NS",
    "Maruti":              "MARUTI.NS",
    "ONGC":                "ONGC.NS",
    "NTPC":                "NTPC.NS",
    "Power Grid":          "POWERGRID.NS",
    "Tata Motors":         "TMCV.NS",
    "Sun Pharma":          "SUNPHARMA.NS",
    "Dr Reddy's":          "DRREDDY.NS",
    "ITC":                 "ITC.NS",
}

# ── RSS fallback feeds ────────────────────────────────────────────────────────
_RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Index quotes
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_index_quotes() -> dict[str, dict]:
    """Return live quotes for all four Indian market indices."""
    result: dict[str, dict] = {}
    for name, ticker in _INDEX_TICKERS.items():
        q = fetch_live_quote(ticker)
        result[name] = q if q else {
            "price": 0.0, "change": 0.0, "change_pct": 0.0,
            "52w_high": None, "52w_low": None,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Order book simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_order_book(
    ltp: float,
    seed: int = 42,
    n_levels: int = 5,
) -> dict:
    """
    Simulate a realistic n-level bid/ask order book around the given LTP.
    Returns:
      asks        — list of {price, qty, orders}, worst first (display top→LTP)
      bids        — list of {price, qty, orders}, best first (display LTP→worst)
      ltp         — the reference price
      spread_pct  — bid-ask spread as % of LTP
      total_volume— total qty across all levels
    """
    rng  = np.random.default_rng(seed)
    tick = max(round(ltp * 0.0005, 2), 0.05)   # ~0.05% tick size

    asks, bids = [], []
    ask_p = ltp + tick
    bid_p = ltp - tick

    for i in range(n_levels):
        scale     = 1 + i * 0.35
        ask_qty   = int(rng.integers(200, 2000) * scale)
        bids_qty  = int(rng.integers(200, 2000) * scale)
        asks.append({"price": round(ask_p, 2),
                     "qty":   ask_qty,
                     "orders": int(rng.integers(3, 25))})
        bids.append({"price": round(bid_p, 2),
                     "qty":   bids_qty,
                     "orders": int(rng.integers(3, 25))})
        ask_p += tick * float(rng.uniform(0.8, 1.6))
        bid_p -= tick * float(rng.uniform(0.8, 1.6))

    spread_pct = round((asks[0]["price"] - bids[0]["price"]) / ltp * 100, 4)
    total_vol  = sum(a["qty"] for a in asks) + sum(b["qty"] for b in bids)
    return {
        "asks":         asks,
        "bids":         bids,
        "ltp":          ltp,
        "spread_pct":   spread_pct,
        "total_volume": total_vol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. News & sentiment
# ─────────────────────────────────────────────────────────────────────────────

def _time_ago(dt: datetime) -> str:
    """Return a human-readable '4m ago' string."""
    try:
        now  = datetime.now(timezone.utc)
        diff = now - dt.astimezone(timezone.utc)
        sec  = int(diff.total_seconds())
        if sec < 60:    return f"{sec}s ago"
        if sec < 3600:  return f"{sec // 60}m ago"
        if sec < 86400: return f"{sec // 3600}h ago"
        return f"{sec // 86400}d ago"
    except Exception:
        return "—"


def _score_sentiment(text: str) -> tuple[str, str]:
    """Return (BULL|BEAR|NEUTRAL, hex_colour) via TextBlob polarity."""
    try:
        from textblob import TextBlob
        pol = TextBlob(text).sentiment.polarity
    except Exception:
        pol = 0.0
    if pol > 0.1:   return "BULL",    "#00D09C"
    if pol < -0.1:  return "BEAR",    "#FF5370"
    return "NEUTRAL", "#F59E0B"


def _rss_news(max_items: int = 12) -> list[dict]:
    """Fallback: parse ET Markets & Moneycontrol RSS."""
    try:
        import feedparser
    except ImportError:
        return []

    items: list[dict] = []
    per_feed = max(max_items // len(_RSS_FEEDS), 4)
    for url in _RSS_FEEDS:
        try:
            feed   = feedparser.parse(url)
            source = "ET Markets" if "economictimes" in url else "Moneycontrol"
            for entry in feed.entries[:per_feed + 2]:
                title   = entry.get("title", "") or ""
                summary = entry.get("summary", "") or title
                try:
                    pub = entry.get("published_parsed")
                    dt  = datetime(*pub[:6], tzinfo=timezone.utc) if pub else datetime.now(timezone.utc)
                except Exception:
                    dt  = datetime.now(timezone.utc)
                label, color = _score_sentiment(title + " " + summary)
                items.append({
                    "title":  title[:150],
                    "source": source,
                    "time":   _time_ago(dt),
                    "label":  label,
                    "color":  color,
                })
        except Exception:
            pass
    return items[:max_items]


def _newsapi_news(api_key: str, max_items: int = 12) -> list[dict]:
    """Fetch from NewsAPI.org."""
    try:
        import requests
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        "NSE OR NIFTY OR \"Indian stock market\"",
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": max_items,
                "apiKey":   api_key,
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        items: list[dict] = []
        for art in resp.json().get("articles", []):
            title  = (art.get("title")       or "").strip()[:150]
            desc   = (art.get("description") or "").strip()
            source = art.get("source", {}).get("name", "News")
            try:
                dt = datetime.fromisoformat(
                    (art.get("publishedAt") or "").replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
            label, color = _score_sentiment(title + " " + desc)
            items.append({
                "title":  title,
                "source": source,
                "time":   _time_ago(dt),
                "label":  label,
                "color":  color,
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_news(api_key: str = "") -> list[dict]:
    """Fetch financial news — NewsAPI first, RSS fallback."""
    if api_key and api_key not in ("", "your_newsapi_org_key"):
        items = _newsapi_news(api_key)
        if items:
            return items
    return _rss_news()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Strategy signal computation (pure pandas/numpy — no pandas-ta needed)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(c: pd.Series, period: int = 14) -> pd.Series:
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr     = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    """Return the ADX series (smoothed directional index)."""
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    dm_pos = (h - prev_h).clip(lower=0).where((h - prev_h) > (prev_l - l), 0)
    dm_neg = (prev_l - l).clip(lower=0).where((prev_l - l) > (h - prev_h), 0)
    atr_s  = _atr(h, l, c, period)
    di_pos = 100 * dm_pos.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()


def _win_prob(rsi: float, signal: str) -> float:
    if signal == "BUY":
        if rsi < 30: return 0.70
        if rsi < 45: return 0.62
        return 0.55
    else:
        if rsi > 70: return 0.68
        if rsi > 55: return 0.60
        return 0.53


def compute_strategy_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
) -> list[dict]:
    """
    Compute three algo signals on daily OHLCV data using pure pandas/numpy.

    Strategies:
      1. EMA Crossover  — EMA9 × EMA21 with ADX > 25
      2. BB Squeeze     — Bollinger Band width at 6-month low + breakout
      3. RSI Divergence — RSI < 30 oversold / > 70 overbought

    Returns list of signal dicts (empty list on error).
    """
    if ohlcv.empty or len(ohlcv) < 60:
        return []

    try:
        c = ohlcv["Close"].astype(float)
        h = ohlcv["High"].astype(float)
        l = ohlcv["Low"].astype(float)

        ltp   = float(c.iloc[-1])
        ema9  = _ema(c, 9)
        ema21 = _ema(c, 21)
        rsi_s = _rsi(c, 14)
        atr_s = _atr(h, l, c, 14)
        adx_s = _adx(h, l, c, 14)

        rsi_v = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
        atr_v = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else ltp * 0.015
        adx_v = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 20.0

        bb_ma  = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_up  = bb_ma + 2 * bb_std
        bb_dn  = bb_ma - 2 * bb_std

        short = ticker.replace(".NS", "").replace(".BO", "")

        def _make(name, icon, sig, entry, target, sl, tags):
            rew = abs(target - entry)
            rsk = abs(sl - entry) or 1e-8
            return {
                "name": name, "icon": icon, "ticker": short,
                "signal": sig, "rsi": round(rsi_v, 1),
                "entry": round(entry, 2), "target": round(target, 2),
                "stop_loss": round(sl, 2),
                "risk_reward": round(rew / rsk, 2),
                "win_prob": _win_prob(rsi_v, sig),
                "exp_return": round(rew / entry * 100, 2),
                "tags": tags,
            }

        signals: list[dict] = []

        # ── Strategy 1: EMA Crossover ──────────────────────────────────────
        e9_now,  e9_prev  = float(ema9.iloc[-1]),  float(ema9.iloc[-2])
        e21_now, e21_prev = float(ema21.iloc[-1]), float(ema21.iloc[-2])
        strong     = adx_v > 25
        cross_up   = e9_prev <= e21_prev and e9_now > e21_now
        cross_down = e9_prev >= e21_prev and e9_now < e21_now

        sig1   = "BUY" if e9_now >= e21_now else "SELL"
        tags1  = (["Bullish EMA"] if sig1 == "BUY" else ["Bearish EMA"]) + \
                 (["ADX > 25"] if strong else ["Weak trend"])
        if cross_up:   tags1.insert(0, "Fresh Cross ▲")
        if cross_down: tags1.insert(0, "Fresh Cross ▼")
        tgt1 = ltp + 2.5 * atr_v if sig1 == "BUY" else ltp - 2.5 * atr_v
        sl1  = ltp - 1.0 * atr_v if sig1 == "BUY" else ltp + 1.0 * atr_v
        signals.append(_make("EMA Crossover", "📈", sig1, ltp, tgt1, sl1, tags1))

        # ── Strategy 2: BB Squeeze ─────────────────────────────────────────
        width   = (bb_up - bb_dn) / bb_ma.replace(0, np.nan)
        w6m_min = width.rolling(min(126, len(width))).min()
        squeeze = (not pd.isna(w6m_min.iloc[-1])
                   and float(width.iloc[-1]) <= float(w6m_min.iloc[-1]) * 1.08)
        bbu_now = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else ltp
        bbl_now = float(bb_dn.iloc[-1]) if not pd.isna(bb_dn.iloc[-1]) else ltp

        if ltp > bbu_now:
            sig2 = "SELL"; tags2 = ["OB at upper BB"]
        elif ltp < bbl_now:
            sig2 = "BUY";  tags2 = ["OS at lower BB"]
        elif squeeze:
            sig2 = "BUY" if e9_now > e21_now else "SELL"
            tags2 = ["BB Squeeze — pending breakout"]
        else:
            sig2 = "BUY" if rsi_v < 50 else "SELL"
            tags2 = ["Inside BB range"]
        if squeeze: tags2.append("Squeeze active")
        tgt2 = ltp + 2.0 * atr_v if sig2 == "BUY" else ltp - 2.0 * atr_v
        sl2  = ltp - 1.2 * atr_v if sig2 == "BUY" else ltp + 1.2 * atr_v
        signals.append(_make("BB Squeeze", "🎯", sig2, ltp, tgt2, sl2, tags2))

        # ── Strategy 3: RSI Divergence ─────────────────────────────────────
        if rsi_v < 30:
            sig3 = "BUY";  tags3 = ["RSI Oversold < 30", "Mean reversion"]
        elif rsi_v > 70:
            sig3 = "SELL"; tags3 = ["RSI Overbought > 70", "Mean reversion"]
        elif rsi_v < 45:
            sig3 = "BUY";  tags3 = ["RSI approaching oversold"]
        else:
            sig3 = "SELL"; tags3 = ["RSI approaching overbought"]
        tgt3 = ltp + 2.0 * atr_v if sig3 == "BUY" else ltp - 2.0 * atr_v
        sl3  = ltp - 1.0 * atr_v if sig3 == "BUY" else ltp + 1.0 * atr_v
        signals.append(_make("RSI Divergence", "📊", sig3, ltp, tgt3, sl3, tags3))

        return signals

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sector heatmap
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_heatmap_data() -> list[dict]:
    """Return live % change for the top-20 NSE heatmap stocks."""
    rows: list[dict] = []
    for name, ticker in HEATMAP_STOCKS.items():
        q = fetch_live_quote(ticker)
        rows.append({
            "name":       name,
            "ticker":     ticker.replace(".NS", "").replace(".BO", ""),
            "change_pct": q.get("change_pct", 0.0) if q else 0.0,
            "price":      q.get("price",      0.0) if q else 0.0,
        })
    return rows
