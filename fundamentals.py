"""
Fundamentals, earnings & news layer for the Stock Movement Predictor
====================================================================
Prediction isn't just past prices — it's earnings surprises, guidance, analyst
views, valuation and news flow. This module supplies those signals in two tiers:

  • BACKTESTABLE  — earnings-cycle features that have real historical dates and
                    can therefore be aligned point-in-time and walk-forward
                    backtested inside the ML model (``earnings_features``).

  • LIVE OVERLAY  — a snapshot of today's context that we CANNOT get years of
                    free point-in-time history for, so it is shown & blended into
                    *today's* call only, never fed to the backtest:
                      - per-stock news sentiment       (yfinance .news + TextBlob)
                      - Screener.in fundamentals       (ROCE / ROE / growth / P·E)
                      - analyst view                    (yfinance target & rating)

Everything network-bound is cached and fails soft (returns empty / neutral) so
the predictor keeps working offline.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:                                     # pragma: no cover
    def _cache(*a, **k):                              # no-op fallback outside streamlit
        def deco(fn):
            return fn
        return deco

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"}


# ═════════════════════════════════════════════════════════════════════════════
# NIFTY 50 UNIVERSE  (name → NSE ticker) — the focus set; grow later
# ═════════════════════════════════════════════════════════════════════════════
NIFTY_50: dict[str, str] = {
    "Adani Enterprises":            "ADANIENT.NS",
    "Adani Ports & SEZ":            "ADANIPORTS.NS",
    "Apollo Hospitals":             "APOLLOHOSP.NS",
    "Asian Paints":                 "ASIANPAINT.NS",
    "Axis Bank":                    "AXISBANK.NS",
    "Bajaj Auto":                   "BAJAJ-AUTO.NS",
    "Bajaj Finance":                "BAJFINANCE.NS",
    "Bajaj Finserv":                "BAJAJFINSV.NS",
    "Bharat Electronics (BEL)":     "BEL.NS",
    "Bharat Petroleum (BPCL)":      "BPCL.NS",
    "Bharti Airtel":                "BHARTIARTL.NS",
    "Cipla":                        "CIPLA.NS",
    "Coal India":                   "COALINDIA.NS",
    "Dr Reddy's Laboratories":      "DRREDDY.NS",
    "Eicher Motors":                "EICHERMOT.NS",
    "Grasim Industries":            "GRASIM.NS",
    "HCL Technologies":             "HCLTECH.NS",
    "HDFC Bank":                    "HDFCBANK.NS",
    "HDFC Life Insurance":          "HDFCLIFE.NS",
    "Hero MotoCorp":                "HEROMOTOCO.NS",
    "Hindalco Industries":          "HINDALCO.NS",
    "Hindustan Unilever":           "HINDUNILVR.NS",
    "ICICI Bank":                   "ICICIBANK.NS",
    "IndusInd Bank":                "INDUSINDBK.NS",
    "Infosys":                      "INFY.NS",
    "ITC":                          "ITC.NS",
    "JSW Steel":                    "JSWSTEEL.NS",
    "Kotak Mahindra Bank":          "KOTAKBANK.NS",
    "Larsen & Toubro (L&T)":        "LT.NS",
    "Mahindra & Mahindra":          "M&M.NS",
    "Maruti Suzuki":                "MARUTI.NS",
    "Nestle India":                 "NESTLEIND.NS",
    "NTPC":                         "NTPC.NS",
    "Oil & Natural Gas (ONGC)":     "ONGC.NS",
    "Power Grid Corp":              "POWERGRID.NS",
    "Reliance Industries":          "RELIANCE.NS",
    "SBI Life Insurance":           "SBILIFE.NS",
    "Shriram Finance":              "SHRIRAMFIN.NS",
    "State Bank of India (SBI)":    "SBIN.NS",
    "Sun Pharmaceutical":           "SUNPHARMA.NS",
    "Tata Consultancy Services":    "TCS.NS",
    "Tata Consumer Products":       "TATACONSUM.NS",
    "Tata Motors":                  "TMCV.NS",
    "Tata Steel":                   "TATASTEEL.NS",
    "Tech Mahindra":                "TECHM.NS",
    "Titan Company":                "TITAN.NS",
    "Trent":                        "TRENT.NS",
    "UltraTech Cement":             "ULTRACEMCO.NS",
    "Wipro":                        "WIPRO.NS",
    "Eternal (Zomato)":             "ETERNAL.NS",
}


def screener_symbol(ticker: str) -> str:
    """RELIANCE.NS → RELIANCE ; handles .NS/.BO and & in symbols."""
    return re.sub(r"\.(NS|BO)$", "", ticker.strip().upper())


# ═════════════════════════════════════════════════════════════════════════════
# 1. EARNINGS  (backtestable — real historical dates)
# ═════════════════════════════════════════════════════════════════════════════
@_cache(ttl=60 * 60 * 6, show_spinner=False)
def get_earnings_calendar(ticker: str) -> pd.DataFrame:
    """Earnings dates + EPS estimate/reported + surprise %, newest first.
    Returns empty DataFrame on any failure (offline-safe)."""
    try:
        import yfinance as yf
        ed = yf.Ticker(ticker).get_earnings_dates(limit=24)
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)
        ed = ed.rename(columns={"Surprise(%)": "surprise_pct",
                                "Reported EPS": "reported_eps",
                                "EPS Estimate": "eps_estimate"})
        keep = [c for c in ["eps_estimate", "reported_eps", "surprise_pct"] if c in ed.columns]
        return ed[keep].sort_index(ascending=False)
    except Exception:
        return pd.DataFrame()


def earnings_features(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time earnings-cycle features aligned to ``index``.

    For each date we only use earnings that had ALREADY happened by then (no
    look-ahead), except ``days_to_earnings`` which uses the *scheduled* next
    date — knowable in advance, so it's fair game.

    Columns:
      • days_since_earnings  trading-day-ish distance since last report (0–60, capped)
      • days_to_earnings     distance to next scheduled report (0–60, capped)
      • earnings_window      1 if within ±3 calendar days of a report
      • last_surprise        most-recent past EPS surprise %, decayed by recency
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None)
    out = pd.DataFrame(index=idx, dtype=float)
    cal = get_earnings_calendar(ticker)

    if cal.empty:
        out["days_since_earnings"] = 90.0
        out["days_to_earnings"]    = 90.0
        out["earnings_window"]     = 0.0
        out["last_surprise"]       = 0.0
        return out

    dates    = np.sort(cal.index.values)                     # ascending datetimes
    surprise = cal["surprise_pct"].reindex(cal.index) if "surprise_pct" in cal else None

    ds, dt, win, surp = [], [], [], []
    for d in idx:
        past   = dates[dates <= np.datetime64(d)]
        future = dates[dates >  np.datetime64(d)]
        since  = (d - pd.Timestamp(past[-1])).days if len(past)   else 90
        to     = (pd.Timestamp(future[0]) - d).days if len(future) else 90
        ds.append(min(since, 90))
        dt.append(min(to, 90))
        win.append(1.0 if min(since, to) <= 3 else 0.0)

        s = 0.0
        if surprise is not None and len(past):
            val = surprise.get(pd.Timestamp(past[-1]))
            if val is not None and not pd.isna(val):
                s = float(val) * np.exp(-since / 45.0)        # decay stale surprises
        surp.append(s)

    out["days_since_earnings"] = ds
    out["days_to_earnings"]    = dt
    out["earnings_window"]     = win
    out["last_surprise"]       = surp
    return out


EARNINGS_FEATURE_NAMES = [
    "days_since_earnings", "days_to_earnings", "earnings_window", "last_surprise",
]


# ═════════════════════════════════════════════════════════════════════════════
# 1b. MARKET REGIME  (backtestable — Nifty 50 + India VIX, full free history)
# ═════════════════════════════════════════════════════════════════════════════
MARKET_FEATURE_NAMES = ["mkt_ret_1", "mkt_ret_5", "mkt_above_ma20", "vix_level", "vix_chg"]


@_cache(ttl=60 * 60, show_spinner=False)
def _index_history(symbol: str, period: str) -> pd.Series:
    try:
        import yfinance as yf
        raw = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.dropna()
    except Exception:
        return pd.Series(dtype=float)


def market_features(index: pd.DatetimeIndex, period: str = "5y") -> pd.DataFrame:
    """Broad-market regime features aligned to ``index`` (no look-ahead — each
    row uses the market close of that same day, exactly like the stock features).

    • mkt_ret_1 / mkt_ret_5   Nifty 50 1- & 5-day returns
    • mkt_above_ma20          1 if Nifty above its 20-day MA (risk-on)
    • vix_level               India VIX, de-meaned & scaled (fear gauge)
    • vix_chg                 daily VIX change
    """
    idx  = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None)
    out  = pd.DataFrame(index=idx, dtype=float)

    nifty = _index_history("^NSEI", period)
    vix   = _index_history("^INDIAVIX", period)

    if not nifty.empty:
        nret  = nifty.pct_change()
        ma20  = nifty.rolling(20).mean()
        out["mkt_ret_1"]      = nret.reindex(idx, method="ffill")
        out["mkt_ret_5"]      = nret.rolling(5).sum().reindex(idx, method="ffill")
        out["mkt_above_ma20"] = (nifty > ma20).astype(float).reindex(idx, method="ffill")
    else:
        out["mkt_ret_1"] = out["mkt_ret_5"] = out["mkt_above_ma20"] = 0.0

    if not vix.empty:
        vz = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-8)
        out["vix_level"] = vz.reindex(idx, method="ffill")
        out["vix_chg"]   = vix.pct_change().reindex(idx, method="ffill")
    else:
        out["vix_level"] = out["vix_chg"] = 0.0

    return out.fillna(0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. NEWS SENTIMENT  (live overlay — per stock)
# ═════════════════════════════════════════════════════════════════════════════
def _polarity(text: str) -> float:
    try:
        from textblob import TextBlob
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


@_cache(ttl=60 * 15, show_spinner=False)
def fetch_news_sentiment(ticker: str, max_items: int = 10) -> dict:
    """Per-stock news headlines + TextBlob sentiment.
    Returns {score:-1..1, label, items:[{title,source,time,polarity,color}]}."""
    items: list[dict] = []
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        for art in news[:max_items]:
            c = art.get("content", art) if isinstance(art, dict) else {}
            title = (c.get("title") or "").strip()
            if not title:
                continue
            summary = (c.get("summary") or c.get("description") or "")[:300]
            prov = c.get("provider") or {}
            source = prov.get("displayName", "") if isinstance(prov, dict) else ""
            when = c.get("pubDate") or c.get("displayTime") or ""
            try:
                dt = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
                ago = _time_ago(dt)
            except Exception:
                ago = ""
            pol = _polarity(f"{title}. {summary}")
            items.append({
                "title":  title[:160],
                "source": source or "News",
                "time":   ago,
                "polarity": pol,
                "color":  "#00D09C" if pol > 0.1 else "#FF5370" if pol < -0.1 else "#F59E0B",
            })
    except Exception:
        pass

    if not items:
        return {"score": 0.0, "label": "No recent news", "n": 0, "items": []}

    score = float(np.mean([i["polarity"] for i in items]))
    label = "Bullish" if score > 0.08 else "Bearish" if score < -0.08 else "Neutral"
    return {"score": score, "label": label, "n": len(items), "items": items}


def _time_ago(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 3600:   return f"{int(secs // 60)}m ago"
    if secs < 86400:  return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


# ═════════════════════════════════════════════════════════════════════════════
# 3. SCREENER.IN FUNDAMENTALS  (live overlay)
# ═════════════════════════════════════════════════════════════════════════════
def _num(text: str):
    """'₹ 22.7 %' / '17,69,785 Cr.' → float (crores stripped, % kept as number)."""
    if not text:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(m.group()) if m else None


@_cache(ttl=60 * 60 * 6, show_spinner=False)
def fetch_screener(ticker: str) -> dict:
    """Scrape the public Screener.in company page for headline ratios,
    growth (sales/profit) and a short 'pros/cons' read. Fails soft to {}."""
    sym = screener_symbol(ticker)
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return {}

    html = None
    for path in (f"{sym}/consolidated/", f"{sym}/"):
        try:
            r = requests.get(f"https://www.screener.in/company/{path}",
                             headers=_UA, timeout=10)
            if r.status_code == 200 and len(r.text) > 5000:
                html = r.text
                break
        except Exception:
            continue
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    ratios: dict[str, float] = {}
    for li in soup.select("#top-ratios li"):
        name = li.select_one(".name")
        val  = li.select_one(".value")
        if not name or not val:
            continue
        key = name.get_text(strip=True)
        num = _num(val.get_text(" ", strip=True))
        if num is not None:
            ratios[key] = num

    def _pick(*keys):
        for k in keys:
            for rk, rv in ratios.items():
                if k.lower() in rk.lower():
                    return rv
        return None

    pros = [li.get_text(" ", strip=True)
            for li in soup.select(".pros li, div.pros ul li")][:4]
    cons = [li.get_text(" ", strip=True)
            for li in soup.select(".cons li, div.cons ul li")][:4]

    return {
        "pe":         _pick("Stock P/E", "P/E"),
        "roce":       _pick("ROCE"),
        "roe":        _pick("ROE"),
        "div_yield":  _pick("Dividend Yield"),
        "book_value": _pick("Book Value"),
        "market_cap": _pick("Market Cap"),
        "price":      _pick("Current Price"),
        "pros":       [p for p in pros if p],
        "cons":       [c for c in cons if c],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. ANALYST / VALUATION SNAPSHOT  (live overlay — yfinance .info)
# ═════════════════════════════════════════════════════════════════════════════
@_cache(ttl=60 * 60 * 6, show_spinner=False)
def fetch_analyst_snapshot(ticker: str) -> dict:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    upside = ((target - price) / price * 100) if (price and target) else None
    return {
        "recommendation": (info.get("recommendationKey") or "").replace("_", " ").title() or None,
        "n_analysts":     info.get("numberOfAnalystOpinions"),
        "target_mean":    target,
        "target_upside":  upside,
        "trailing_pe":    info.get("trailingPE"),
        "forward_pe":     info.get("forwardPE"),
        "profit_margin":  (info.get("profitMargins") or 0) * 100 if info.get("profitMargins") else None,
        "sector":         info.get("sector"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. COMBINED CONTEXT + BIAS SCORE  (live overlay orchestrator)
# ═════════════════════════════════════════════════════════════════════════════
_RECO_SCORE = {
    "strong buy": 1.0, "buy": 0.6, "outperform": 0.5, "overweight": 0.5,
    "hold": 0.0, "neutral": 0.0,
    "underperform": -0.5, "underweight": -0.5, "sell": -0.6, "strong sell": -1.0,
}


def fundamental_context(ticker: str) -> dict:
    """Gather all live-overlay signals and reduce them to a single bias in
    [-1, +1] plus per-signal contributions (for transparency in the UI)."""
    news     = fetch_news_sentiment(ticker)
    scr      = fetch_screener(ticker)
    analyst  = fetch_analyst_snapshot(ticker)
    earn     = earnings_summary(ticker)

    contrib: dict[str, float] = {}

    # Recent earnings surprise (fades over ~45 days)
    if earn.get("last_surprise") is not None and earn.get("days_since") is not None \
            and earn["days_since"] <= 60:
        decay = float(np.exp(-earn["days_since"] / 45.0))
        contrib["Recent earnings"] = float(np.clip(earn["last_surprise"] / 15.0, -1, 1)) * decay

    # News sentiment  (−1..1) → scaled
    contrib["News sentiment"] = float(np.clip(news.get("score", 0.0) * 2.5, -1, 1))

    # Analyst recommendation
    reco = (analyst.get("recommendation") or "").lower()
    if reco in _RECO_SCORE:
        contrib["Analyst rating"] = _RECO_SCORE[reco]

    # Target-price upside (±20% → ±1)
    up = analyst.get("target_upside")
    if up is not None:
        contrib["Target upside"] = float(np.clip(up / 20.0, -1, 1))

    # Fundamental quality from Screener (ROE/ROCE high = good)
    quality = []
    for key, good in (("roe", 15), ("roce", 15)):
        v = scr.get(key)
        if v is not None:
            quality.append(np.clip((v - good) / good, -1, 1))
    if quality:
        contrib["Fundamentals (ROE/ROCE)"] = float(np.mean(quality))

    # Screener pros vs cons tilt
    n_pro, n_con = len(scr.get("pros", [])), len(scr.get("cons", []))
    if n_pro or n_con:
        contrib["Pros vs cons"] = float(np.clip((n_pro - n_con) / 4.0, -1, 1))

    bias  = float(np.mean(list(contrib.values()))) if contrib else 0.0
    label = ("Bullish" if bias > 0.15 else "Bearish" if bias < -0.15 else "Neutral")
    return {
        "bias":     bias,          # −1..+1
        "label":    label,
        "contrib":  contrib,       # per-signal, for display
        "news":     news,
        "screener": scr,
        "analyst":  analyst,
        "earnings": earn,
    }


def earnings_summary(ticker: str) -> dict:
    """Compact earnings read for the UI: next date, days to/since, last surprise."""
    cal = get_earnings_calendar(ticker)
    if cal.empty:
        return {}
    today  = pd.Timestamp.now().normalize()
    future = cal.index[cal.index > today]
    past   = cal.index[cal.index <= today]
    out: dict = {}
    if len(future):
        nd = future.min()
        out["next_date"] = nd
        out["days_to"]   = int((nd - today).days)
    if len(past):
        ld = past.max()
        out["last_date"]  = ld
        out["days_since"] = int((today - ld).days)
        if "surprise_pct" in cal.columns:
            v = cal.loc[ld, "surprise_pct"]
            out["last_surprise"] = None if pd.isna(v) else float(v)
    return out
