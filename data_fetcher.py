"""
Data fetcher — supports the full NSE/BSE universe + Tata presets.
Any stock searchable by name or ticker on NSE (.NS) or BSE (.BO).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Risk-free rate ─────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.065   # 6.5% Indian 10-yr G-Sec

# ── Benchmark tickers ──────────────────────────────────────────────────────────
NIFTY50_TICKER = "^NSEI"
SENSEX_TICKER  = "^BSESN"

# ══════════════════════════════════════════════════════════════════════════════
# Full NSE stock universe  (name → NSE ticker)
# Covers Nifty 500 + other popular counters
# ══════════════════════════════════════════════════════════════════════════════
NSE_STOCKS = {
    # ── Tata Group ─────────────────────────────────────────────────────────
    "Tata Consultancy Services":  "TCS.NS",
    "Tata Motors":                "TATAMOTORS.NS",
    "Tata Steel":                 "TATASTEEL.NS",
    "Tata Power":                 "TATAPOWER.NS",
    "Tata Consumer Products":     "TATACONSUM.NS",
    "Titan Company":              "TITAN.NS",
    "Tata Chemicals":             "TATACHEM.NS",
    "Indian Hotels (IHCL)":       "INDHOTEL.NS",
    "Tata Communications":        "TATACOMM.NS",
    "Tata Elxsi":                 "TATAELXSI.NS",
    "Trent":                      "TRENT.NS",
    "Voltas":                     "VOLTAS.NS",
    "Tata Investment Corp":       "TATAINVEST.NS",
    # ── Reliance & Energy ──────────────────────────────────────────────────
    "Reliance Industries":        "RELIANCE.NS",
    "ONGC":                       "ONGC.NS",
    "Coal India":                 "COALINDIA.NS",
    "NTPC":                       "NTPC.NS",
    "Power Grid Corp":            "POWERGRID.NS",
    "Adani Enterprises":          "ADANIENT.NS",
    "Adani Ports":                "ADANIPORTS.NS",
    "Adani Power":                "ADANIPOWER.NS",
    "Adani Green Energy":         "ADANIGREEN.NS",
    "Adani Total Gas":            "ATGL.NS",
    "Vedanta":                    "VEDL.NS",
    "Hindalco":                   "HINDALCO.NS",
    "JSW Steel":                  "JSWSTEEL.NS",
    "SAIL":                       "SAIL.NS",
    # ── Banking & Finance ──────────────────────────────────────────────────
    "HDFC Bank":                  "HDFCBANK.NS",
    "ICICI Bank":                 "ICICIBANK.NS",
    "State Bank of India":        "SBIN.NS",
    "Axis Bank":                  "AXISBANK.NS",
    "Kotak Mahindra Bank":        "KOTAKBANK.NS",
    "IndusInd Bank":              "INDUSINDBK.NS",
    "Bajaj Finance":              "BAJFINANCE.NS",
    "Bajaj Finserv":              "BAJAJFINSV.NS",
    "HDFC Life Insurance":        "HDFCLIFE.NS",
    "SBI Life Insurance":         "SBILIFE.NS",
    "ICICI Prudential Life":      "ICICIPRULI.NS",
    "Shriram Finance":            "SHRIRAMFIN.NS",
    "Muthoot Finance":            "MUTHOOTFIN.NS",
    "LIC Housing Finance":        "LICHSGFIN.NS",
    "PFC":                        "PFC.NS",
    "REC Limited":                "RECLTD.NS",
    "Yes Bank":                   "YESBANK.NS",
    "Federal Bank":               "FEDERALBNK.NS",
    "Bandhan Bank":               "BANDHANBNK.NS",
    # ── IT & Technology ────────────────────────────────────────────────────
    "Infosys":                    "INFY.NS",
    "Wipro":                      "WIPRO.NS",
    "HCL Technologies":           "HCLTECH.NS",
    "Tech Mahindra":              "TECHM.NS",
    "LTIMindtree":                "LTIM.NS",
    "L&T Technology Services":    "LTTS.NS",
    "Mphasis":                    "MPHASIS.NS",
    "Persistent Systems":         "PERSISTENT.NS",
    "Coforge":                    "COFORGE.NS",
    "Zensar Technologies":        "ZENSARTECH.NS",
    # ── FMCG & Consumer ────────────────────────────────────────────────────
    "Hindustan Unilever":         "HINDUNILVR.NS",
    "ITC":                        "ITC.NS",
    "Nestle India":               "NESTLEIND.NS",
    "Britannia Industries":       "BRITANNIA.NS",
    "Dabur India":                "DABUR.NS",
    "Marico":                     "MARICO.NS",
    "Godrej Consumer":            "GODREJCP.NS",
    "Colgate-Palmolive India":    "COLPAL.NS",
    "Emami":                      "EMAMILTD.NS",
    "United Spirits":             "MCDOWELL-N.NS",
    # ── Pharma & Healthcare ────────────────────────────────────────────────
    "Sun Pharmaceutical":         "SUNPHARMA.NS",
    "Dr Reddy's Laboratories":    "DRREDDY.NS",
    "Cipla":                      "CIPLA.NS",
    "Divi's Laboratories":        "DIVISLAB.NS",
    "Aurobindo Pharma":           "AUROPHARMA.NS",
    "Lupin":                      "LUPIN.NS",
    "Biocon":                     "BIOCON.NS",
    "Apollo Hospitals":           "APOLLOHOSP.NS",
    "Max Healthcare":             "MAXHEALTH.NS",
    "Fortis Healthcare":          "FORTIS.NS",
    # ── Automobile ─────────────────────────────────────────────────────────
    "Maruti Suzuki":              "MARUTI.NS",
    "Hero MotoCorp":              "HEROMOTOCO.NS",
    "Bajaj Auto":                 "BAJAJ-AUTO.NS",
    "Eicher Motors":              "EICHERMOT.NS",
    "TVS Motor":                  "TVSMOTOR.NS",
    "Mahindra & Mahindra":        "M&M.NS",
    "Ashok Leyland":              "ASHOKLEY.NS",
    "Bosch":                      "BOSCHLTD.NS",
    "Motherson Sumi":             "MOTHERSON.NS",
    # ── Infrastructure & Capital Goods ────────────────────────────────────
    "Larsen & Toubro":            "LT.NS",
    "Siemens India":              "SIEMENS.NS",
    "ABB India":                  "ABB.NS",
    "Havells India":              "HAVELLS.NS",
    "Bharat Electronics":         "BEL.NS",
    "HAL":                        "HAL.NS",
    "Bharat Forge":               "BHARATFORG.NS",
    "Thermax":                    "THERMAX.NS",
    "Cummins India":              "CUMMINSIND.NS",
    # ── Telecom & Media ────────────────────────────────────────────────────
    "Bharti Airtel":              "BHARTIARTL.NS",
    "Vodafone Idea":              "IDEA.NS",
    "Indus Towers":               "INDUSTOWER.NS",
    "Zee Entertainment":          "ZEEL.NS",
    "Sun TV Network":             "SUNTV.NS",
    # ── Real Estate ────────────────────────────────────────────────────────
    "DLF":                        "DLF.NS",
    "Godrej Properties":          "GODREJPROP.NS",
    "Oberoi Realty":              "OBEROIRLTY.NS",
    "Prestige Estates":           "PRESTIGE.NS",
    "Macrotech (Lodha)":          "LODHA.NS",
    # ── Cement & Materials ────────────────────────────────────────────────
    "UltraTech Cement":           "ULTRACEMCO.NS",
    "Shree Cement":               "SHREECEM.NS",
    "ACC":                        "ACC.NS",
    "Ambuja Cements":             "AMBUJACEM.NS",
    "Grasim Industries":          "GRASIM.NS",
    "Asian Paints":               "ASIANPAINT.NS",
    "Berger Paints":              "BERGEPAINT.NS",
    "Pidilite Industries":        "PIDILITIND.NS",
    # ── Retail & E-commerce ────────────────────────────────────────────────
    "Avenue Supermarts (DMart)":  "DMART.NS",
    "Nykaa (FSN E-Commerce)":     "NYKAA.NS",
    "Zomato":                     "ZOMATO.NS",
    "Paytm (One97)":              "PAYTM.NS",
    "PB Fintech (PolicyBazaar)":  "POLICYBZR.NS",
    "Devyani International":      "DEVYANI.NS",
    # ── Others ────────────────────────────────────────────────────────────
    "ITC Hotels":                 "ITCHOTELS.NS",
    "Container Corp":             "CONCOR.NS",
    "Interglobe Aviation (IndiGo)":"INDIGO.NS",
    "SpiceJet":                   "SPICEJET.NS",
    "UPL":                        "UPL.NS",
    "PI Industries":              "PIIND.NS",
    "Dalmia Bharat":              "DALBHARAT.NS",
    "Hindustan Aeronautics (HAL)":"HAL.NS",
    "Dixon Technologies":         "DIXONTECH.NS",
    "Astral":                     "ASTRAL.NS",
    "Alkem Laboratories":         "ALKEM.NS",
    "Torrent Pharma":             "TORNTPHARM.NS",
    "IPCA Laboratories":          "IPCALAB.NS",
    "Crompton Greaves Consumer":  "CROMPTON.NS",
    "Blue Star":                  "BLUESTARCO.NS",
    "Page Industries":            "PAGEIND.NS",
    "Jubilant FoodWorks":         "JUBLFOOD.NS",
    "Westlife Foodworld":         "WESTLIFE.NS",
    "Info Edge (Naukri)":         "NAUKRI.NS",
    "Just Dial":                  "JUSTDIAL.NS",
    "IRCTC":                      "IRCTC.NS",
    "IRFC":                       "IRFC.NS",
    "Indian Railway Finance Corp":"IRFC.NS",
}

# ══════════════════════════════════════════════════════════════════════════════
# BSE stock universe  (name → BSE ticker with .BO suffix)
# ══════════════════════════════════════════════════════════════════════════════
BSE_STOCKS = {name: ticker.replace(".NS", ".BO")
              for name, ticker in NSE_STOCKS.items()}

# ── Tata preset  (for backward compatibility) ──────────────────────────────────
TATA_COMPANIES = [
    "Tata Consultancy Services", "Tata Motors", "Tata Steel",
    "Tata Power", "Tata Consumer Products", "Titan Company",
    "Tata Chemicals", "Indian Hotels (IHCL)", "Tata Communications",
    "Tata Elxsi", "Trent", "Voltas", "Tata Investment Corp",
]

# For legacy compatibility
TATA_STOCKS = {
    name: (NSE_STOCKS[name], BSE_STOCKS[name])
    for name in TATA_COMPANIES
    if name in NSE_STOCKS
}

SECTOR_MAP = {
    "Tata Consultancy Services": "IT",
    "Tata Motors":               "Automobile",
    "Tata Steel":                "Metal & Mining",
    "Tata Power":                "Energy",
    "Tata Consumer Products":    "FMCG",
    "Titan Company":             "Consumer Goods",
    "Tata Chemicals":            "Chemicals",
    "Indian Hotels (IHCL)":      "Hospitality",
    "Tata Communications":       "Telecom",
    "Tata Elxsi":                "IT Services",
    "Trent":                     "Retail",
    "Voltas":                    "Consumer Durables",
    "Tata Investment Corp":      "Financial Services",
    "Reliance Industries":       "Energy",
    "HDFC Bank":                 "Banking",
    "ICICI Bank":                "Banking",
    "State Bank of India":       "Banking",
    "Infosys":                   "IT",
    "Wipro":                     "IT",
    "HCL Technologies":          "IT",
    "Hindustan Unilever":        "FMCG",
    "ITC":                       "FMCG",
    "Sun Pharmaceutical":        "Pharma",
    "Maruti Suzuki":             "Automobile",
    "Larsen & Toubro":           "Infrastructure",
    "Bharti Airtel":             "Telecom",
    "Asian Paints":              "Chemicals",
    "Bajaj Finance":             "Finance",
    "Zomato":                    "Technology",
    "DLF":                       "Real Estate",
    "UltraTech Cement":          "Cement",
}

MARKET_CAP_TIER = {
    "Tata Consultancy Services": "Large Cap",
    "Tata Motors":               "Large Cap",
    "Tata Steel":                "Large Cap",
    "Tata Power":                "Mid Cap",
    "Tata Consumer Products":    "Large Cap",
    "Titan Company":             "Large Cap",
    "Tata Chemicals":            "Mid Cap",
    "Indian Hotels (IHCL)":      "Mid Cap",
    "Tata Communications":       "Mid Cap",
    "Tata Elxsi":                "Mid Cap",
    "Trent":                     "Large Cap",
    "Voltas":                    "Mid Cap",
    "Tata Investment Corp":      "Small Cap",
    "Reliance Industries":       "Large Cap",
    "HDFC Bank":                 "Large Cap",
    "ICICI Bank":                "Large Cap",
    "Infosys":                   "Large Cap",
}


def get_stock_universe(exchange: str = "NSE") -> dict[str, str]:
    """Return full {company_name: ticker} universe for an exchange."""
    return NSE_STOCKS if exchange.upper() == "NSE" else BSE_STOCKS


def get_ticker_list(exchange: str = "NSE") -> dict[str, str]:
    """Return Tata-only {company_name: ticker} for backward compatibility."""
    stocks = get_stock_universe(exchange)
    return {name: stocks[name] for name in TATA_COMPANIES if name in stocks}


# ── Price data ─────────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        if len(tickers) == 1:
            close.columns = tickers
    close.dropna(axis=1, how="all", inplace=True)
    close.dropna(axis=0, how="all", inplace=True)
    return close


def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return full OHLCV DataFrame for a single ticker (for charting/prediction)."""
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw
    except Exception:
        return pd.DataFrame()


def fetch_live_quote(ticker: str) -> dict:
    """Return latest live price + metadata using yfinance fast_info."""
    try:
        t    = yf.Ticker(ticker)
        info = t.fast_info

        last_price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)

        # Fallback to history if fast_info is empty
        if last_price is None:
            hist = t.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty:
                return {}
            closes     = hist["Close"].dropna()
            last_price = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_price

        last_price = float(last_price)
        prev_close = float(prev_close) if prev_close is not None else last_price
        change     = last_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "ticker":     ticker,
            "price":      round(last_price, 2),
            "prev_close": round(prev_close, 2),
            "change":     round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume":     getattr(info, "three_month_average_volume", None),
            "market_cap": getattr(info, "market_cap", None),
            "52w_high":   getattr(info, "year_high", None),
            "52w_low":    getattr(info, "year_low", None),
        }
    except Exception:
        return {}


def fetch_all_live_quotes(
    company_names: list[str],
    exchange: str = "NSE",
) -> dict:
    """Fetch live quotes for a list of company names. Returns {company_name: quote_dict}."""
    universe = get_stock_universe(exchange)
    result = {}
    for name in company_names:
        ticker = universe.get(name)
        if not ticker:
            continue
        q = fetch_live_quote(ticker)
        if q:
            q["company"]  = name
            q["sector"]   = SECTOR_MAP.get(name, "Other")
            q["cap_tier"] = MARKET_CAP_TIER.get(name, "—")
            result[name] = q
    return result


def fetch_benchmark(period: str = "1y") -> pd.DataFrame:
    """Download Nifty 50 and Sensex."""
    raw = yf.download(
        [NIFTY50_TICKER, SENSEX_TICKER],
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        bench = raw["Close"][[NIFTY50_TICKER, SENSEX_TICKER]].copy()
    else:
        bench = raw[["Close"]].copy()
        bench.columns = ["Nifty50"]
        bench["Sensex"] = bench["Nifty50"]
        return bench
    bench.columns = ["Nifty50", "Sensex"]
    return bench


def compute_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Compute returns from a price DataFrame."""
    if freq == "weekly":
        return prices.resample("W").last().pct_change().dropna()
    if freq == "monthly":
        return prices.resample("ME").last().pct_change().dropna()
    return prices.pct_change().dropna()


def get_company_info(ticker: str) -> dict:
    """Fetch fundamental info for a stock."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "longName":      info.get("longName", ticker),
            "sector":        info.get("sector", "N/A"),
            "industry":      info.get("industry", "N/A"),
            "marketCap":     info.get("marketCap"),
            "trailingPE":    info.get("trailingPE"),
            "forwardPE":     info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "beta":          info.get("beta"),
            "52WeekHigh":    info.get("fiftyTwoWeekHigh"),
            "52WeekLow":     info.get("fiftyTwoWeekLow"),
            "description":   info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {}
