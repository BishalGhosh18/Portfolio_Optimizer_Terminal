"""
Risk Analysis Engine for Tata Portfolio Optimizer.

Metrics computed:
  - Annualised Return & Volatility
  - Sharpe / Sortino / Calmar Ratios
  - Value-at-Risk (Historical, Parametric, CVaR/ES)
  - Maximum Drawdown
  - Beta vs Benchmark
  - Correlation & Covariance matrices
  - Rolling Volatility
  - Individual stock risk scorecard
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from data_fetcher import RISK_FREE_RATE

TRADING_DAYS = 252   # Indian markets


# ─────────────────────────────────────────────
# 1. Basic return / vol statistics
# ─────────────────────────────────────────────

def annualised_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Compound annualised return from daily returns."""
    return (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1


def annualised_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Annualised standard deviation from daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series | pd.DataFrame, rf: float = RISK_FREE_RATE) -> float | pd.Series:
    """Sharpe ratio: excess return per unit of total risk."""
    ann_ret  = annualised_return(returns)
    ann_vol  = annualised_volatility(returns)
    return (ann_ret - rf) / ann_vol


def sortino_ratio(returns: pd.Series | pd.DataFrame, rf: float = RISK_FREE_RATE) -> float | pd.Series:
    """Sortino ratio: excess return per unit of downside risk."""
    ann_ret    = annualised_return(returns)
    daily_rf   = (1 + rf) ** (1 / TRADING_DAYS) - 1
    downside   = returns[returns < daily_rf]
    down_std   = downside.std() * np.sqrt(TRADING_DAYS)
    if isinstance(down_std, pd.Series):
        down_std = down_std.replace(0, np.nan)
    elif down_std == 0:
        down_std = np.nan
    return (ann_ret - rf) / down_std


def calmar_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    ann_ret = annualised_return(returns)
    mdd     = max_drawdown(returns)
    return ann_ret / abs(mdd) if mdd != 0 else np.nan


# ─────────────────────────────────────────────
# 2. Drawdown analysis
# ─────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown time series from daily returns."""
    wealth   = (1 + returns).cumprod()
    peak     = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from daily returns."""
    return drawdown_series(returns).min()


# ─────────────────────────────────────────────
# 3. Value-at-Risk & CVaR
# ─────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) VaR at given confidence level."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    mu, sigma = returns.mean(), returns.std()
    return float(norm.ppf(1 - confidence, mu, sigma))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
    var   = historical_var(returns, confidence)
    tail  = returns[returns <= var]
    return float(tail.mean()) if not tail.empty else var


def var_summary(returns: pd.Series, confidence: float = 0.95, holding_days: int = 1) -> dict:
    """Full VaR/CVaR summary for a single return series."""
    scale    = np.sqrt(holding_days)
    hist_var = historical_var(returns, confidence) * scale
    para_var = parametric_var(returns, confidence) * scale
    es       = cvar(returns, confidence) * scale
    return {
        "Historical VaR":  round(hist_var * 100, 3),
        "Parametric VaR":  round(para_var * 100, 3),
        "CVaR (ES)":       round(es * 100, 3),
        "Confidence":      confidence,
        "Holding Period":  holding_days,
    }


# ─────────────────────────────────────────────
# 4. Beta & correlation
# ─────────────────────────────────────────────

def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """CAPM beta of a stock vs benchmark."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    cov_mat = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return cov_mat[0, 1] / cov_mat[1, 1]


def alpha(stock_returns: pd.Series, benchmark_returns: pd.Series,
          rf: float = RISK_FREE_RATE) -> float:
    """Jensen's alpha."""
    b        = beta(stock_returns, benchmark_returns)
    ann_ret  = annualised_return(stock_returns)
    bench_r  = annualised_return(benchmark_returns.reindex(stock_returns.index).dropna())
    return ann_ret - (rf + b * (bench_r - rf))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of returns."""
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance matrix."""
    return returns.cov() * TRADING_DAYS


# ─────────────────────────────────────────────
# 5. Rolling metrics
# ─────────────────────────────────────────────

def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.Series, window: int = 63,
                   rf: float = RISK_FREE_RATE) -> pd.Series:
    """Rolling Sharpe ratio (63-day ≈ quarterly)."""
    roll_ret = returns.rolling(window).apply(
        lambda x: annualised_return(pd.Series(x)), raw=False
    )
    roll_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (roll_ret - rf) / roll_vol


# ─────────────────────────────────────────────
# 6. Individual stock risk scorecard
# ─────────────────────────────────────────────

def risk_scorecard(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Build a comprehensive risk scorecard for each stock.
    Returns a DataFrame with one row per stock.
    """
    records = []
    for col in returns.columns:
        s   = returns[col].dropna()
        if s.empty or len(s) < 10:
            continue
        ann_r   = annualised_return(s)
        ann_v   = annualised_volatility(s)
        sh      = sharpe_ratio(s)
        so      = sortino_ratio(s)
        ca      = calmar_ratio(s)
        mdd     = max_drawdown(s)
        b       = beta(s, benchmark_returns)
        vs      = var_summary(s, confidence=0.95)
        skew    = float(s.skew())
        kurt    = float(s.kurtosis())

        # Risk score (0–100) — higher = riskier
        # Weighted combination of normalised metrics
        vol_score   = min(ann_v * 100, 100)          # raw vol %
        var_score   = min(abs(vs["Historical VaR"]) * 2, 100)
        dd_score    = min(abs(mdd) * 100, 100)
        beta_score  = min(abs(b) * 25, 100) if not np.isnan(b) else 50
        risk_score  = 0.35 * vol_score + 0.25 * var_score + 0.25 * dd_score + 0.15 * beta_score

        records.append({
            "Ticker":             col,
            "Ann. Return (%)":    round(ann_r * 100, 2),
            "Ann. Volatility (%)":round(ann_v * 100, 2),
            "Sharpe Ratio":       round(sh, 3),
            "Sortino Ratio":      round(so, 3) if not np.isnan(so) else None,
            "Calmar Ratio":       round(ca, 3) if not np.isnan(ca) else None,
            "Max Drawdown (%)":   round(mdd * 100, 2),
            "Beta":               round(b, 3) if not np.isnan(b) else None,
            "Hist VaR 95% (%)":   vs["Historical VaR"],
            "CVaR 95% (%)":       vs["CVaR (ES)"],
            "Skewness":           round(skew, 3),
            "Kurtosis":           round(kurt, 3),
            "Risk Score":         round(risk_score, 1),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["Risk Level"] = pd.cut(
            df["Risk Score"],
            bins=[0, 30, 55, 75, 100],
            labels=["Low", "Moderate", "High", "Very High"],
        )
    return df


# ─────────────────────────────────────────────
# 7. Portfolio-level risk
# ─────────────────────────────────────────────

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute weighted portfolio returns."""
    aligned = returns.dropna()
    w       = np.array(weights)
    return aligned.dot(w)


def portfolio_risk_summary(
    returns: pd.DataFrame,
    weights: np.ndarray,
    benchmark_returns: pd.Series,
) -> dict:
    """Full risk summary for a weighted portfolio."""
    port_ret = portfolio_returns(returns, weights)
    ann_r    = annualised_return(port_ret)
    ann_v    = annualised_volatility(port_ret)
    sh       = sharpe_ratio(port_ret)
    so       = sortino_ratio(port_ret)
    ca       = calmar_ratio(port_ret)
    mdd      = max_drawdown(port_ret)
    b        = beta(port_ret, benchmark_returns)
    vs       = var_summary(port_ret)
    return {
        "Ann. Return (%)":    round(ann_r * 100, 2),
        "Ann. Volatility (%)":round(ann_v * 100, 2),
        "Sharpe Ratio":       round(sh, 3),
        "Sortino Ratio":      round(so, 3) if not np.isnan(so) else None,
        "Calmar Ratio":       round(ca, 3) if not np.isnan(ca) else None,
        "Max Drawdown (%)":   round(mdd * 100, 2),
        "Beta":               round(b, 3) if not np.isnan(b) else None,
        **vs,
    }
