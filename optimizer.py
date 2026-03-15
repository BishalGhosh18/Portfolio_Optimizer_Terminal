"""
Portfolio Optimization Engine — Markowitz Mean-Variance Framework.

Strategies implemented:
  1. Maximum Sharpe Ratio (MSR)
  2. Minimum Volatility (MinVol)
  3. Maximum Diversification
  4. Risk Parity (Equal Risk Contribution)
  5. Equal Weight (baseline)
  6. Efficient Frontier (Monte Carlo + SLSQP sweep)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk_engine import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    covariance_matrix,
    portfolio_returns,
    TRADING_DAYS,
    RISK_FREE_RATE,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _port_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray) -> tuple:
    """Return (annualised_return, annualised_vol, sharpe)."""
    w       = np.array(weights)
    ret     = np.dot(w, mean_returns) * TRADING_DAYS
    vol     = np.sqrt(w @ cov @ w)
    sharpe  = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _constraints_and_bounds(n: int, allow_short: bool = False, weight_cap: float = 0.40):
    bounds      = [(-weight_cap, weight_cap) if allow_short else (0.0, weight_cap)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


# ─────────────────────────────────────────────
# Optimisation routines
# ─────────────────────────────────────────────

def max_sharpe_weights(
    returns: pd.DataFrame,
    allow_short: bool = False,
    weight_cap: float = 0.40,
) -> np.ndarray:
    """Maximise Sharpe ratio (SLSQP)."""
    mean_r  = returns.mean().values
    cov     = covariance_matrix(returns).values
    n       = len(mean_r)
    bounds, cons = _constraints_and_bounds(n, allow_short, weight_cap)
    x0      = np.ones(n) / n

    def neg_sharpe(w):
        ret, vol, _ = _port_stats(w, mean_r, cov)
        return -(ret - RISK_FREE_RATE) / vol if vol > 0 else 1e10

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else x0


def min_volatility_weights(
    returns: pd.DataFrame,
    allow_short: bool = False,
    weight_cap: float = 0.40,
) -> np.ndarray:
    """Minimise portfolio volatility."""
    cov     = covariance_matrix(returns).values
    n       = cov.shape[0]
    bounds, cons = _constraints_and_bounds(n, allow_short, weight_cap)
    x0      = np.ones(n) / n

    def port_vol(w):
        return np.sqrt(w @ cov @ w)

    res = minimize(port_vol, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else x0


def risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Equal Risk Contribution (ERC / Risk Parity).
    Each asset contributes equally to total portfolio risk.
    """
    cov = covariance_matrix(returns).values
    n   = cov.shape[0]
    x0  = np.ones(n) / n

    def risk_budget_objective(w):
        w       = np.array(w)
        sigma   = np.sqrt(w @ cov @ w)
        mrc     = cov @ w / sigma          # marginal risk contribution
        rc      = w * mrc                  # risk contribution
        target  = sigma / n                # equal contribution target
        return np.sum((rc - target) ** 2)

    bounds = [(0.001, 1.0)] * n
    cons   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res    = minimize(risk_budget_objective, x0, method="SLSQP",
                      bounds=bounds, constraints=cons,
                      options={"ftol": 1e-14, "maxiter": 2000})
    w = res.x if res.success else x0
    return w / w.sum()


def max_diversification_weights(
    returns: pd.DataFrame,
    weight_cap: float = 0.40,
) -> np.ndarray:
    """
    Maximise Diversification Ratio = weighted avg vol / portfolio vol.
    """
    cov      = covariance_matrix(returns).values
    ind_vols = np.sqrt(np.diag(cov))
    n        = len(ind_vols)
    bounds, cons = _constraints_and_bounds(n, allow_short=False, weight_cap=weight_cap)
    x0       = np.ones(n) / n

    def neg_dr(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 1e10
        return -(w @ ind_vols) / port_vol

    res = minimize(neg_dr, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else x0


def equal_weight(returns: pd.DataFrame) -> np.ndarray:
    n = len(returns.columns)
    return np.ones(n) / n


# ─────────────────────────────────────────────
# Efficient Frontier
# ─────────────────────────────────────────────

def efficient_frontier_monte_carlo(
    returns: pd.DataFrame,
    n_portfolios: int = 5000,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Simulate random portfolios to map the efficient frontier.
    Returns a DataFrame with columns: Return, Volatility, Sharpe, Weights.
    """
    mean_r  = returns.mean().values * TRADING_DAYS
    cov     = covariance_matrix(returns).values
    n       = len(mean_r)
    records = []

    rng = np.random.default_rng(42)
    for _ in range(n_portfolios):
        if allow_short:
            raw = rng.uniform(-1, 1, n)
        else:
            raw = rng.dirichlet(np.ones(n))
        w   = raw / np.abs(raw).sum() if allow_short else raw
        vol = float(np.sqrt(w @ cov @ w))
        ret = float(w @ mean_r)
        sr  = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
        records.append({
            "Return":     round(ret * 100, 3),
            "Volatility": round(vol * 100, 3),
            "Sharpe":     round(sr, 4),
            "Weights":    w.tolist(),
        })

    return pd.DataFrame(records)


def efficient_frontier_sweep(
    returns: pd.DataFrame,
    n_points: int = 60,
    weight_cap: float = 0.40,
) -> pd.DataFrame:
    """
    Compute the true efficient frontier by minimising vol for a range of target returns.
    """
    mean_r = returns.mean().values * TRADING_DAYS
    cov    = covariance_matrix(returns).values
    n      = len(mean_r)
    bounds = [(0.0, weight_cap)] * n

    min_ret = mean_r.min()
    max_ret = mean_r.max()
    targets = np.linspace(min_ret + 1e-5, max_ret - 1e-5, n_points)
    records = []

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_r - t},
        ]
        res = minimize(
            lambda w: np.sqrt(w @ cov @ w),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            w   = res.x
            vol = float(np.sqrt(w @ cov @ w))
            ret = float(w @ mean_r)
            sr  = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
            records.append({
                "Return":     round(ret * 100, 3),
                "Volatility": round(vol * 100, 3),
                "Sharpe":     round(sr, 4),
                "Weights":    w.tolist(),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Strategy dispatcher
# ─────────────────────────────────────────────

STRATEGIES = {
    "Maximum Sharpe":         max_sharpe_weights,
    "Minimum Volatility":     min_volatility_weights,
    "Risk Parity":            risk_parity_weights,
    "Maximum Diversification":max_diversification_weights,
    "Equal Weight":           equal_weight,
}


def run_strategy(strategy_name: str, returns: pd.DataFrame, **kwargs) -> np.ndarray:
    fn = STRATEGIES.get(strategy_name)
    if fn is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    try:
        return fn(returns, **kwargs)
    except TypeError:
        return fn(returns)


def all_strategies_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run all strategies and return a comparison table with key metrics.
    """
    mean_r  = returns.mean().values * TRADING_DAYS
    cov     = covariance_matrix(returns).values
    rows    = []

    for name in STRATEGIES:
        w   = run_strategy(name, returns)
        ret = float(w @ mean_r)
        vol = float(np.sqrt(w @ cov @ w))
        sr  = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
        rows.append({
            "Strategy":           name,
            "Ann. Return (%)":    round(ret * 100, 2),
            "Ann. Volatility (%)":round(vol * 100, 2),
            "Sharpe Ratio":       round(sr, 3),
        })

    return pd.DataFrame(rows).set_index("Strategy")
