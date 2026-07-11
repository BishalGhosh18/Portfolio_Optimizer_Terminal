"""
Long-term price forecast (1–4 months) for the Movement Predictor
================================================================
Predicting a *point* price months ahead is unreliable — the honest output is a
distribution. We use a Monte-Carlo Geometric-Brownian-Motion model calibrated to
the stock's own drift & volatility, with two corrections that keep it grounded:

  • DAMPENED DRIFT   raw historical drift over-extrapolates; we shrink it.
  • ANALYST ANCHOR   the yfinance mean target price (~12-month analyst view) is
                     blended into the drift so the forecast leans toward where
                     professionals expect the stock to go.

For each horizon we report the median path plus 50% (p25–p75) and 90% (p5–p95)
confidence bands, and the % upside vs the last close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_MONTH = 21


@dataclass
class PriceForecast:
    last_close:     float
    months:         list[int]
    horizon_days:   list[int]
    median:         dict          # month -> median price
    p25:            dict
    p75:            dict
    p5:             dict
    p95:            dict
    upside:         dict          # month -> % vs last close
    # path arrays for charting (length = max horizon + 1)
    path_median:    np.ndarray
    path_p25:       np.ndarray
    path_p75:       np.ndarray
    path_p5:        np.ndarray
    path_p95:       np.ndarray
    analyst_target: float | None = None
    drift_annual:   float = 0.0
    vol_annual:     float = 0.0
    meta:           dict = field(default_factory=dict)


def forecast_price(close: pd.Series,
                   analyst_target: float | None = None,
                   months: tuple[int, ...] = (1, 2, 3, 4),
                   n_paths: int = 5000,
                   drift_damping: float = 0.5,
                   analyst_weight: float = 0.5,
                   seed: int = 7) -> PriceForecast:
    close = close.dropna().astype(float)
    s0    = float(close.iloc[-1])
    logret = np.log(close / close.shift(1)).dropna()

    mu_hist = float(logret.mean())          # daily log drift
    sigma   = float(logret.std())           # daily log vol

    # 1) dampen the historical drift (raw drift wildly over-extrapolates)
    mu_used = mu_hist * drift_damping

    # 2) blend toward the analyst target's implied drift (≈12-month view)
    if analyst_target and analyst_target > 0:
        mu_analyst = np.log(analyst_target / s0) / TRADING_DAYS
        mu_used    = analyst_weight * mu_analyst + (1 - analyst_weight) * mu_used

    max_days = max(months) * DAYS_PER_MONTH
    rng      = np.random.default_rng(seed)

    # GBM: log-increments ~ Normal(mu_used - 0.5σ², σ)
    incr  = rng.normal(mu_used - 0.5 * sigma ** 2, sigma, size=(max_days, n_paths))
    paths = s0 * np.exp(np.vstack([np.zeros((1, n_paths)), np.cumsum(incr, axis=0)]))

    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    path_p5, path_p25, path_median, path_p75, path_p95 = q

    median, p25, p75, p5, p95, upside, hdays = {}, {}, {}, {}, {}, {}, []
    for m in months:
        d = m * DAYS_PER_MONTH
        hdays.append(d)
        median[m] = float(path_median[d])
        p25[m]    = float(path_p25[d])
        p75[m]    = float(path_p75[d])
        p5[m]     = float(path_p5[d])
        p95[m]    = float(path_p95[d])
        upside[m] = float((path_median[d] - s0) / s0 * 100)

    return PriceForecast(
        last_close=s0, months=list(months), horizon_days=hdays,
        median=median, p25=p25, p75=p75, p5=p5, p95=p95, upside=upside,
        path_median=path_median, path_p25=path_p25, path_p75=path_p75,
        path_p5=path_p5, path_p95=path_p95,
        analyst_target=float(analyst_target) if analyst_target else None,
        drift_annual=float(mu_used * TRADING_DAYS),
        vol_annual=float(sigma * np.sqrt(TRADING_DAYS)),
        meta={"n_paths": n_paths, "drift_damping": drift_damping,
              "analyst_weight": analyst_weight if analyst_target else 0.0},
    )
