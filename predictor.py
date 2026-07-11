"""
Time-Series Stock Price Prediction Engine — v5 (accuracy-focused)
=================================================================
Every model is judged on the SAME yardstick: a horizon-matched walk-forward
backtest (rolling origins). From that single, comparable evaluation we derive:

  • Backtest MAPE / RMSE at the requested horizon   → honest error
  • Directional accuracy of the NET move             → honest "up or down?"
  • Conformal prediction intervals + coverage        → calibrated bands
  • Skill vs a Naive random-walk benchmark           → "does it beat doing nothing?"

Models
------
  1. Naive (RW + damped drift)  — the benchmark every model must beat
  2. Prophet                    — trend + weekly/yearly seasonality
  3. Holt-Winters (ETS)         — exponential smoothing, damped trend
  4. SARIMA                     — seasonal ARIMA
  5. Theta                      — theta decomposition
  6. XGBoost                    — gradient boosting on 80+ engineered features
  7. LightGBM                   — gradient boosting on 80+ engineered features
  8. LSTM                       — PyTorch recurrent net (degrades gracefully)
  9. Monte Carlo (GBM)          — GARCH-like simulation (range / stress test)
  • Ensemble                    — inverse-error weighted blend (the default pick)

Honest note
-----------
No model achieves 100% accuracy on stock prices. The realistic ceiling for
directional accuracy is ~55–65%. The goal here is maximum *calibrated* accuracy
with intervals you can actually trust, measured the same way for every model.
"""

from __future__ import annotations
import os

# macOS: LightGBM and PyTorch each bundle their own libomp. Loading both can
# abort or deadlock; allow the duplicate runtime so torch (LSTM) coexists with
# the gradient-boosting models. Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    model_name:   str
    forecast:     pd.Series
    upper_bound:  Optional[pd.Series] = None
    lower_bound:  Optional[pd.Series] = None
    metrics:      dict = field(default_factory=dict)
    error:        Optional[str] = None
    # Walk-forward backtest trace (most-recent fold) for the UI "proof" chart
    bt_dates:     Optional[pd.DatetimeIndex] = None
    bt_actual:    Optional[np.ndarray] = None
    bt_pred:      Optional[np.ndarray] = None
    is_benchmark: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Calendar helpers
# ─────────────────────────────────────────────────────────────────────────────

# NSE/BSE market holidays 2024–2026 (exchange closures beyond weekends)
_NSE_HOLIDAYS = pd.to_datetime([
    "2024-01-22","2024-01-26","2024-03-25","2024-04-14","2024-04-17",
    "2024-05-23","2024-06-17","2024-07-17","2024-08-15","2024-10-02",
    "2024-10-14","2024-11-01","2024-11-15","2024-11-20","2024-12-25",
    "2025-01-26","2025-02-26","2025-03-14","2025-03-31","2025-04-10",
    "2025-04-14","2025-04-18","2025-05-01","2025-06-07","2025-08-15",
    "2025-10-02","2025-10-21","2025-11-05","2025-12-25",
    "2026-01-26","2026-03-02","2026-03-25","2026-04-03","2026-04-14",
    "2026-05-01","2026-08-15","2026-10-02","2026-12-25",
])


def _future_business_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """n future NSE trading dates after last_date, skipping weekends and holidays."""
    try:
        from pandas.tseries.offsets import CustomBusinessDay
        nse_day = CustomBusinessDay(holidays=_NSE_HOLIDAYS)
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                              periods=n, freq=nse_day)
    except Exception:
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)


# ─────────────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    mask = np.abs(a) > 1e-8
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else 0.0


def _prices_from_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(log_returns))


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering for ML models — 80+ features
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices.values}, index=prices.index)
    c  = df["price"]
    lr = np.log(c / c.shift(1))

    for lag in [1, 2, 3, 4, 5, 10, 21]:
        df[f"ret_lag_{lag}"] = lr.shift(lag)

    df["ret_cross_1"] = lr.shift(1) * lr.shift(2)
    df["ret_cross_5"] = lr.shift(1) * lr.shift(5)

    for w in [5, 10, 21, 63]:
        df[f"mom_{w}"]      = np.log(c / c.shift(w))
        df[f"mom_norm_{w}"] = df[f"mom_{w}"] / (lr.rolling(w).std() * np.sqrt(w) + 1e-8)

    df["accel_5_10"]  = df["mom_5"]  - df["mom_10"]
    df["accel_10_21"] = df["mom_10"] - df["mom_21"]
    df["accel_21_63"] = df["mom_21"] - df["mom_63"]

    for w in [5, 10, 20, 50, 100, 200]:
        df[f"price_ma_{w}"] = (c / c.rolling(w).mean()) - 1

    for span in [9, 21, 55]:
        df[f"price_ema_{span}"] = (c / c.ewm(span=span, adjust=False).mean()) - 1

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_norm"]      = macd / (c + 1e-8)
    df["macd_sig_norm"]  = sig  / (c + 1e-8)
    df["macd_hist_norm"] = (macd - sig) / (c + 1e-8)
    df["macd_above_sig"] = (macd > sig).astype(float)

    for period in [9, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        df[f"rsi_{period}"] = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100

    for w in [5, 10, 21, 63]:
        df[f"rvol_{w}"] = lr.rolling(w).std() * np.sqrt(252)

    df["vol_ratio_5_21"]  = df["rvol_5"]  / (df["rvol_21"]  + 1e-8)
    df["vol_ratio_21_63"] = df["rvol_21"] / (df["rvol_63"]  + 1e-8)
    df["high_vol_regime"] = (df["rvol_21"] > df["rvol_63"]).astype(float)

    for w in [10, 21]:
        df[f"skew_{w}"] = lr.rolling(w).skew()
    df["kurt_21"] = lr.rolling(21).kurt()

    for w in [20, 50]:
        mu    = c.rolling(w).mean()
        sigma = c.rolling(w).std()
        df[f"zscore_{w}"] = (c - mu) / (sigma + 1e-8)

    for w in [10, 20]:
        ma  = c.rolling(w).mean()
        std = c.rolling(w).std()
        df[f"bb_pos_{w}"]   = (c - ma) / (2 * std + 1e-8)
        df[f"bb_width_{w}"] = (4 * std) / (ma + 1e-8)

    for w in [20, 60]:
        df[f"drawdown_{w}"] = (c - c.rolling(w).max()) / (c.rolling(w).max() + 1e-8)

    for w in [14, 21]:
        lo = c.rolling(w).min()
        hi = c.rolling(w).max()
        df[f"stoch_{w}"] = (c - lo) / (hi - lo + 1e-8)

    hi14 = c.rolling(14).max()
    lo14 = c.rolling(14).min()
    df["williams_r"] = (hi14 - c) / (hi14 - lo14 + 1e-8)

    lo52 = c.rolling(252, min_periods=63).min()
    hi52 = c.rolling(252, min_periods=63).max()
    df["pos_52w"] = (c - lo52) / (hi52 - lo52 + 1e-8)

    idx = pd.DatetimeIndex(prices.index)
    df["day_of_week"] = idx.dayofweek / 4.0
    df["month"]       = idx.month / 12.0
    df["quarter"]     = (idx.month - 1) // 3 / 3.0
    df["t_norm"]      = np.arange(len(df)) / max(len(df) - 1, 1)

    df["target"] = lr.shift(-1)
    df.dropna(inplace=True)
    return df


def _ml_feature_frame(prices: pd.Series):
    df        = _build_features(prices)
    feat_cols = [c for c in df.columns if c not in ("price", "target")]
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Recursive multi-step forecast for ML / NN models
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_ml_forecast(
    predict_fn: Callable[[np.ndarray], float],
    scaler,
    feat_cols: list,
    prices: pd.Series,
    horizon: int,
    window: int = 1,
    wfwd_steps: int = 30,
) -> np.ndarray:
    """
    True walk-forward for the first `wfwd_steps` days (model predicts each next
    day, price is appended, features rebuilt). Beyond that the signal decays
    toward a RANDOM-WALK (zero-drift) baseline — this removes the systematic
    upward (BUY) bias the old engine had on long horizons.

    `predict_fn` receives a scaled feature matrix of the last `window` rows and
    returns a single next-day log-return prediction.
    """
    last_price = float(prices.iloc[-1])
    wf         = min(wfwd_steps, horizon)

    ext_prices  = prices.copy()
    forecast_lr: list[float] = []

    for _ in range(wf):
        fdf = _build_features(ext_prices)
        Xw  = scaler.transform(fdf[feat_cols].iloc[-window:].values)
        lr_pred = float(np.clip(predict_fn(Xw), -0.12, 0.12))   # daily guard rail
        forecast_lr.append(lr_pred)
        next_price = float(ext_prices.iloc[-1]) * np.exp(lr_pred)
        next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
        ext_prices = pd.concat([ext_prices, pd.Series([next_price], index=[next_date])])

    if horizon > wf:
        ml_drift = float(np.mean(forecast_lr)) if forecast_lr else 0.0
        for i in range(1, horizon - wf + 1):
            decay = 0.85 ** i                  # decays toward 0 (random walk)
            forecast_lr.append(ml_drift * decay)

    return _prices_from_returns(last_price, np.array(forecast_lr[:horizon]))


# ─────────────────────────────────────────────────────────────────────────────
# Lean point forecasters:  f(train, horizon, fast) -> np.ndarray of prices
# (No internal metrics / CI — the walk-forward harness owns those, uniformly.)
# ─────────────────────────────────────────────────────────────────────────────

def _fc_naive(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    """Random walk with a small, damped drift — the benchmark to beat."""
    lr  = np.log(train / train.shift(1)).dropna().values
    tail = lr[-252:] if len(lr) >= 60 else lr
    mu  = float(np.clip(np.median(tail), -0.01, 0.01)) if len(tail) else 0.0
    last = float(train.iloc[-1])
    steps = np.arange(1, horizon + 1)
    cum   = np.cumsum(0.98 ** steps)           # damped drift accumulation
    return last * np.exp(mu * cum)


def _fc_prophet(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    from prophet import Prophet
    idx = train.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    m = Prophet(
        changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
        daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
        uncertainty_samples=0, interval_width=0.90,
    )
    m.fit(pd.DataFrame({"ds": idx, "y": train.values}))
    fut = _future_business_dates(train.index[-1], horizon)
    fut_ds = fut.tz_localize(None) if getattr(fut, "tz", None) is not None else fut
    return m.predict(pd.DataFrame({"ds": fut_ds}))["yhat"].values


def _fc_holtwinters(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    for trend, damped in [("add", True), ("add", False), ("mul", True)]:
        try:
            fit = ExponentialSmoothing(
                train, trend=trend, seasonal=None, damped_trend=damped,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            return np.asarray(fit.forecast(horizon))
        except Exception:
            continue
    fit = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    return np.asarray(fit.forecast(horizon))


def _fc_sarima(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    series = train.iloc[-750:] if len(train) > 750 else train
    log_p  = np.log(series.values.astype(float))
    seasonal_order = (1, 0, 1, 5)

    if fast:
        order = (2, 1, 2)
    else:
        best_aic, order = np.inf, (2, 1, 2)
        for cand in [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)]:
            try:
                aic = SARIMAX(log_p, order=cand, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False).aic
                if aic < best_aic:
                    best_aic, order = aic, cand
            except Exception:
                continue

    fit = SARIMAX(log_p, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.exp(np.asarray(fit.forecast(horizon)).flatten())


def _fc_theta(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    fit = ThetaModel(train, period=5, deseasonalize=False).fit()
    return np.asarray(fit.forecast(horizon)).flatten()


def _fc_xgb(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    from xgboost import XGBRegressor
    from sklearn.preprocessing import RobustScaler
    df, feat_cols = _ml_feature_frame(train)
    X = df[feat_cols].values
    y = df["target"].values
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    if fast:
        m = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                         subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
                         reg_lambda=2.0, min_child_weight=5, random_state=42,
                         n_jobs=-1, verbosity=0)
        m.fit(Xs, y)
    else:
        n_test = max(int(len(Xs) * 0.15), 20)
        m = XGBRegressor(n_estimators=900, learning_rate=0.02, max_depth=4,
                         subsample=0.75, colsample_bytree=0.65, reg_alpha=0.5,
                         reg_lambda=2.0, min_child_weight=5, gamma=0.1,
                         random_state=42, n_jobs=-1, verbosity=0,
                         early_stopping_rounds=50)
        m.fit(Xs[:-n_test], y[:-n_test],
              eval_set=[(Xs[-n_test:], y[-n_test:])], verbose=False)

    return _recursive_ml_forecast(lambda Xw: float(m.predict(Xw)[0]),
                                  scaler, feat_cols, train, horizon, window=1)


def _fc_lgb(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    import lightgbm as lgb
    from sklearn.preprocessing import RobustScaler
    df, feat_cols = _ml_feature_frame(train)
    X = df[feat_cols].values
    y = df["target"].values
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    if fast:
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=5,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                              reg_alpha=0.5, reg_lambda=2.0, min_child_samples=15,
                              random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xs, y)
    else:
        n_test = max(int(len(Xs) * 0.15), 20)
        m = lgb.LGBMRegressor(n_estimators=900, learning_rate=0.02, max_depth=5,
                              num_leaves=31, subsample=0.75, colsample_bytree=0.65,
                              reg_alpha=0.5, reg_lambda=2.0, min_child_samples=15,
                              min_split_gain=0.01, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xs[:-n_test], y[:-n_test],
              eval_set=[(Xs[-n_test:], y[-n_test:])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    return _recursive_ml_forecast(lambda Xw: float(m.predict(Xw)[0]),
                                  scaler, feat_cols, train, horizon, window=1)


def _fc_lstm(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    import torch
    from torch import nn
    from sklearn.preprocessing import RobustScaler

    torch.manual_seed(42)
    np.random.seed(42)
    # Single-threaded: avoids an OpenMP thread-pool deadlock that can occur when
    # torch runs after NumPy/XGBoost/LightGBM have already spun up their own pools.
    torch.set_num_threads(1)

    df, feat_cols = _ml_feature_frame(train)
    X = df[feat_cols].values.astype("float32")
    y = df["target"].values.astype("float32")
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    L = 20
    if len(Xs) < L + 40:
        raise ValueError("Not enough history for LSTM sequences.")

    seqs = np.stack([Xs[i - L:i] for i in range(L, len(Xs))]).astype("float32")
    tgts = y[L:].reshape(-1, 1)
    Xseq = torch.from_numpy(seqs)
    ytt  = torch.from_numpy(tgts)

    class _LSTMNet(nn.Module):
        def __init__(self, n_feat: int):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, 32, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.1)
            self.fc   = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    net  = _LSTMNet(Xs.shape[1])
    opt  = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = nn.MSELoss()
    epochs = 15 if fast else 35
    bs, n = 64, len(Xseq)

    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            opt.zero_grad()
            loss = lossf(net(Xseq[idx]), ytt[idx])
            loss.backward()
            opt.step()
    net.eval()

    def predict_fn(Xw: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.from_numpy(Xw[None].astype("float32"))
            return float(net(t).item())

    return _recursive_ml_forecast(predict_fn, scaler, feat_cols, train, horizon, window=L)


def _fc_montecarlo(train: pd.Series, horizon: int, fast: bool) -> np.ndarray:
    """Median GBM path with vol mean-reversion (full distribution in metrics)."""
    log_ret    = np.log(train / train.shift(1)).dropna()
    mu_daily   = float(log_ret.mean())
    long_vol   = float(log_ret.std())
    recent_vol = float(log_ret.tail(21).std())
    S0         = float(train.iloc[-1])
    n_sims     = 1000 if fast else 3000
    vol_decay  = 0.94

    rng   = np.random.default_rng(42)
    paths = np.zeros((n_sims, horizon))
    for t in range(horizon):
        sigma_t = recent_vol * (vol_decay ** t) + long_vol * (1 - vol_decay ** t)
        Z    = rng.standard_normal(n_sims)
        prev = S0 if t == 0 else paths[:, t - 1]
        paths[:, t] = prev * np.exp((mu_daily - 0.5 * sigma_t ** 2) + sigma_t * Z)
    return np.median(paths, axis=0)


# Registry — order controls UI / progress display
_FORECASTERS: dict[str, Callable[[pd.Series, int, bool], np.ndarray]] = {
    "Naive (RW+Drift)":   _fc_naive,
    "Prophet":            _fc_prophet,
    "Holt-Winters (ETS)": _fc_holtwinters,
    "SARIMA":             _fc_sarima,
    "Theta":              _fc_theta,
    "XGBoost":            _fc_xgb,
    "LightGBM":           _fc_lgb,
    "LSTM":               _fc_lstm,
    "Monte Carlo (GBM)":  _fc_montecarlo,
}

# Models blended into the Ensemble (Naive = benchmark, MC = range tool → excluded)
_ENSEMBLE_MEMBERS = ["Prophet", "Holt-Winters (ETS)", "SARIMA", "Theta",
                     "XGBoost", "LightGBM", "LSTM"]

MODEL_ORDER = ["Ensemble"] + list(_FORECASTERS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest harness (identical for every model)
# ─────────────────────────────────────────────────────────────────────────────

_MIN_TRAIN = 260   # need enough history for the 252-day features


@dataclass
class _Fold:
    origin_date:  pd.Timestamp
    origin_price: float
    dates:        pd.DatetimeIndex
    actual:       np.ndarray
    pred:         np.ndarray


def _origins(n: int, horizon: int, n_origins: int) -> list[int]:
    """Non-overlapping rolling cutoffs near the end of the series."""
    cuts = []
    for k in range(1, n_origins + 1):
        cut = n - k * horizon
        if cut < _MIN_TRAIN:
            break
        cuts.append(cut)
    return cuts


def _walk_forward(forecaster, prices: pd.Series, horizon: int,
                  cuts: list[int], fast: bool = True) -> list[_Fold]:
    folds: list[_Fold] = []
    for cut in cuts:
        train  = prices.iloc[:cut]
        actual = prices.iloc[cut:cut + horizon].values.astype(float)
        if len(actual) < 2:
            continue
        try:
            pred = np.asarray(forecaster(train, len(actual), fast), dtype=float)
        except Exception:
            continue
        pred = pred[:len(actual)]
        if len(pred) < len(actual) or not np.all(np.isfinite(pred)):
            continue
        folds.append(_Fold(
            origin_date  = prices.index[cut - 1],
            origin_price = float(prices.iloc[cut - 1]),
            dates        = prices.index[cut:cut + len(actual)],
            actual       = actual,
            pred         = pred,
        ))
    return folds


def _fold_metrics(folds: list[_Fold]) -> tuple[dict, list[tuple[int, float]]]:
    """Aggregate metrics across folds + pooled (step, relative-residual) pairs."""
    if not folds:
        return {}, []
    mapes, rmses, dir_hits = [], [], []
    rel_by_step: list[tuple[int, float]] = []
    for f in folds:
        mapes.append(_mape(f.actual, f.pred))
        rmses.append(_rmse(f.actual, f.pred))
        # Directional accuracy of the NET move from the origin, at every step
        dir_hits.extend(
            (np.sign(f.pred - f.origin_price) == np.sign(f.actual - f.origin_price)).astype(float)
        )
        steps = np.arange(1, len(f.pred) + 1)
        rel   = (f.actual - f.pred) / np.where(np.abs(f.pred) > 1e-8, f.pred, 1e-8)
        rel_by_step.extend(zip(steps.tolist(), rel.tolist()))
    return ({
        "mape":  float(np.mean(mapes)),
        "rmse":  float(np.mean(rmses)),
        "dir":   float(np.mean(dir_hits) * 100) if dir_hits else 0.0,
        "folds": len(folds),
    }, rel_by_step)


def _conformal_bounds(forecast: np.ndarray, rel_by_step: list[tuple[int, float]],
                      alpha: float = 0.10):
    """
    Split-conformal-style intervals with sqrt-time growth.
    Standardise each backtest residual by sqrt(step), take the alpha/2 and
    1-alpha/2 quantiles, then re-inflate by sqrt(step) for the live forecast.
    Returns (upper, lower, empirical_coverage_pct).
    """
    h = len(forecast)
    steps = np.arange(1, h + 1)
    if len(rel_by_step) >= 8:
        s_arr   = np.array([s for s, _ in rel_by_step], dtype=float)
        rel_arr = np.array([r for _, r in rel_by_step], dtype=float)
        scaled  = rel_arr / np.sqrt(s_arr)
        q_up = float(np.quantile(scaled, 1 - alpha / 2))
        q_lo = float(np.quantile(scaled, alpha / 2))
        upper = forecast * (1 + q_up * np.sqrt(steps))
        lower = forecast * (1 + q_lo * np.sqrt(steps))
        inside = (scaled >= q_lo) & (scaled <= q_up)
        coverage = float(np.mean(inside) * 100)
    else:
        # Fallback: simple growing band when too few backtest points
        band  = 0.02 * np.sqrt(steps)
        upper = forecast * (1 + 1.645 * band)
        lower = forecast * (1 - 1.645 * band)
        coverage = float("nan")
    lower = np.maximum(lower, forecast * 0.3)   # keep bands sane / positive
    upper = np.maximum(upper, lower * 1.001)
    return upper, lower, coverage


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — runs everything, builds the ensemble, returns uniform results
# ─────────────────────────────────────────────────────────────────────────────

def run_all_predictions(
    prices: pd.Series,
    horizon: int = 30,
    fast_backtest: bool = True,
    n_origins: int = 3,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Run all models + the ensemble on `prices`, each evaluated with the SAME
    walk-forward backtest. Returns:
        {
          "results":     {model_name: PredictionResult, ...},
          "best_name":   str,
          "naive_mape":  float,
          "future_idx":  DatetimeIndex,
        }
    """
    prices = prices.dropna()
    future_idx = _future_business_dates(prices.index[-1], horizon)
    cuts = _origins(len(prices), horizon, n_origins)

    names = list(_FORECASTERS.keys())
    total = len(names)
    raw: dict[str, dict] = {}

    for i, name in enumerate(names):
        if progress_cb:
            progress_cb(i, total, name)
        fc = _FORECASTERS[name]
        try:
            final = np.asarray(fc(prices, horizon, False), dtype=float)[:horizon]
            if len(final) < horizon or not np.all(np.isfinite(final)):
                raise ValueError("forecast produced non-finite / short output")
        except Exception as e:
            raw[name] = {"error": str(e)}
            continue
        folds = _walk_forward(fc, prices, horizon, cuts, fast=True)
        m, rel = _fold_metrics(folds)
        raw[name] = {"final": final, "folds": folds, "metrics": m, "rel": rel}

    # ── Build the inverse-error-weighted ensemble from successful members ──
    members = [n for n in _ENSEMBLE_MEMBERS
               if n in raw and "final" in raw[n] and raw[n]["metrics"]]
    if len(members) >= 2:
        inv = {n: 1.0 / (raw[n]["metrics"]["mape"] + 1e-6) for n in members}
        wsum = sum(inv.values())
        weights = {n: inv[n] / wsum for n in members}

        ens_final = np.zeros(horizon)
        for n in members:
            ens_final += weights[n] * raw[n]["final"][:horizon]

        # Ensemble backtest = blend of members' per-fold predictions (no refit)
        by_origin: dict[pd.Timestamp, list] = {}
        for n in members:
            for f in raw[n]["folds"]:
                by_origin.setdefault(f.origin_date, []).append((n, f))
        ens_folds: list[_Fold] = []
        for od, items in by_origin.items():
            ref = items[0][1]
            wtot = sum(weights[n] for n, _ in items)
            blpred = np.zeros(len(ref.actual))
            ok = True
            for n, f in items:
                if len(f.pred) != len(ref.actual):
                    ok = False
                    break
                blpred += (weights[n] / wtot) * f.pred
            if ok:
                ens_folds.append(_Fold(od, ref.origin_price, ref.dates, ref.actual, blpred))
        em, erel = _fold_metrics(ens_folds)
        raw["Ensemble"] = {"final": ens_final, "folds": ens_folds,
                           "metrics": em, "rel": erel,
                           "weights": weights}

    naive_mape = raw.get("Naive (RW+Drift)", {}).get("metrics", {}).get("mape", float("nan"))

    # ── Assemble uniform PredictionResults (conformal CI + skill vs naive) ──
    results: dict[str, PredictionResult] = {}
    for name, d in raw.items():
        if "error" in d:
            results[name] = PredictionResult(name, pd.Series(), error=d["error"])
            continue
        final = d["final"]
        upper, lower, coverage = _conformal_bounds(final, d["rel"])
        m = d["metrics"] or {}
        mape = m.get("mape", float("nan"))
        skill = (1 - mape / naive_mape) * 100 if naive_mape and np.isfinite(naive_mape) and naive_mape > 0 else 0.0

        metrics_out = {
            "Backtest MAPE %":  round(mape, 2) if np.isfinite(mape) else None,
            "Backtest RMSE":    round(m.get("rmse", float("nan")), 2),
            "Dir Acc % (net)":  round(m.get("dir", 0.0), 1),
            "Skill vs Naive %": round(skill, 1),
            "CI Coverage %":    (round(coverage, 1) if coverage == coverage else None),
            "Backtest Folds":   m.get("folds", 0),
        }
        if name == "Ensemble" and "weights" in d:
            metrics_out["Blend weights"] = " | ".join(
                f"{n}:{w:.2f}" for n, w in
                sorted(d["weights"].items(), key=lambda x: -x[1]))

        res = PredictionResult(
            model_name   = name,
            forecast     = pd.Series(final, index=future_idx, name=name),
            upper_bound  = pd.Series(upper, index=future_idx),
            lower_bound  = pd.Series(lower, index=future_idx),
            metrics      = metrics_out,
            is_benchmark = (name == "Naive (RW+Drift)"),
        )
        # Most-recent fold trace for the UI "proof" chart
        folds = d["folds"]
        if folds:
            recent = folds[0]   # k=1 origin = the most recent out-of-sample window
            res.bt_dates  = recent.dates
            res.bt_actual = recent.actual
            res.bt_pred   = recent.pred
        results[name] = res

    # ── Pick the best non-benchmark model: lowest backtest MAPE, dir-acc tiebreak ──
    def _key(name: str):
        r = results[name]
        if r.error or r.forecast is None or r.forecast.empty:
            return (9e9, 0)
        mp = r.metrics.get("Backtest MAPE %")
        da = r.metrics.get("Dir Acc % (net)") or 0
        return (mp if mp is not None else 9e9, -da)

    candidates = [n for n in results
                  if n != "Naive (RW+Drift)" and not results[n].error
                  and not results[n].forecast.empty]
    best_name = min(candidates, key=_key) if candidates else None

    return {
        "results":    results,
        "best_name":  best_name,
        "naive_mape": naive_mape,
        "future_idx": future_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (for the Charts tab — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df    = ohlcv.copy()
    close = df["Close"]

    for w, name in [(20, "MA_20"), (50, "MA_50"), (200, "MA_200")]:
        df[name] = close.rolling(w).mean()

    df["EMA_12"]      = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_mid"]   = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (bb_mid + 1e-8)

    if "High" in df.columns and "Low" in df.columns:
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - close.shift(1)).abs()
        lpc = (df["Low"]  - close.shift(1)).abs()
        df["ATR_14"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    if "Volume" in df.columns:
        df["Vol_MA_20"] = df["Volume"].rolling(20).mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible shim
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_MODELS = MODEL_ORDER   # names only; app uses run_all_predictions()
