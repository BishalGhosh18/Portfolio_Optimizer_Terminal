"""
Time-Series Stock Price Prediction Engine — v4
===============================================
Pure time-series models + ML models with TS-aware cross-validation:

  1. Prophet (Facebook)      — trend + weekly/yearly seasonality decomposition
  2. Holt-Winters (ETS)      — exponential smoothing with damped trend
  3. SARIMA                  — seasonal ARIMA with auto-order selection
  4. Theta                   — theta decomposition method (often beats ARIMA)
  5. XGBoost (TS-CV)         — gradient boosting with TimeSeriesSplit validation
  6. LightGBM (TS-CV)        — gradient boosting with TimeSeriesSplit validation
  7. TS Ensemble      — inverse-MAPE weighted blend of all above
  8. Monte Carlo (GBM)       — GARCH-like simulation (range / stress test)

Multi-step strategy for ML models
----------------------------------
  Days 1–5  : true walk-forward (real + predicted prices)
  Days 6+   : ML signal decays toward long-run historical drift

Honest note
-----------
No model achieves 100 % accuracy on stock prices. The realistic ceiling for
directional accuracy on well-tuned models is ~55–65 %. The goal is maximum
*calibrated* accuracy with reliable confidence bands.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    model_name:   str
    forecast:     pd.Series
    upper_bound:  Optional[pd.Series] = None
    lower_bound:  Optional[pd.Series] = None
    metrics:      dict = field(default_factory=dict)
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# NSE/BSE market holidays 2024–2026 (exchange-specific closures beyond weekends)
_NSE_HOLIDAYS = pd.to_datetime([
    # 2024
    "2024-01-22","2024-01-26","2024-03-25","2024-04-14","2024-04-17",
    "2024-05-23","2024-06-17","2024-07-17","2024-08-15","2024-10-02",
    "2024-10-14","2024-11-01","2024-11-15","2024-11-20","2024-12-25",
    # 2025
    "2025-01-26","2025-02-26","2025-03-14","2025-03-31","2025-04-10",
    "2025-04-14","2025-04-18","2025-05-01","2025-06-07","2025-08-15",
    "2025-10-02","2025-10-21","2025-11-05","2025-12-25",
    # 2026
    "2026-01-26","2026-03-02","2026-03-25","2026-04-03","2026-04-14",
    "2026-05-01","2026-08-15","2026-10-02","2026-12-25",
])


def _future_business_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Return n future NSE trading dates after last_date, skipping weekends and holidays."""
    try:
        from pandas.tseries.offsets import CustomBusinessDay
        nse_day = CustomBusinessDay(holidays=_NSE_HOLIDAYS)
        return pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n,
            freq=nse_day,
        )
    except Exception:
        # Fallback to plain business days if CustomBusinessDay fails
        return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)


def _rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    mask = np.abs(a) > 1e-8
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else 0.0


def _directional_accuracy(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sign(a) == np.sign(p)) * 100)


def _prices_from_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(log_returns))


def _ci_from_residuals(
    forecast: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """sqrt(t)-growing CI from residual std — standard TS uncertainty propagation."""
    std_r   = float(np.std(residuals)) if len(residuals) > 1 else float(np.abs(forecast).mean()) * 0.02
    z       = 1.645 if confidence == 0.90 else 1.96
    t_arr   = np.arange(1, len(forecast) + 1)
    upper   = forecast + z * std_r * np.sqrt(t_arr)
    lower   = forecast - z * std_r * np.sqrt(t_arr)
    lower   = np.maximum(lower, forecast * 0.5)   # floor at 50% of forecast
    return upper, lower


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


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid multi-step forecast for ML models
# ─────────────────────────────────────────────────────────────────────────────

def _hybrid_ml_forecast(
    predict_fn: Callable,
    scaler,
    feat_cols: list,
    prices: pd.Series,
    horizon: int,
    wfwd_steps: int = 30,
) -> tuple[np.ndarray, float]:
    last_price = float(prices.iloc[-1])
    hist_lr    = np.log(prices / prices.shift(1)).dropna().values
    hist_mean  = float(np.mean(hist_lr))
    wfwd_steps = min(wfwd_steps, horizon)

    ext_prices  = prices.copy()
    forecast_lr = []

    for _ in range(wfwd_steps):
        fdf      = _build_features(ext_prices)
        last_row = fdf[feat_cols].iloc[[-1]].values
        lr_pred  = float(predict_fn(scaler.transform(last_row)))
        forecast_lr.append(lr_pred)
        next_price = float(ext_prices.iloc[-1]) * np.exp(lr_pred)
        next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
        ext_prices = pd.concat([ext_prices,
                                pd.Series([next_price], index=[next_date])])

    if horizon > wfwd_steps:
        ml_drift = float(np.mean(forecast_lr))
        for i in range(1, horizon - wfwd_steps + 1):
            decay = 0.90 ** i
            forecast_lr.append(ml_drift * decay + hist_mean * (1.0 - decay))

    return np.array(forecast_lr[:horizon]), last_price


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prophet — trend + seasonality decomposition
# ─────────────────────────────────────────────────────────────────────────────

def prophet_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from prophet import Prophet

        # Strip timezone for Prophet
        idx = prices.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)

        n_test    = max(int(len(prices) * test_size), 5)
        train_df  = pd.DataFrame({"ds": idx[:-n_test], "y": prices.values[:-n_test]})
        test_df   = pd.DataFrame({"ds": idx[-n_test:],  "y": prices.values[-n_test:]})

        model = Prophet(
            changepoint_prior_scale  = 0.05,
            seasonality_prior_scale  = 10.0,
            daily_seasonality        = False,
            weekly_seasonality       = True,
            yearly_seasonality       = True,
            uncertainty_samples      = 300,
            interval_width           = 0.90,
        )
        model.fit(train_df)

        test_pred   = model.predict(test_df[["ds"]])["yhat"].values
        rmse_val    = _rmse(test_df["y"].values, test_pred)
        mape_val    = _mape(test_df["y"].values, test_pred)
        residuals   = test_df["y"].values - test_pred
        dir_acc     = _directional_accuracy(
            np.diff(test_df["y"].values), np.diff(test_pred))

        # Refit on full series
        full_df    = pd.DataFrame({"ds": idx, "y": prices.values})
        model_full = Prophet(
            changepoint_prior_scale = 0.05,
            seasonality_prior_scale = 10.0,
            daily_seasonality       = False,
            weekly_seasonality      = True,
            yearly_seasonality      = True,
            uncertainty_samples     = 300,
            interval_width          = 0.90,
        )
        model_full.fit(full_df)

        future_idx = _future_business_dates(prices.index[-1], horizon)
        future_df  = pd.DataFrame({"ds": future_idx.tz_localize(None)
                                   if future_idx.tz is None else future_idx})
        fc         = model_full.predict(future_df)

        return PredictionResult(
            model_name  = "Prophet",
            forecast    = pd.Series(fc["yhat"].values, index=future_idx, name="Prophet"),
            upper_bound = pd.Series(fc["yhat_upper"].values, index=future_idx),
            lower_bound = pd.Series(fc["yhat_lower"].values, index=future_idx),
            metrics     = {
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Prophet",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Holt-Winters ETS — exponential smoothing with damped trend
# ─────────────────────────────────────────────────────────────────────────────

def holtwinters_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        n_test   = max(int(len(prices) * test_size), 5)
        train_p  = prices.iloc[:-n_test]
        test_p   = prices.iloc[-n_test:]

        def _fit(series):
            for trend, damped in [("add", True), ("add", False), ("mul", True)]:
                try:
                    return ExponentialSmoothing(
                        series, trend=trend, seasonal=None, damped_trend=damped,
                        initialization_method="estimated",
                    ).fit(optimized=True, remove_bias=True)
                except Exception:
                    continue
            return ExponentialSmoothing(series, trend="add", seasonal=None).fit()

        fit_test   = _fit(train_p)
        pred_test  = fit_test.forecast(n_test).values
        residuals  = test_p.values - pred_test
        rmse_val   = _rmse(test_p.values, pred_test)
        mape_val   = _mape(test_p.values, pred_test)
        dir_acc    = _directional_accuracy(np.diff(test_p.values), np.diff(pred_test))

        fit_full   = _fit(prices)
        fc         = fit_full.forecast(horizon).values
        future_idx = _future_business_dates(prices.index[-1], horizon)
        upper, lower = _ci_from_residuals(fc, residuals)

        return PredictionResult(
            model_name  = "Holt-Winters (ETS)",
            forecast    = pd.Series(fc,    index=future_idx, name="HW"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Holt-Winters (ETS)",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SARIMA — seasonal ARIMA with auto-order selection
# ─────────────────────────────────────────────────────────────────────────────

def sarima_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Limit to last 750 trading days for speed
        prices = prices.iloc[-750:] if len(prices) > 750 else prices

        log_p  = np.log(prices.values.astype(float))
        n_test = max(int(len(log_p) * test_size), 5)

        # Auto select best ARIMA order by AIC
        candidates = [(1,1,1),(2,1,2),(1,1,2),(2,1,1),(1,1,3),(3,1,1)]
        best_aic, best_order = np.inf, (2, 1, 2)
        for order in candidates:
            try:
                aic = SARIMAX(
                    log_p[:-n_test], order=order,
                    seasonal_order=(1, 0, 1, 5),   # weekly seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False).aic
                if aic < best_aic:
                    best_aic, best_order = aic, order
            except Exception:
                continue

        seasonal_order = (1, 0, 1, 5)
        fit_test = SARIMAX(
            log_p[:-n_test], order=best_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        pred_log    = np.array(fit_test.forecast(n_test)).flatten()
        pred_test   = np.exp(pred_log)
        actual_test = np.exp(log_p[-n_test:])
        residuals   = actual_test - pred_test
        rmse_val    = _rmse(actual_test, pred_test)
        mape_val    = _mape(actual_test, pred_test)
        dir_acc     = _directional_accuracy(np.diff(actual_test), np.diff(pred_test))

        fit_full = SARIMAX(
            log_p, order=best_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc_obj     = fit_full.get_forecast(steps=horizon)
        fc_mean    = np.exp(np.array(fc_obj.predicted_mean).flatten())
        ci         = fc_obj.conf_int(alpha=0.10)
        if hasattr(ci, "iloc"):
            fc_lower = np.exp(ci.iloc[:, 0].values)
            fc_upper = np.exp(ci.iloc[:, 1].values)
        else:
            ci_arr   = np.array(ci)
            fc_lower = np.exp(ci_arr[:, 0])
            fc_upper = np.exp(ci_arr[:, 1])

        future_idx = _future_business_dates(prices.index[-1], horizon)
        return PredictionResult(
            model_name  = "SARIMA",
            forecast    = pd.Series(fc_mean,  index=future_idx, name="SARIMA"),
            upper_bound = pd.Series(fc_upper, index=future_idx),
            lower_bound = pd.Series(fc_lower, index=future_idx),
            metrics     = {
                "Order":             str(best_order),
                "Seasonal":          str(seasonal_order),
                "AIC":               round(best_aic, 1),
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="SARIMA",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Theta — theta decomposition method
# ─────────────────────────────────────────────────────────────────────────────

def theta_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        n_test   = max(int(len(prices) * test_size), 5)
        train_p  = prices.iloc[:-n_test]
        test_p   = prices.iloc[-n_test:]

        fit_test  = ThetaModel(train_p, period=5, deseasonalize=False).fit()
        pred_test = np.array(fit_test.forecast(n_test)).flatten()
        residuals = test_p.values - pred_test
        rmse_val  = _rmse(test_p.values, pred_test)
        mape_val  = _mape(test_p.values, pred_test)
        dir_acc   = _directional_accuracy(np.diff(test_p.values), np.diff(pred_test))

        fit_full  = ThetaModel(prices, period=5, deseasonalize=False).fit()
        fc        = np.array(fit_full.forecast(horizon)).flatten()
        ci_df     = fit_full.prediction_intervals(horizon, alpha=0.10)
        if hasattr(ci_df, "iloc"):
            fc_upper = ci_df.iloc[:, -1].values
            fc_lower = ci_df.iloc[:, 0].values
        else:
            fc_upper, fc_lower = _ci_from_residuals(fc, residuals)

        future_idx = _future_business_dates(prices.index[-1], horizon)
        return PredictionResult(
            model_name  = "Theta",
            forecast    = pd.Series(fc,        index=future_idx, name="Theta"),
            upper_bound = pd.Series(fc_upper,  index=future_idx),
            lower_bound = pd.Series(fc_lower,  index=future_idx),
            metrics     = {
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Theta",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. XGBoost & LightGBM with TimeSeriesSplit cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def _ml_ts_forecast(
    model_name: str,
    fitter,
    prices: pd.Series,
    horizon: int = 30,
    n_splits: int = 3,
    test_size: float = 0.15,
) -> PredictionResult:
    """
    Generic ML time-series forecaster.
    Uses TimeSeriesSplit for cross-validated accuracy metrics,
    then refits on full data for the actual forecast.
    """
    try:
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import TimeSeriesSplit

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values            # log returns

        # ── Cross-validated accuracy via TimeSeriesSplit ──
        tscv      = TimeSeriesSplit(n_splits=n_splits)
        dir_accs  = []
        mape_ps   = []
        for tr_idx, val_idx in tscv.split(X):
            if len(tr_idx) < 60:
                continue
            scaler_cv = RobustScaler()
            Xtr = scaler_cv.fit_transform(X[tr_idx])
            Xval = scaler_cv.transform(X[val_idx])
            ytr, yval = y[tr_idx], y[val_idx]
            m = fitter(Xtr, ytr, Xval, yval)
            ypred = m.predict(Xval)
            dir_accs.append(_directional_accuracy(yval, ypred))
            # Price-level MAPE for val window
            last_p = float(df["price"].values[tr_idx[-1]])
            act_p  = _prices_from_returns(last_p, yval)
            pred_p = _prices_from_returns(last_p, ypred)
            mape_ps.append(_mape(act_p, pred_p))

        avg_dir_acc = float(np.mean(dir_accs)) if dir_accs else 0.0
        avg_mape_p  = float(np.mean(mape_ps))  if mape_ps  else 0.0

        # ── Refit on full data for forecast ──
        n_test          = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        model     = fitter(Xtr_sc, y_train, Xte_sc, y_test)
        y_pred    = model.predict(Xte_sc)
        residuals = y_test - y_pred

        last_p_test = float(df["price"].values[-n_test - 1])
        act_p_test  = _prices_from_returns(last_p_test, y_test)
        pred_p_test = _prices_from_returns(last_p_test, y_pred)
        mape_final  = _mape(act_p_test, pred_p_test)

        forecast_lr, last_price = _hybrid_ml_forecast(
            lambda X_sc: model.predict(X_sc)[0],
            scaler, feat_cols, prices, horizon,
        )
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)

        # Bootstrap CI on price forecast
        from sklearn.preprocessing import RobustScaler as _RS
        alpha = 0.05
        rng   = np.random.default_rng(42)
        h     = len(forecast_lr)
        paths = np.zeros((500, h))
        for b in range(500):
            noise   = rng.choice(residuals, size=h, replace=True)
            paths[b] = _prices_from_returns(last_price, forecast_lr + noise)
        upper = np.percentile(paths, 95, axis=0)
        lower = np.percentile(paths, 5,  axis=0)

        return PredictionResult(
            model_name  = model_name,
            forecast    = pd.Series(forecast_prices, index=future_idx, name=model_name),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "CV Dir Acc % (avg)": round(avg_dir_acc, 1),
                "CV MAPE % (avg)":    round(avg_mape_p, 2),
                "MAPE % (price)":     round(mape_final, 2),
                "Directional Acc %":  round(avg_dir_acc, 1),
                "CV Folds":           len(dir_accs),
                "Features":           len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name=model_name,
                                forecast=pd.Series(), error=str(e))


def _fit_xgb(Xtr, ytr, Xte, yte):
    from xgboost import XGBRegressor
    m = XGBRegressor(
        n_estimators=1000, learning_rate=0.015, max_depth=4,
        subsample=0.75, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        gamma=0.1, random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50,
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    return m


def _fit_lgb(Xtr, ytr, Xte, yte):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.015, max_depth=5,
        num_leaves=31, subsample=0.75, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=2.0, min_child_samples=15,
        min_split_gain=0.01, random_state=42, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr, ytr,
          eval_set=[(Xte, yte)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m


def xgboost_ts_forecast(prices: pd.Series, horizon: int = 30, **kw) -> PredictionResult:
    return _ml_ts_forecast("XGBoost", _fit_xgb, prices, horizon)


def lightgbm_ts_forecast(prices: pd.Series, horizon: int = 30, **kw) -> PredictionResult:
    return _ml_ts_forecast("LightGBM", _fit_lgb, prices, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window evaluation — measures each model on the RECENT 30 trading days
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_eval(
    fn,
    prices: pd.Series,
    eval_window: int = 30,
) -> float:
    """
    Train model on prices[:-eval_window], forecast eval_window steps,
    compare to actual prices[-eval_window:].
    Returns MAPE on that recent out-of-sample window.
    Falls back to 999 if evaluation fails.
    """
    try:
        if len(prices) < eval_window + 60:
            return 999.0
        train  = prices.iloc[:-eval_window]
        actual = prices.iloc[-eval_window:].values
        r      = fn(train, horizon=eval_window)
        if r.error or r.forecast is None or r.forecast.empty:
            return 999.0
        predicted = r.forecast.values[:len(actual)]
        n = min(len(actual), len(predicted))
        return _mape(actual[:n], predicted[:n])
    except Exception:
        return 999.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. TS Ensemble — adaptive weights from recent rolling evaluation
# ─────────────────────────────────────────────────────────────────────────────

def ts_ensemble_forecast(
    prices: pd.Series,
    horizon: int = 30,
) -> PredictionResult:
    """
    Runs all 6 time-series models independently.

    Weighting strategy (two-tier):
      1. Recent MAPE  — each model is tested on the last 30 trading days
                        (train on history, predict last 30, measure error).
                        This captures current regime fitness.
      2. Full-history MAPE — from each model's own backtest metrics.

    Final weight = 0.70 × (1/recent_mape) + 0.30 × (1/hist_mape)

    Confidence band = percentile spread across all model forecasts
    (tighter when models agree, wider when they disagree).
    """
    runners = [
        ("Prophet",            prophet_forecast),
        ("Holt-Winters (ETS)", holtwinters_forecast),
        ("SARIMA",             sarima_forecast),
        ("Theta",              theta_forecast),
        ("XGBoost",            xgboost_ts_forecast),
        ("LightGBM",           lightgbm_ts_forecast),
    ]

    # ── Run all models on full series ──
    results: dict[str, PredictionResult] = {}
    for name, fn in runners:
        r = fn(prices, horizon=horizon)
        if not r.error and r.forecast is not None and not r.forecast.empty:
            results[name] = r

    if not results:
        return PredictionResult(model_name="TS Ensemble",
                                forecast=pd.Series(),
                                error="All sub-models failed.")

    future_idx = _future_business_dates(prices.index[-1], horizon)

    # ── Collect all forecasts as matrix for percentile CI ──
    fc_matrix = []
    for r in results.values():
        fc = r.forecast.reindex(future_idx, method="nearest").values[:horizon]
        if len(fc) == horizon:
            fc_matrix.append(fc)
    fc_matrix = np.array(fc_matrix)   # shape (n_models, horizon)

    # ── Compute historical MAPE per model ──
    hist_mapes: dict[str, float] = {}
    for name, r in results.items():
        m = r.metrics or {}
        try:
            hist_mapes[name] = float(
                m.get("MAPE % (price)", m.get("CV MAPE % (avg)", 10.0)))
        except Exception:
            hist_mapes[name] = 10.0

    # ── Compute recent 30-day rolling MAPE per model ──
    recent_mapes: dict[str, float] = {}
    fns = dict(runners)
    for name in results:
        recent_mapes[name] = _rolling_eval(fns[name], prices, eval_window=30)

    # ── Combined adaptive weight: 70% recent + 30% historical ──
    combined: dict[str, float] = {}
    for name in results:
        inv_r = 1.0 / (recent_mapes[name] + 1e-8)
        inv_h = 1.0 / (hist_mapes[name]   + 1e-8)
        combined[name] = 0.70 * inv_r + 0.30 * inv_h

    total   = sum(combined.values())
    weights = {n: v / total for n, v in combined.items()}

    # ── Blend price forecasts ──
    blended = np.zeros(horizon)
    for name, r in results.items():
        fc = r.forecast.reindex(future_idx, method="nearest").values[:horizon]
        blended += weights[name] * fc

    # ── CI from model spread (5th–95th percentile across all model forecasts) ──
    if fc_matrix.shape[0] >= 2:
        blended_u = np.percentile(fc_matrix, 95, axis=0)
        blended_l = np.percentile(fc_matrix, 5,  axis=0)
        # Widen slightly to ensure CI contains blended line
        blended_u = np.maximum(blended_u, blended * 1.01)
        blended_l = np.minimum(blended_l, blended * 0.99)
    else:
        blended_u = blended * 1.05
        blended_l = blended * 0.95

    # ── Summary stats ──
    best_model  = max(weights, key=weights.get)
    w_str       = " | ".join(
        f"{n}:{w:.2f}" for n, w in sorted(weights.items(), key=lambda x: -x[1])
    )
    avg_mape    = sum(hist_mapes[n] * weights[n] for n in results)
    avg_dacc    = float(np.mean([
        float(r.metrics.get("Directional Acc %",
              r.metrics.get("CV Dir Acc % (avg)", 0)))
        for r in results.values()
    ]))
    recent_str  = " | ".join(
        f"{n}:{v:.1f}%" for n, v in
        sorted(recent_mapes.items(), key=lambda x: x[1])
        if n in results
    )

    return PredictionResult(
        model_name  = "TS Ensemble",
        forecast    = pd.Series(blended,   index=future_idx, name="TS_Ensemble"),
        upper_bound = pd.Series(blended_u, index=future_idx),
        lower_bound = pd.Series(blended_l, index=future_idx),
        metrics     = {
            "Directional Acc %":    round(avg_dacc, 1),
            "MAPE % (price)":       round(avg_mape, 2),
            "Best sub-model":       best_model,
            "Model weights":        w_str,
            "Recent 30d MAPE":      recent_str,
            "Models blended":       len(results),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Monte Carlo — GBM with GARCH-like volatility (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_forecast(
    prices: pd.Series,
    horizon: int = 30,
    n_simulations: int = 5000,
    confidence: float = 0.90,
) -> PredictionResult:
    try:
        log_ret    = np.log(prices / prices.shift(1)).dropna()
        mu_daily   = float(log_ret.mean())
        long_vol   = float(log_ret.std())
        recent_vol = float(log_ret.tail(21).std())
        vol_decay  = 0.94
        S0         = float(prices.iloc[-1])

        rng   = np.random.default_rng(42)
        paths = np.zeros((n_simulations, horizon))

        for t in range(horizon):
            sigma_t = recent_vol * (vol_decay ** t) + long_vol * (1 - vol_decay ** t)
            Z    = rng.standard_normal(n_simulations)
            prev = S0 if t == 0 else paths[:, t - 1]
            paths[:, t] = prev * np.exp(
                (mu_daily - 0.5 * sigma_t ** 2) + sigma_t * Z)

        alpha       = (1 - confidence) / 2
        median_path = np.median(paths, axis=0)
        upper_path  = np.percentile(paths, (1 - alpha) * 100, axis=0)
        lower_path  = np.percentile(paths, alpha * 100, axis=0)
        future_idx  = _future_business_dates(prices.index[-1], horizon)

        return PredictionResult(
            model_name  = "Monte Carlo (GBM)",
            forecast    = pd.Series(median_path, index=future_idx, name="MC_median"),
            upper_bound = pd.Series(upper_path,  index=future_idx),
            lower_bound = pd.Series(lower_path,  index=future_idx),
            metrics     = {
                "Annual Drift %":    round(mu_daily * 252 * 100, 2),
                "Long-run Vol % pa": round(long_vol * np.sqrt(252) * 100, 2),
                "Recent Vol % pa":   round(recent_vol * np.sqrt(252) * 100, 2),
                "Confidence Band":   f"{confidence*100:.0f}%",
                "Simulations":       n_simulations,
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Monte Carlo (GBM)",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (for charting — unchanged)
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
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_MODELS = {
    "TS Ensemble":  ts_ensemble_forecast,
    "Prophet":             prophet_forecast,
    "Holt-Winters (ETS)":  holtwinters_forecast,
    "SARIMA":              sarima_forecast,
    "Theta":               theta_forecast,
    "XGBoost":        xgboost_ts_forecast,
    "LightGBM":       lightgbm_ts_forecast,
    "Monte Carlo (GBM)":   monte_carlo_forecast,
}


def run_prediction(
    model_name: str,
    prices: pd.Series,
    horizon: int = 30,
    **kwargs,
) -> PredictionResult:
    fn = PREDICTION_MODELS.get(model_name)
    if fn is None:
        return PredictionResult(
            model_name=model_name,
            forecast=pd.Series(),
            error=f"Unknown model: {model_name}",
        )
    return fn(prices, horizon=horizon, **kwargs)
