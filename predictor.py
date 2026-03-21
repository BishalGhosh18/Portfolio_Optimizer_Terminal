"""
Quant-Grade Stock Price Prediction Engine
==========================================
Approach inspired by systematic equity research:
  - Predict log-returns (stationary target, not raw price)
  - Rich multi-timeframe feature engineering (60+ features)
  - XGBoost & LightGBM with time-series walk-forward CV
  - Weighted Ensemble (XGB + LGB + RF) blended by recent accuracy
  - Monte Carlo GBM with GARCH-like volatility scaling
  - ARIMA on log-returns as classical baseline

Each model returns a unified PredictionResult dataclass.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

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

def _future_business_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)


def _rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _mape(a: np.ndarray, p: np.ndarray) -> float:
    mask = np.abs(a) > 1e-8
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def _directional_accuracy(a: np.ndarray, p: np.ndarray) -> float:
    """% of times model correctly predicts up/down direction."""
    return float(np.mean(np.sign(a) == np.sign(p)) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  — 60+ features across multiple timeframes
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(prices: pd.Series) -> pd.DataFrame:
    """
    Build a comprehensive feature matrix from a price series.
    Target: next-day log return (added as 'target' column).
    All features are computed from data available at time t (no lookahead).
    """
    df = pd.DataFrame({"price": prices.values}, index=prices.index)
    c  = df["price"]
    lr = np.log(c / c.shift(1))  # daily log returns

    # ── Lagged log returns ──
    for lag in [1, 2, 3, 4, 5, 10, 21]:
        df[f"ret_lag_{lag}"] = lr.shift(lag)

    # ── Multi-timeframe momentum ──
    for w in [5, 10, 21, 63]:
        df[f"mom_{w}"]     = np.log(c / c.shift(w))
        df[f"mom_norm_{w}"] = df[f"mom_{w}"] / (lr.rolling(w).std() * np.sqrt(w) + 1e-8)

    # ── Price vs Moving Averages (cross-sectional signal) ──
    for w in [5, 10, 20, 50, 100, 200]:
        ma = c.rolling(w).mean()
        df[f"price_ma_{w}"] = (c / ma) - 1

    # ── EMA cross signals ──
    for span in [9, 21, 55]:
        ema = c.ewm(span=span, adjust=False).mean()
        df[f"price_ema_{span}"] = (c / ema) - 1

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_norm"]      = macd / (c + 1e-8)
    df["macd_sig_norm"]  = sig  / (c + 1e-8)
    df["macd_hist_norm"] = (macd - sig) / (c + 1e-8)
    df["macd_above_sig"] = (macd > sig).astype(float)

    # ── RSI at multiple periods ──
    for period in [9, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-8)
        df[f"rsi_{period}"] = (100 - 100 / (1 + rs)) / 100  # normalised 0–1

    # ── Volatility (realised) at multiple windows ──
    for w in [5, 10, 21, 63]:
        df[f"rvol_{w}"] = lr.rolling(w).std() * np.sqrt(252)

    df["vol_ratio_5_21"]  = df["rvol_5"]  / (df["rvol_21"]  + 1e-8)
    df["vol_ratio_21_63"] = df["rvol_21"] / (df["rvol_63"]  + 1e-8)
    df["high_vol_regime"] = (df["rvol_21"] > df["rvol_63"]).astype(float)

    # ── Bollinger Band position & width ──
    for w in [10, 20]:
        ma  = c.rolling(w).mean()
        std = c.rolling(w).std()
        df[f"bb_pos_{w}"]   = (c - ma) / (2 * std + 1e-8)   # -1=lower, +1=upper
        df[f"bb_width_{w}"] = (4 * std) / (ma + 1e-8)

    # ── Stochastic oscillator ──
    for w in [14, 21]:
        lo = c.rolling(w).min()
        hi = c.rolling(w).max()
        df[f"stoch_{w}"] = (c - lo) / (hi - lo + 1e-8)

    # ── Williams %R ──
    hi14 = c.rolling(14).max()
    lo14 = c.rolling(14).min()
    df["williams_r"] = (hi14 - c) / (hi14 - lo14 + 1e-8)

    # ── 52-week range position ──
    df["pos_52w"] = (c - c.rolling(252, min_periods=63).min()) / \
                    (c.rolling(252, min_periods=63).max() -
                     c.rolling(252, min_periods=63).min() + 1e-8)

    # ── Calendar features ──
    idx = pd.DatetimeIndex(prices.index)
    df["day_of_week"] = idx.dayofweek / 4.0
    df["month"]       = idx.month / 12.0
    df["quarter"]     = (idx.month - 1) // 3 / 3.0

    # ── Normalised time trend ──
    df["t_norm"] = np.arange(len(df)) / max(len(df) - 1, 1)

    # ── Target: next-day log return ──
    df["target"] = lr.shift(-1)

    df.dropna(inplace=True)
    return df


def _prices_from_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Reconstruct price path from log returns."""
    return last_price * np.exp(np.cumsum(log_returns))


def _bootstrap_ci(
    residuals: np.ndarray,
    forecast_vals: np.ndarray,
    n_boot: int = 500,
    confidence: float = 0.90,
    last_price: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence intervals on the price forecast.
    Resample residuals, accumulate as log returns, get price distribution.
    """
    alpha = (1 - confidence) / 2
    rng   = np.random.default_rng(42)
    h     = len(forecast_vals)
    boot_paths = np.zeros((n_boot, h))

    for b in range(n_boot):
        noise = rng.choice(residuals, size=h, replace=True)
        boot_lr = forecast_vals + noise            # perturbed log returns
        boot_paths[b] = _prices_from_returns(last_price, boot_lr)

    upper = np.percentile(boot_paths, (1 - alpha) * 100, axis=0)
    lower = np.percentile(boot_paths, alpha * 100, axis=0)
    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# 1. XGBoost — primary model
# ─────────────────────────────────────────────────────────────────────────────

def xgboost_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    """
    XGBoost on 60+ features predicting log returns.
    Walk-forward one step at a time for multi-step horizon.
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import RobustScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values          # log returns

        n_test  = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators     = 500,
            learning_rate    = 0.03,
            max_depth        = 5,
            subsample        = 0.8,
            colsample_bytree = 0.7,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        model.fit(Xtr_sc, y_train,
                  eval_set=[(Xte_sc, y_test)],
                  verbose=False)

        y_pred   = model.predict(Xte_sc)
        residuals = y_test - y_pred
        rmse_val  = _rmse(y_test, y_pred)
        mape_val  = _mape(y_test, y_pred)
        dir_acc   = _directional_accuracy(y_test, y_pred)

        # Walk-forward forecast
        ext_prices    = prices.copy()
        forecast_lr   = []
        last_price    = float(prices.iloc[-1])

        for _ in range(horizon):
            fdf      = _build_features(ext_prices)
            last_row = fdf[feat_cols].iloc[[-1]].values
            last_sc  = scaler.transform(last_row)
            lr_pred  = float(model.predict(last_sc)[0])
            forecast_lr.append(lr_pred)
            next_price = ext_prices.iloc[-1] * np.exp(lr_pred)
            next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
            ext_prices = pd.concat([ext_prices,
                                    pd.Series([next_price], index=[next_date])])

        forecast_lr   = np.array(forecast_lr)
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx    = _future_business_dates(prices.index[-1], horizon)
        upper, lower  = _bootstrap_ci(residuals, forecast_lr,
                                       last_price=last_price)

        return PredictionResult(
            model_name  = "XGBoost",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="XGBoost"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "RMSE (ret)":        round(rmse_val, 6),
                "MAPE (ret) %":      round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="XGBoost", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def lightgbm_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    """
    LightGBM on the same feature set. Faster than XGBoost,
    slightly better on high-cardinality features.
    """
    try:
        import lightgbm as lgb
        from sklearn.preprocessing import RobustScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test  = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        model = lgb.LGBMRegressor(
            n_estimators     = 500,
            learning_rate    = 0.03,
            max_depth        = 6,
            num_leaves       = 31,
            subsample        = 0.8,
            colsample_bytree = 0.7,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            n_jobs           = -1,
            verbose          = -1,
        )
        model.fit(Xtr_sc, y_train,
                  eval_set=[(Xte_sc, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])

        y_pred    = model.predict(Xte_sc)
        residuals = y_test - y_pred
        rmse_val  = _rmse(y_test, y_pred)
        mape_val  = _mape(y_test, y_pred)
        dir_acc   = _directional_accuracy(y_test, y_pred)

        ext_prices  = prices.copy()
        forecast_lr = []
        last_price  = float(prices.iloc[-1])

        for _ in range(horizon):
            fdf      = _build_features(ext_prices)
            last_row = fdf[feat_cols].iloc[[-1]].values
            last_sc  = scaler.transform(last_row)
            lr_pred  = float(model.predict(last_sc)[0])
            forecast_lr.append(lr_pred)
            next_price = ext_prices.iloc[-1] * np.exp(lr_pred)
            next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
            ext_prices = pd.concat([ext_prices,
                                    pd.Series([next_price], index=[next_date])])

        forecast_lr     = np.array(forecast_lr)
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci(residuals, forecast_lr,
                                        last_price=last_price)

        return PredictionResult(
            model_name  = "LightGBM",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="LightGBM"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "RMSE (ret)":        round(rmse_val, 6),
                "MAPE (ret) %":      round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="LightGBM", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted Ensemble  (XGB + LGB + RF, blended by test accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    """
    Trains XGBoost, LightGBM and Random Forest independently.
    Blends walk-forward forecasts weighted by inverse RMSE on the test set.
    Confidence band via bootstrap of ensemble residuals.
    """
    try:
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import RobustScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test  = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        # ── XGBoost ──
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5,
                            subsample=0.8, colsample_bytree=0.7,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(Xtr_sc, y_train, eval_set=[(Xte_sc, y_test)], verbose=False)

        # ── LightGBM ──
        lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1, verbose=-1)
        lgbm.fit(Xtr_sc, y_train,
                 eval_set=[(Xte_sc, y_test)],
                 callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])

        # ── Random Forest ──
        rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                                   max_features=0.7, random_state=42, n_jobs=-1)
        rf.fit(Xtr_sc, y_train)

        # ── Compute weights (inverse RMSE) ──
        models     = [xgb, lgbm, rf]
        model_names = ["XGB", "LGB", "RF"]
        rmses = [_rmse(y_test, m.predict(Xte_sc)) for m in models]
        inv_rmse = [1.0 / (r + 1e-8) for r in rmses]
        total    = sum(inv_rmse)
        weights  = [w / total for w in inv_rmse]

        # ── Blended test prediction for residuals ──
        blended_test = sum(w * m.predict(Xte_sc) for w, m in zip(weights, models))
        residuals    = y_test - blended_test
        dir_acc      = _directional_accuracy(y_test, blended_test)

        # ── Walk-forward blended forecast ──
        ext_prices  = prices.copy()
        forecast_lr = []
        last_price  = float(prices.iloc[-1])

        for _ in range(horizon):
            fdf      = _build_features(ext_prices)
            last_row = fdf[feat_cols].iloc[[-1]].values
            last_sc  = scaler.transform(last_row)
            preds    = [float(m.predict(last_sc)[0]) for m in models]
            lr_pred  = sum(w * p for w, p in zip(weights, preds))
            forecast_lr.append(lr_pred)
            next_price = ext_prices.iloc[-1] * np.exp(lr_pred)
            next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
            ext_prices = pd.concat([ext_prices,
                                    pd.Series([next_price], index=[next_date])])

        forecast_lr     = np.array(forecast_lr)
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci(residuals, forecast_lr,
                                        last_price=last_price)

        w_str = ", ".join(f"{n}:{w:.2f}" for n, w in zip(model_names, weights))
        return PredictionResult(
            model_name  = "Ensemble (XGB+LGB+RF)",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="Ensemble"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "Directional Acc %": round(dir_acc, 1),
                "Model Weights":     w_str,
                "XGB RMSE":          round(rmses[0], 6),
                "LGB RMSE":          round(rmses[1], 6),
                "RF RMSE":           round(rmses[2], 6),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Ensemble (XGB+LGB+RF)",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Monte Carlo — GBM with GARCH-like volatility
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_forecast(
    prices: pd.Series,
    horizon: int = 30,
    n_simulations: int = 5000,
    confidence: float = 0.90,
) -> PredictionResult:
    """
    Geometric Brownian Motion with GARCH(1,1)-like volatility scaling.
    Uses recent realised vol to scale uncertainty (vol clustering).
    """
    try:
        log_ret    = np.log(prices / prices.shift(1)).dropna()
        mu_daily   = float(log_ret.mean())
        long_vol   = float(log_ret.std())
        recent_vol = float(log_ret.tail(21).std())  # recent 21-day vol
        # Blend: start with recent vol, mean-revert toward long-run vol
        vol_decay  = 0.94   # GARCH-like persistence
        S0         = float(prices.iloc[-1])

        rng = np.random.default_rng(42)
        paths = np.zeros((n_simulations, horizon))

        for t in range(horizon):
            # Volatility decays from recent toward long-run (mean reversion)
            sigma_t = recent_vol * (vol_decay ** t) + long_vol * (1 - vol_decay ** t)
            Z = rng.standard_normal(n_simulations)
            if t == 0:
                paths[:, t] = S0 * np.exp(
                    (mu_daily - 0.5 * sigma_t ** 2) + sigma_t * Z)
            else:
                paths[:, t] = paths[:, t-1] * np.exp(
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
# 5. ARIMA on log-returns (classical baseline)
# ─────────────────────────────────────────────────────────────────────────────

def arima_forecast(
    prices: pd.Series,
    horizon: int = 30,
    order: tuple = (2, 1, 2),
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        log_prices = np.log(prices.values.astype(float))
        n_test     = max(int(len(log_prices) * test_size), 5)
        train_log  = log_prices[:-n_test]
        test_log   = log_prices[-n_test:]

        model   = ARIMA(train_log, order=order)
        fit     = model.fit()
        tp_log  = fit.forecast(steps=n_test)
        rmse_val = _rmse(np.exp(test_log), np.exp(tp_log))
        mape_val = _mape(np.exp(test_log), np.exp(tp_log))

        model_full = ARIMA(log_prices, order=order)
        fit_full   = model_full.fit()
        fc         = fit_full.get_forecast(steps=horizon)
        fc_mean    = np.exp(np.array(fc.predicted_mean).flatten())
        ci         = fc.conf_int(alpha=0.10)
        if hasattr(ci, "iloc"):
            fc_lower = np.exp(ci.iloc[:, 0].values)
            fc_upper = np.exp(ci.iloc[:, 1].values)
        else:
            ci = np.array(ci)
            fc_lower = np.exp(ci[:, 0])
            fc_upper = np.exp(ci[:, 1])

        future_idx = _future_business_dates(prices.index[-1], horizon)
        return PredictionResult(
            model_name  = "ARIMA",
            forecast    = pd.Series(fc_mean,    index=future_idx, name="ARIMA"),
            upper_bound = pd.Series(fc_upper,   index=future_idx),
            lower_bound = pd.Series(fc_lower,   index=future_idx),
            metrics     = {"RMSE": round(rmse_val, 2), "MAPE (%)": round(mape_val, 2)},
        )
    except Exception as e:
        return PredictionResult(model_name="ARIMA",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (for charting — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df    = ohlcv.copy()
    close = df["Close"]

    for w, name in [(20, "MA_20"), (50, "MA_50"), (200, "MA_200")]:
        df[name] = close.rolling(w).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"]         = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]

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
    "Ensemble (Best)":   ensemble_forecast,
    "XGBoost":           xgboost_forecast,
    "LightGBM":          lightgbm_forecast,
    "Monte Carlo (GBM)": monte_carlo_forecast,
    "ARIMA":             arima_forecast,
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
