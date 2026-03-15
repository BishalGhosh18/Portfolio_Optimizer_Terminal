"""
Stock Price Prediction Engine
==============================
Models implemented:
  1. ARIMA          — Classical time-series model (statsmodels)
  2. Linear Regression — Feature-engineered trend model (sklearn)
  3. Random Forest  — Non-linear ensemble (sklearn)
  4. Monte Carlo    — Geometric Brownian Motion simulation
  5. Moving Average — Naive exponential-smoothing baseline

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
    forecast:     pd.Series           # predicted prices, DatetimeIndex
    upper_bound:  Optional[pd.Series] = None   # confidence / MC upper
    lower_bound:  Optional[pd.Series] = None   # confidence / MC lower
    metrics:      dict = field(default_factory=dict)
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _future_business_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Generate n future business dates (Mon-Fri) starting after last_date."""
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)
    return dates


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _build_features(prices: pd.Series) -> pd.DataFrame:
    """Build lag + technical-indicator features from a price series."""
    df = pd.DataFrame({"price": prices})
    # Lag features
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    # Moving averages
    for w in [5, 10, 21, 50]:
        df[f"ma_{w}"] = df["price"].rolling(w).mean()
    # Momentum
    df["mom_5"]  = df["price"].pct_change(5)
    df["mom_21"] = df["price"].pct_change(21)
    # Volatility (rolling std of returns)
    df["vol_21"] = df["price"].pct_change().rolling(21).std()
    # RSI (14)
    delta = df["price"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # Bollinger Band position
    ma20  = df["price"].rolling(20).mean()
    std20 = df["price"].rolling(20).std()
    df["bb_pos"] = (df["price"] - ma20) / (2 * std20 + 1e-8)
    # Time trend
    df["t"] = np.arange(len(df))
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def arima_forecast(
    prices: pd.Series,
    horizon: int = 30,
    order: tuple = (2, 1, 2),
    test_size: float = 0.1,
) -> PredictionResult:
    """
    Fit ARIMA on log-prices, forecast horizon trading days.
    Returns prices (exponentiated back from log-space).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA

        log_prices  = np.log(prices.values.astype(float))
        n_test      = max(int(len(log_prices) * test_size), 5)
        train_log   = log_prices[:-n_test]
        test_log    = log_prices[-n_test:]

        # Fit on training set
        model = ARIMA(train_log, order=order)
        fit   = model.fit()

        # In-sample metrics on test window
        test_pred_log = fit.forecast(steps=n_test)
        test_pred     = np.exp(test_pred_log)
        test_actual   = np.exp(test_log)
        rmse_val = _rmse(test_actual, test_pred)
        mape_val = _mape(test_actual, test_pred)

        # Refit on ALL data for final forecast
        model_full = ARIMA(log_prices, order=order)
        fit_full   = model_full.fit()
        fc         = fit_full.get_forecast(steps=horizon)
        fc_mean    = np.exp(fc.predicted_mean)
        ci         = fc.conf_int(alpha=0.10)
        fc_lower   = np.exp(ci.iloc[:, 0])
        fc_upper   = np.exp(ci.iloc[:, 1])

        future_idx = _future_business_dates(prices.index[-1], horizon)

        return PredictionResult(
            model_name  = "ARIMA",
            forecast    = pd.Series(fc_mean, index=future_idx, name="ARIMA"),
            upper_bound = pd.Series(fc_upper.values, index=future_idx),
            lower_bound = pd.Series(fc_lower.values, index=future_idx),
            metrics     = {"RMSE": round(rmse_val, 2), "MAPE (%)": round(mape_val, 2)},
        )
    except Exception as e:
        return PredictionResult(model_name="ARIMA", forecast=pd.Series(),
                                error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

def linear_regression_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    """
    Predicts price using lag/momentum features + linear regression.
    Walks forward one step at a time to build the forecast horizon.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error

        df = _build_features(prices)
        feat_cols = [c for c in df.columns if c != "price"]
        X, y = df[feat_cols].values, df["price"].values

        n_test  = max(int(len(X) * test_size), 10)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_sc, y_train)

        y_pred   = model.predict(X_te_sc)
        rmse_val = _rmse(y_test, y_pred)
        mape_val = _mape(y_test, y_pred)

        # Walk-forward forecast: extend price series step by step
        ext_prices = prices.copy()
        forecast_vals = []
        residual_std  = float(np.std(y_train - model.predict(X_tr_sc)))

        for _ in range(horizon):
            ext_df   = _build_features(ext_prices)
            last_row = ext_df[feat_cols].iloc[[-1]].values
            last_sc  = scaler.transform(last_row)
            pred_val = float(model.predict(last_sc)[0])
            pred_val = max(pred_val, 0.01)   # no negative prices
            forecast_vals.append(pred_val)
            next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
            ext_prices = pd.concat([ext_prices,
                                    pd.Series([pred_val], index=[next_date])])

        future_idx = _future_business_dates(prices.index[-1], horizon)
        fc_series  = pd.Series(forecast_vals, index=future_idx, name="LinReg")

        # Confidence band ≈ ±1.96 * residual_std (growing with horizon)
        std_band = pd.Series(
            [residual_std * np.sqrt(i + 1) * 1.96 for i in range(horizon)],
            index=future_idx,
        )
        return PredictionResult(
            model_name  = "Linear Regression",
            forecast    = fc_series,
            upper_bound = fc_series + std_band,
            lower_bound = (fc_series - std_band).clip(lower=0),
            metrics     = {"RMSE": round(rmse_val, 2), "MAPE (%)": round(mape_val, 2)},
        )
    except Exception as e:
        return PredictionResult(model_name="Linear Regression",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def random_forest_forecast(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
    n_estimators: int = 200,
) -> PredictionResult:
    """
    Random Forest regressor on lag + technical features.
    Walk-forward forecast for the horizon.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        df = _build_features(prices)
        feat_cols = [c for c in df.columns if c != "price"]
        X, y = df[feat_cols].values, df["price"].values

        n_test  = max(int(len(X) * test_size), 10)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=8,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr_sc, y_train)

        y_pred   = rf.predict(X_te_sc)
        rmse_val = _rmse(y_test, y_pred)
        mape_val = _mape(y_test, y_pred)

        # Walk-forward
        ext_prices = prices.copy()
        forecast_vals = []
        # Uncertainty: collect predictions from all trees per step
        upper_vals, lower_vals = [], []

        for _ in range(horizon):
            ext_df   = _build_features(ext_prices)
            last_row = ext_df[feat_cols].iloc[[-1]].values
            last_sc  = scaler.transform(last_row)
            # Individual tree predictions for uncertainty
            tree_preds = np.array([t.predict(last_sc)[0] for t in rf.estimators_])
            pred_val   = float(tree_preds.mean())
            pred_val   = max(pred_val, 0.01)
            forecast_vals.append(pred_val)
            upper_vals.append(float(np.percentile(tree_preds, 90)))
            lower_vals.append(max(float(np.percentile(tree_preds, 10)), 0.01))
            next_date  = _future_business_dates(ext_prices.index[-1], 1)[0]
            ext_prices = pd.concat([ext_prices,
                                    pd.Series([pred_val], index=[next_date])])

        future_idx = _future_business_dates(prices.index[-1], horizon)
        return PredictionResult(
            model_name  = "Random Forest",
            forecast    = pd.Series(forecast_vals, index=future_idx, name="RF"),
            upper_bound = pd.Series(upper_vals, index=future_idx),
            lower_bound = pd.Series(lower_vals, index=future_idx),
            metrics     = {"RMSE": round(rmse_val, 2), "MAPE (%)": round(mape_val, 2)},
        )
    except Exception as e:
        return PredictionResult(model_name="Random Forest",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Monte Carlo — Geometric Brownian Motion
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_forecast(
    prices: pd.Series,
    horizon: int = 30,
    n_simulations: int = 1000,
    confidence: float = 0.90,
) -> PredictionResult:
    """
    Simulate price paths using Geometric Brownian Motion.
    Returns the median path + confidence-interval band.
    """
    try:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu_daily    = float(log_returns.mean())
        sigma_daily = float(log_returns.std())
        S0          = float(prices.iloc[-1])

        rng   = np.random.default_rng(42)
        dt    = 1  # 1 trading day

        # Simulate n_simulations paths
        Z      = rng.standard_normal((n_simulations, horizon))
        drift  = (mu_daily - 0.5 * sigma_daily ** 2) * dt
        shock  = sigma_daily * np.sqrt(dt) * Z
        paths  = S0 * np.exp(np.cumsum(drift + shock, axis=1))

        alpha      = (1 - confidence) / 2
        median_path = np.median(paths, axis=0)
        upper_path  = np.percentile(paths, (1 - alpha) * 100, axis=0)
        lower_path  = np.percentile(paths, alpha * 100, axis=0)

        future_idx = _future_business_dates(prices.index[-1], horizon)
        return PredictionResult(
            model_name  = "Monte Carlo (GBM)",
            forecast    = pd.Series(median_path, index=future_idx, name="MC_median"),
            upper_bound = pd.Series(upper_path,  index=future_idx),
            lower_bound = pd.Series(lower_path,  index=future_idx),
            metrics     = {
                "Drift (ann.)":   round(mu_daily * 252 * 100, 2),
                "Volatility (ann.)": round(sigma_daily * np.sqrt(252) * 100, 2),
                f"Confidence": f"{confidence*100:.0f}%",
                "Simulations":  n_simulations,
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Monte Carlo (GBM)",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Exponential Moving Average baseline
# ─────────────────────────────────────────────────────────────────────────────

def ema_forecast(
    prices: pd.Series,
    horizon: int = 30,
    span: int = 21,
) -> PredictionResult:
    """
    Simple baseline: project the current EMA trend forward.
    Confidence band = ±1 std of recent residuals.
    """
    try:
        ema     = prices.ewm(span=span, adjust=False).mean()
        residuals = prices - ema
        std_res = float(residuals[-60:].std())

        last_ema   = float(ema.iloc[-1])
        daily_drift = float((ema.iloc[-1] - ema.iloc[-span]) / span)

        future_idx = _future_business_dates(prices.index[-1], horizon)
        fc_vals    = [last_ema + daily_drift * (i + 1) for i in range(horizon)]
        fc_series  = pd.Series(fc_vals, index=future_idx, name="EMA")

        return PredictionResult(
            model_name  = "EMA Trend",
            forecast    = fc_series,
            upper_bound = fc_series + 1.96 * std_res,
            lower_bound = (fc_series - 1.96 * std_res).clip(lower=0),
            metrics     = {"EMA span": span, "Residual Std": round(std_res, 2)},
        )
    except Exception as e:
        return PredictionResult(model_name="EMA Trend",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (for charting)
# ─────────────────────────────────────────────────────────────────────────────

def compute_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append common technical indicators to an OHLCV DataFrame.
    Expects columns: Open, High, Low, Close, Volume.
    """
    df = ohlcv.copy()
    close = df["Close"]

    # Moving averages
    df["MA_20"]  = close.rolling(20).mean()
    df["MA_50"]  = close.rolling(50).mean()
    df["MA_200"] = close.rolling(200).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2σ)
    df["BB_mid"]   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

    # ATR (14)
    if "High" in df.columns and "Low" in df.columns:
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - close.shift(1)).abs()
        lpc = (df["Low"]  - close.shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

    # Volume MA
    if "Volume" in df.columns:
        df["Vol_MA_20"] = df["Volume"].rolling(20).mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_MODELS = {
    "ARIMA":            arima_forecast,
    "Linear Regression":linear_regression_forecast,
    "Random Forest":    random_forest_forecast,
    "Monte Carlo (GBM)":monte_carlo_forecast,
    "EMA Trend":        ema_forecast,
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
