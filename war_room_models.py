"""
War Room Models — Additional prediction models for the War Room tab.
====================================================================
Adds 6 new models on top of the existing predictor.py:
  1. SVR (RBF kernel)         — sklearn support vector regression
  2. Extra Trees              — sklearn ExtraTreesRegressor
  3. CatBoost                 — gradient boosting with symmetric trees
  4. Stacked Ensemble ⭐      — Ridge meta-learner over XGB+LGB+ET+CB (5-fold OOF)
  5. Prophet                  — Facebook Prophet with RSI as extra regressor
  6. LSTM                     — TensorFlow/Keras LSTM (60-step sequences)

Rules:
  - Reuses _build_features(), _prices_from_returns(), _hybrid_ml_forecast(),
    _fit_xgb(), _fit_lgb(), PredictionResult from predictor.py
  - Never modifies predictor.py
  - Models are cached at module level to avoid retraining on page reruns
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

# ── Import shared building blocks from existing predictor.py ─────────────────
from predictor import (
    PredictionResult,
    _build_features,
    _prices_from_returns,
    _future_business_dates,
    _rmse, _mape, _directional_accuracy,
    _hybrid_ml_forecast,
    _fit_xgb, _fit_lgb,
    monte_carlo_forecast,
)

# Module-level model cache (key → fitted model dict)
_MODEL_CACHE: dict = {}

# Indian market holidays used as Prophet changepoints
_INDIA_HOLIDAYS = pd.DataFrame({
    "ds": pd.to_datetime([
        "2024-01-22","2024-01-26","2024-03-25","2024-04-14","2024-04-17",
        "2024-05-23","2024-06-17","2024-08-15","2024-10-02","2024-11-01",
        "2024-11-15","2024-12-25",
        "2025-01-26","2025-02-26","2025-03-14","2025-04-10","2025-04-14",
        "2025-04-18","2025-05-01","2025-06-07","2025-08-15","2025-10-02",
        "2025-10-21","2025-11-05","2025-12-25",
        "2026-01-26","2026-03-02","2026-03-25","2026-04-02","2026-04-14",
        "2026-05-01","2026-08-15","2026-10-02","2026-12-25",
    ]),
    "holiday": "India Market Holiday",
})


# ─────────────────────────────────────────────────────────────────────────────
# Shared bootstrap CI helper
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci_wr(
    residuals: np.ndarray,
    forecast_lr: np.ndarray,
    last_price: float,
    n_boot: int = 500,
    confidence: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1 - confidence) / 2
    rng   = np.random.default_rng(42)
    h     = len(forecast_lr)
    paths = np.zeros((n_boot, h))
    for b in range(n_boot):
        noise    = rng.choice(residuals, size=h, replace=True)
        paths[b] = _prices_from_returns(last_price, forecast_lr + noise)
    upper = np.percentile(paths, (1 - alpha) * 100, axis=0)
    lower = np.percentile(paths, alpha * 100, axis=0)
    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# Identity scaler — used for models that have their own internal scaler
# ─────────────────────────────────────────────────────────────────────────────

class _IdentityScaler:
    def fit_transform(self, X): return np.array(X)
    def transform(self, X):     return np.array(X)
    def fit(self, X):           return self


# ─────────────────────────────────────────────────────────────────────────────
# 1. SVR (RBF kernel)
# ─────────────────────────────────────────────────────────────────────────────

def svr_war_room(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    try:
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test  = max(int(len(X) * test_size), 20)
        n_train = min(600, len(X) - n_test)      # SVR is O(n²) — cap training size

        X_train = X[-n_test - n_train:-n_test]
        y_train = y[-n_test - n_train:-n_test]
        X_test  = X[-n_test:]
        y_test  = y[-n_test:]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=100.0, epsilon=0.001, gamma="scale")),
        ])
        pipe.fit(X_train, y_train)

        y_pred    = pipe.predict(X_test)
        residuals = y_test - y_pred
        dir_acc   = _directional_accuracy(y_test, y_pred)
        last_p0   = float(df["price"].values[-n_test - 1])
        mape_val  = _mape(
            _prices_from_returns(last_p0, y_test),
            _prices_from_returns(last_p0, y_pred),
        )

        # Walk-forward with identity scaler (pipeline handles scaling internally)
        forecast_lr, last_price = _hybrid_ml_forecast(
            lambda X_sc: pipe.predict(X_sc)[0],
            _IdentityScaler(), feat_cols, prices, horizon,
        )
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci_wr(residuals, forecast_lr, last_price)

        return PredictionResult(
            model_name  = "SVR (RBF)",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="SVR"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "Directional Acc %": round(dir_acc, 1),
                "MAPE % (price)":    round(mape_val, 2),
                "Train samples":     n_train,
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="SVR (RBF)", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extra Trees Regressor
# ─────────────────────────────────────────────────────────────────────────────

def extra_trees_war_room(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import RobustScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test          = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        model = ExtraTreesRegressor(
            n_estimators=300, max_depth=10, max_features=0.7,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        model.fit(Xtr_sc, y_train)

        y_pred    = model.predict(Xte_sc)
        residuals = y_test - y_pred
        dir_acc   = _directional_accuracy(y_test, y_pred)
        last_p0   = float(df["price"].values[-n_test - 1])
        mape_val  = _mape(
            _prices_from_returns(last_p0, y_test),
            _prices_from_returns(last_p0, y_pred),
        )

        # Top 5 feature importances
        imp     = model.feature_importances_
        top5    = sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:5]
        top5_str = ", ".join(f"{n}:{v:.3f}" for n, v in top5)

        forecast_lr, last_price = _hybrid_ml_forecast(
            lambda X_sc: model.predict(X_sc)[0],
            scaler, feat_cols, prices, horizon,
        )
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci_wr(residuals, forecast_lr, last_price)

        return PredictionResult(
            model_name  = "Extra Trees",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="ET"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "Directional Acc %": round(dir_acc, 1),
                "MAPE % (price)":    round(mape_val, 2),
                "Top features":      top5_str,
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Extra Trees", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CatBoost
# ─────────────────────────────────────────────────────────────────────────────

def catboost_war_room(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
) -> PredictionResult:
    try:
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import RobustScaler

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test          = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        model = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=8,
            verbose=0, random_seed=42, early_stopping_rounds=50,
        )
        model.fit(Xtr_sc, y_train, eval_set=(Xte_sc, y_test))

        y_pred    = model.predict(Xte_sc)
        residuals = y_test - y_pred
        dir_acc   = _directional_accuracy(y_test, y_pred)
        last_p0   = float(df["price"].values[-n_test - 1])
        mape_val  = _mape(
            _prices_from_returns(last_p0, y_test),
            _prices_from_returns(last_p0, y_pred),
        )

        forecast_lr, last_price = _hybrid_ml_forecast(
            lambda X_sc: model.predict(X_sc)[0],
            scaler, feat_cols, prices, horizon,
        )
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci_wr(residuals, forecast_lr, last_price)

        return PredictionResult(
            model_name  = "CatBoost",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="CatBoost"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "Directional Acc %": round(dir_acc, 1),
                "MAPE % (price)":    round(mape_val, 2),
                "Iterations used":   int(model.best_iteration_) if hasattr(model, "best_iteration_") else 500,
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="CatBoost", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stacked Ensemble ⭐ BEST
# ─────────────────────────────────────────────────────────────────────────────

def stacked_ensemble_war_room(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.15,
    n_splits: int = 3,
) -> PredictionResult:
    """
    Base learners: XGBoost + LightGBM + Extra Trees + CatBoost.
    Meta-learner: Ridge trained on TimeSeriesSplit OOF predictions.
    """
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import TimeSeriesSplit

        # Attempt to import CatBoost; fall back to GBM if unavailable
        try:
            from catboost import CatBoostRegressor as _CB
            _catboost_ok = True
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor as _CB
            _catboost_ok = False

        df        = _build_features(prices)
        feat_cols = [c for c in df.columns if c not in ("price", "target")]
        X = df[feat_cols].values
        y = df["target"].values

        n_test          = max(int(len(X) * test_size), 20)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        scaler  = RobustScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        # ── Build OOF stacking matrix from training set ──
        n_base   = 4
        oof_preds = np.zeros((len(X_train), n_base))
        tscv      = TimeSeriesSplit(n_splits=n_splits)

        for tr_idx, val_idx in tscv.split(Xtr_sc):
            if len(tr_idx) < 30:
                continue
            X_f, y_f = Xtr_sc[tr_idx], y_train[tr_idx]
            X_v, y_v = Xtr_sc[val_idx], y_train[val_idx]

            # XGBoost
            m0 = _fit_xgb(X_f, y_f, X_v, y_v)
            oof_preds[val_idx, 0] = m0.predict(X_v)

            # LightGBM
            m1 = _fit_lgb(X_f, y_f, X_v, y_v)
            oof_preds[val_idx, 1] = m1.predict(X_v)

            # Extra Trees
            m2 = ExtraTreesRegressor(n_estimators=200, max_depth=8,
                                     random_state=42, n_jobs=-1)
            m2.fit(X_f, y_f)
            oof_preds[val_idx, 2] = m2.predict(X_v)

            # CatBoost / GBM fallback
            if _catboost_ok:
                m3 = _CB(iterations=300, learning_rate=0.05, depth=6,
                         verbose=0, random_seed=42)
                m3.fit(X_f, y_f)
            else:
                m3 = _CB(n_estimators=200, learning_rate=0.05,
                         max_depth=6, random_state=42)
                m3.fit(X_f, y_f)
            oof_preds[val_idx, 3] = m3.predict(X_v)

        # Ridge meta-learner on OOF predictions
        # Only fit on rows where OOF was populated
        filled = np.any(oof_preds != 0, axis=1)
        ridge  = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(oof_preds[filled], y_train[filled])

        # ── Refit all base models on full training data ──
        xgb_f = _fit_xgb(Xtr_sc, y_train, Xte_sc, y_test)
        lgb_f = _fit_lgb(Xtr_sc, y_train, Xte_sc, y_test)
        et_f  = ExtraTreesRegressor(n_estimators=300, max_depth=10,
                                    random_state=42, n_jobs=-1)
        et_f.fit(Xtr_sc, y_train)

        if _catboost_ok:
            cb_f = _CB(iterations=500, learning_rate=0.05, depth=8,
                       verbose=0, random_seed=42, early_stopping_rounds=50)
            cb_f.fit(Xtr_sc, y_train, eval_set=(Xte_sc, y_test))
        else:
            cb_f = _CB(n_estimators=300, max_depth=6, random_state=42)
            cb_f.fit(Xtr_sc, y_train)

        base_models  = [xgb_f, lgb_f, et_f, cb_f]
        base_names   = ["XGB", "LGB", "ET", "CB"]

        # Test-set metrics
        test_meta    = np.column_stack([m.predict(Xte_sc) for m in base_models])
        test_pred    = ridge.predict(test_meta)
        residuals    = y_test - test_pred
        dir_acc      = _directional_accuracy(y_test, test_pred)
        last_p0      = float(df["price"].values[-n_test - 1])
        mape_val     = _mape(
            _prices_from_returns(last_p0, y_test),
            _prices_from_returns(last_p0, test_pred),
        )

        weights_str  = " | ".join(
            f"{n}:{w:+.3f}" for n, w in zip(base_names, ridge.coef_))

        # ── Walk-forward forecast via stacked predict ──
        def stacked_predict(X_sc: np.ndarray) -> float:
            preds = np.array([float(m.predict(X_sc)[0]) for m in base_models])
            return float(ridge.predict(preds.reshape(1, -1))[0])

        forecast_lr, last_price = _hybrid_ml_forecast(
            stacked_predict, scaler, feat_cols, prices, horizon,
        )
        forecast_prices = _prices_from_returns(last_price, forecast_lr)
        future_idx      = _future_business_dates(prices.index[-1], horizon)
        upper, lower    = _bootstrap_ci_wr(residuals, forecast_lr, last_price)

        return PredictionResult(
            model_name  = "Stacked Ensemble",
            forecast    = pd.Series(forecast_prices, index=future_idx, name="Stacked"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "Directional Acc %": round(dir_acc, 1),
                "MAPE % (price)":    round(mape_val, 2),
                "Meta weights":      weights_str,
                "CV folds":          n_splits,
                "Features":          len(feat_cols),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Stacked Ensemble",
                                forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prophet with RSI as extra regressor
# ─────────────────────────────────────────────────────────────────────────────

def prophet_war_room(
    prices: pd.Series,
    horizon: int = 30,
    test_size: float = 0.10,
) -> PredictionResult:
    try:
        from prophet import Prophet

        # Compute RSI-14 as extra regressor
        delta  = prices.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_14 = (100 - 100 / (1 + gain / (loss + 1e-8))).fillna(50)

        # Strip timezone for Prophet
        idx = prices.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)

        df_full = pd.DataFrame({
            "ds":  idx,
            "y":   prices.values,
            "rsi": rsi_14.values,
        }).dropna()

        n_test   = max(int(len(df_full) * test_size), 5)
        df_train = df_full.iloc[:-n_test]
        df_test  = df_full.iloc[-n_test:]

        def _make_prophet():
            m = Prophet(
                changepoint_prior_scale  = 0.05,
                seasonality_prior_scale  = 10.0,
                holidays                 = _INDIA_HOLIDAYS,
                daily_seasonality        = False,
                weekly_seasonality       = True,
                yearly_seasonality       = True,
                uncertainty_samples      = 300,
                interval_width           = 0.90,
            )
            m.add_regressor("rsi")
            return m

        m_test = _make_prophet()
        m_test.fit(df_train[["ds", "y", "rsi"]])
        pred_test  = m_test.predict(df_test[["ds", "rsi"]])["yhat"].values
        residuals  = df_test["y"].values - pred_test
        rmse_val   = _rmse(df_test["y"].values, pred_test)
        mape_val   = _mape(df_test["y"].values, pred_test)
        dir_acc    = _directional_accuracy(
            np.diff(df_test["y"].values), np.diff(pred_test))

        # Refit on full data
        m_full = _make_prophet()
        m_full.fit(df_full[["ds", "y", "rsi"]])

        future_idx  = _future_business_dates(prices.index[-1], horizon)
        future_ds   = future_idx.tz_localize(None) if future_idx.tz is None else future_idx
        # Carry forward last RSI value for future
        future_rsi  = float(rsi_14.iloc[-1])
        future_df   = pd.DataFrame({"ds": future_ds, "rsi": future_rsi})
        fc          = m_full.predict(future_df)

        return PredictionResult(
            model_name  = "Prophet",
            forecast    = pd.Series(fc["yhat"].values,       index=future_idx, name="Prophet"),
            upper_bound = pd.Series(fc["yhat_upper"].values, index=future_idx),
            lower_bound = pd.Series(fc["yhat_lower"].values, index=future_idx),
            metrics     = {
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
                "Extra regressor":   "RSI-14",
                "Holidays added":    len(_INDIA_HOLIDAYS),
            },
        )
    except Exception as e:
        return PredictionResult(model_name="Prophet", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM (TensorFlow/Keras)
# ─────────────────────────────────────────────────────────────────────────────

def lstm_war_room(
    prices: pd.Series,
    horizon: int = 30,
    window: int = 60,
    cache_key: Optional[str] = None,
) -> PredictionResult:
    """
    LSTM on 8-feature sequences of 60 timesteps.
    Model is cached in _MODEL_CACHE to avoid retraining on reruns.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler

        tf.get_logger().setLevel("ERROR")

        # ── Build 8-feature matrix from price series ──
        c  = prices.astype(float)
        lr = np.log(c / c.shift(1))

        feat_df = pd.DataFrame(index=c.index)
        feat_df["close"]       = c
        feat_df["ema9"]        = c.ewm(span=9,  adjust=False).mean()
        feat_df["ema21"]       = c.ewm(span=21, adjust=False).mean()
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feat_df["rsi"]         = 100 - 100 / (1 + gain / (loss + 1e-8))
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        feat_df["macd"]        = ema12 - ema26
        bb_ma  = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        feat_df["bb_pos"]      = (c - bb_ma) / (2 * bb_std + 1e-8)
        feat_df["rvol_21"]     = lr.rolling(21).std() * np.sqrt(252)
        feat_df["drawdown_20"] = (c - c.rolling(20).max()) / (c.rolling(20).max() + 1e-8)
        feat_df.dropna(inplace=True)

        n_features   = feat_df.shape[1]
        feat_values  = feat_df.values.astype(np.float32)
        price_values = feat_df["close"].values.astype(np.float32)

        # Scale features
        scaler   = MinMaxScaler()
        feat_sc  = scaler.fit_transform(feat_values)

        # Scale target (close price only, column 0)
        p_scaler = MinMaxScaler()
        p_sc     = p_scaler.fit_transform(price_values.reshape(-1, 1))

        # Create sequences
        def _make_sequences(feat, target, w):
            X_seq, y_seq = [], []
            for i in range(w, len(feat)):
                X_seq.append(feat[i - w:i])
                y_seq.append(target[i, 0])
            return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

        X_seq, y_seq = _make_sequences(feat_sc, p_sc, window)

        n_test   = max(int(len(X_seq) * 0.10), 5)
        X_train  = X_seq[:-n_test]
        y_train  = y_seq[:-n_test]
        X_test   = X_seq[-n_test:]
        y_test   = y_seq[-n_test:]

        # ── Load or train LSTM model ──
        ck = cache_key or f"lstm_{len(prices)}"
        if ck in _MODEL_CACHE:
            model = _MODEL_CACHE[ck]["model"]
        else:
            model = Sequential([
                LSTM(128, return_sequences=True,
                     input_shape=(window, n_features)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(monitor="val_loss", patience=10,
                               restore_best_weights=True, verbose=0)
            model.fit(
                X_train, y_train,
                epochs=60, batch_size=32, verbose=0,
                validation_split=0.1, callbacks=[es],
            )
            _MODEL_CACHE[ck] = {"model": model, "scaler": scaler, "p_scaler": p_scaler}

        # ── Evaluate on test set ──
        y_pred_sc = model.predict(X_test, verbose=0).flatten()
        y_pred_p  = p_scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
        y_test_p  = p_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse_val  = _rmse(y_test_p, y_pred_p)
        mape_val  = _mape(y_test_p, y_pred_p)
        dir_acc   = _directional_accuracy(np.diff(y_test_p), np.diff(y_pred_p))
        residuals = y_test_p - y_pred_p

        # ── Walk-forward forecast ──
        last_seq      = feat_sc[-window:].copy()   # shape (window, n_features)
        forecast_prices = []
        last_price    = float(price_values[-1])
        current_feats = feat_sc[-1].copy()         # shape (n_features,)

        for _ in range(horizon):
            seq_input   = last_seq.reshape(1, window, n_features)
            pred_sc     = float(model.predict(seq_input, verbose=0)[0, 0])
            pred_price  = float(p_scaler.inverse_transform([[pred_sc]])[0, 0])
            forecast_prices.append(pred_price)

            # Build next feature vector (simplified: update close-derived features)
            prev_close   = float(p_scaler.inverse_transform([[last_seq[-1, 0]]])[0, 0])
            new_close_sc = pred_sc
            new_feat     = current_feats.copy()
            new_feat[0]  = new_close_sc           # close (scaled)
            last_seq     = np.vstack([last_seq[1:], new_feat])

        forecast_arr = np.array(forecast_prices)
        future_idx   = _future_business_dates(prices.index[-1], horizon)

        # CI from residuals
        std_r   = float(np.std(residuals)) if len(residuals) > 1 else last_price * 0.01
        t_arr   = np.arange(1, horizon + 1)
        upper   = forecast_arr + 1.645 * std_r * np.sqrt(t_arr)
        lower   = np.maximum(forecast_arr - 1.645 * std_r * np.sqrt(t_arr),
                             forecast_arr * 0.5)

        return PredictionResult(
            model_name  = "LSTM",
            forecast    = pd.Series(forecast_arr, index=future_idx, name="LSTM"),
            upper_bound = pd.Series(upper, index=future_idx),
            lower_bound = pd.Series(lower, index=future_idx),
            metrics     = {
                "RMSE (price)":      round(rmse_val, 2),
                "MAPE % (price)":    round(mape_val, 2),
                "Directional Acc %": round(dir_acc, 1),
                "Window":            window,
                "Features":          n_features,
            },
        )
    except ImportError:
        return PredictionResult(
            model_name="LSTM",
            forecast=pd.Series(),
            error="TensorFlow not installed. Run: pip install tensorflow>=2.16.0",
        )
    except Exception as e:
        return PredictionResult(model_name="LSTM", forecast=pd.Series(), error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Registry and batch runner
# ─────────────────────────────────────────────────────────────────────────────

WAR_ROOM_MODELS = {
    "Stacked Ensemble ⭐": stacked_ensemble_war_room,
    "CatBoost":            catboost_war_room,
    "Extra Trees":         extra_trees_war_room,
    "SVR (RBF)":           svr_war_room,
    "Prophet":             prophet_war_room,
    "LSTM":                lstm_war_room,
}

MODEL_ICONS = {
    "Stacked Ensemble ⭐": "⭐",
    "CatBoost":            "🐱",
    "Extra Trees":         "🌲",
    "SVR (RBF)":           "📐",
    "Prophet":             "🔮",
    "LSTM":                "🧠",
}


def run_all_war_room(
    prices: pd.Series,
    horizon: int = 30,
    cache_key: Optional[str] = None,
    include_monte_carlo: bool = True,
) -> dict[str, PredictionResult]:
    """Run all War Room models and return {model_name: PredictionResult}."""
    results = {}
    for name, fn in WAR_ROOM_MODELS.items():
        kw = {"cache_key": cache_key} if name == "LSTM" else {}
        results[name] = fn(prices, horizon=horizon, **kw)
    if include_monte_carlo:
        results["Monte Carlo"] = monte_carlo_forecast(prices, horizon=horizon)
    return results
