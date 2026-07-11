"""
Stock Movement Predictor
========================
Predicts next-day *direction* (up / down) rather than the exact price — the
classic finance-ML entry point.

Pipeline (mirrors the reference approach):
  1. DATA        — historical OHLCV (pulled with yfinance upstream)
  2. FEATURES    — lagged returns, moving-average ratios, volatility, RSI,
                   MACD, momentum, volume ratio
  3. MODEL       — train a classifier (Logistic Regression / Random Forest /
                   XGBoost) to predict whether the stock closes up tomorrow
  4. BACKTEST    — trade the signal and compare against buy & hold + random
  5. EVALUATION  — accuracy, precision, recall, F1 + strategy returns

Everything is chronological (no shuffling / look-ahead): the test window is the
most-recent slice the model never saw during training.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:                                     # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:                                     # pragma: no cover
    _HAS_LGBM = False

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
PRICE_FEATURE_NAMES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "vol_5", "vol_10", "vol_20",
    "rsi_14", "macd_hist", "mom_10",
    # ── volume / buy-sell pressure (inferred from OHLCV) ──
    "vol_ratio", "obv_slope", "mfi_14", "ad_slope",
]
# Populated per-call to include earnings + market-regime features when a ticker
# is supplied. Kept as a module global so downstream code can read the columns.
FEATURE_NAMES = list(PRICE_FEATURE_NAMES)


def engineer_features(ohlcv: pd.DataFrame, ticker: str | None = None,
                      use_fundamentals: bool = True) -> pd.DataFrame:
    """Build the model matrix + next-day direction target from OHLCV data.

    Returns a DataFrame with the FEATURE_NAMES columns plus:
      • ``ret_next``   next-day simple return (used for backtesting)
      • ``target``     1 if the stock closes up tomorrow, else 0
      • ``close``      that day's close (used to price the forecast)
    Only warm-up rows (NaN features) are dropped — the final row is *kept* so it
    can be used to forecast tomorrow (its ``target``/``ret_next`` stay NaN).
    """
    df    = ohlcv.copy()
    close = df["Close"].astype(float)
    ret   = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # Lagged returns — momentum / mean-reversion signal
    for k in (1, 2, 3, 5, 10):
        feat[f"ret_{k}"] = ret.shift(0).rolling(k).sum() if k > 1 else ret

    # Price relative to its moving averages (trend)
    for w in (5, 10, 20):
        feat[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0

    # Rolling volatility (risk regime)
    for w in (5, 10, 20):
        feat[f"vol_{w}"] = ret.rolling(w).std()

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feat["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # 10-day momentum
    feat["mom_10"] = close / close.shift(10) - 1.0

    # ── Volume / buy-sell pressure (proxied from OHLCV) ───────────────────────
    # Free OHLCV has no true tick-level bought-vs-sold split, so we infer buying
    # vs selling pressure from where price closes within its range × volume.
    if "Volume" in df.columns and df["Volume"].notna().any():
        vol  = df["Volume"].astype(float).fillna(0.0)
        high = df["High"].astype(float) if "High" in df.columns else close
        low  = df["Low"].astype(float)  if "Low"  in df.columns else close

        # participation vs its own 20-day average
        feat["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-8) - 1.0

        # On-Balance Volume — volume added on up-days, removed on down-days;
        # its slope = net accumulation. Normalised by 20d avg volume.
        obv = (np.sign(close.diff().fillna(0.0)) * vol).cumsum()
        feat["obv_slope"] = (obv - obv.shift(10)) / (vol.rolling(20).mean() * 10 + 1e-8)

        # Money-Flow Index (14) — RSI computed on price×volume money flow.
        tp        = (high + low + close) / 3.0
        mf        = tp * vol
        pos_mf    = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
        neg_mf    = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
        feat["mfi_14"] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-8))

        # Accumulation/Distribution — close position within the H-L range × volume;
        # its slope tells accumulation (buying) vs distribution (selling).
        clv = ((close - low) - (high - close)) / ((high - low).replace(0, np.nan))
        ad  = (clv.fillna(0.0) * vol).cumsum()
        feat["ad_slope"] = (ad - ad.shift(10)) / (vol.rolling(20).mean() * 10 + 1e-8)
    else:
        feat["vol_ratio"] = feat["obv_slope"] = feat["mfi_14"] = feat["ad_slope"] = 0.0

    feature_cols = list(PRICE_FEATURE_NAMES)

    # ── Backtestable fundamentals: earnings-cycle + market regime (Nifty/VIX) ──
    if use_fundamentals and ticker:
        try:
            from fundamentals import (
                earnings_features, market_features,
                EARNINGS_FEATURE_NAMES, MARKET_FEATURE_NAMES,
            )
            ef = earnings_features(ticker, feat.index)
            mf = market_features(feat.index)
            for col in EARNINGS_FEATURE_NAMES:
                feat[col] = ef[col].values
            for col in MARKET_FEATURE_NAMES:
                feat[col] = mf[col].values
            feature_cols += EARNINGS_FEATURE_NAMES + MARKET_FEATURE_NAMES
        except Exception:
            pass                                         # stay price-only on failure

    # Expose the active feature set for downstream consumers.
    global FEATURE_NAMES
    FEATURE_NAMES = feature_cols

    # Target: does tomorrow close higher than today? (NaN on the final row)
    feat["ret_next"] = ret.shift(-1)
    feat["target"]   = np.where(feat["ret_next"].isna(), np.nan,
                                (feat["ret_next"] > 0).astype(float))
    feat["close"]    = close

    # Drop warm-up rows only (NaN features); keep the final row for forecasting.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=feature_cols + ["close"])
    feat.attrs["feature_cols"] = feature_cols
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 3./5. MODELS + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MovementResult:
    name:        str
    # test-window predictions
    test_dates:  pd.DatetimeIndex
    y_true:      np.ndarray
    y_pred:      np.ndarray
    proba:       np.ndarray
    # classification metrics
    accuracy:    float
    precision:   float
    recall:      float
    f1:          float
    confusion:   np.ndarray
    # backtest (strategy that goes long when the model predicts "up")
    equity:      pd.Series               # cumulative strategy value (starts at 1)
    strat_return: float                  # total strategy return over test window
    strat_sharpe: float
    win_rate:    float
    # tomorrow's live call (model retrained on ALL labelled data)
    next_up_prob: float
    next_signal:  str
    next_exp_return: float = 0.0         # expected next-day return (fraction)
    next_price:      float = 0.0         # expected next-day close (₹)
    last_close:      float = 0.0
    feat_importance: dict = field(default_factory=dict)
    error:       Optional[str] = None


def _build_model(name: str):
    """Model lineup chosen for tabular financial direction:
       gradient-boosted trees (best on tabular) + a linear baseline."""
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
        ])
    if name == "LightGBM" and _HAS_LGBM:
        return LGBMClassifier(
            n_estimators=400, num_leaves=15, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    if name == "XGBoost" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Unknown / unavailable model: {name}")


def _importance_vals(model, name: str) -> Optional[np.ndarray]:
    try:
        if name == "Logistic Regression":
            vals = np.abs(model.named_steps["clf"].coef_[0])
        elif name in ("LightGBM", "XGBoost"):
            vals = model.feature_importances_
        else:
            return None
        vals = np.asarray(vals, dtype=float)
        return vals / vals.sum() if vals.sum() > 0 else vals
    except Exception:
        return None


def _assemble(name: str, y_pred, proba, y_te, r_te, test_index,
              next_prob: float, ret_next: pd.Series, last_close: float,
              feature_cols: list[str], imp_vals: Optional[np.ndarray]) -> MovementResult:
    """Build a MovementResult from raw predictions + a live probability.
    Shared by single models and the soft-voting ensemble."""
    y_pred = np.asarray(y_pred).astype(int)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    cm   = confusion_matrix(y_te, y_pred, labels=[0, 1])

    # Backtest: long (1) when predicted up, flat (0) otherwise
    pos       = pd.Series(y_pred, index=test_index).astype(float)
    strat_ret = pos * r_te
    equity    = (1.0 + strat_ret).cumprod()
    total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sd        = strat_ret.std()
    sharpe    = float(strat_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    traded    = strat_ret[pos > 0]
    win_rate  = float((traded > 0).mean() * 100) if len(traded) else 0.0

    next_signal = "UP" if next_prob >= 0.5 else "DOWN"

    # Direction probability → expected price move (× historical avg up/down move)
    up_moves = ret_next[ret_next > 0]
    dn_moves = ret_next[ret_next <= 0]
    avg_up   = float(up_moves.mean()) if len(up_moves) else 0.0
    avg_dn   = float(dn_moves.mean()) if len(dn_moves) else 0.0
    exp_ret  = next_prob * avg_up + (1 - next_prob) * avg_dn

    imp = {}
    if imp_vals is not None and len(imp_vals) == len(feature_cols):
        imp = dict(sorted(zip(feature_cols, imp_vals), key=lambda x: -x[1]))

    return MovementResult(
        name=name, test_dates=test_index, y_true=np.asarray(y_te),
        y_pred=y_pred, proba=np.asarray(proba),
        accuracy=acc, precision=prec, recall=rec, f1=f1, confusion=cm,
        equity=equity, strat_return=total_ret, strat_sharpe=sharpe, win_rate=win_rate,
        next_up_prob=float(next_prob), next_signal=next_signal,
        next_exp_return=float(exp_ret), next_price=float(last_close * (1 + exp_ret)),
        last_close=float(last_close),
        feat_importance=imp,
    )


def _train_one(name: str, X: pd.DataFrame, y: pd.Series,
               ret_next: pd.Series, split: int,
               live_row: pd.DataFrame, last_close: float,
               feature_cols: list[str]) -> tuple[MovementResult, dict]:
    """Train one base model. Returns (result, raw) where ``raw`` carries the
    test-set probabilities & live probability the ensemble averages over."""
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    r_te       = ret_next.iloc[split:]

    model = _build_model(name)
    model.fit(X_tr, y_tr)
    try:
        proba = model.predict_proba(X_te)[:, 1]
    except Exception:
        proba = model.predict(X_te).astype(float)
    y_pred = (proba >= 0.5).astype(int)

    # Live call: retrain on ALL labelled data, predict the live (forecast) row
    model_full = _build_model(name)
    model_full.fit(X, y)
    try:
        next_prob = float(model_full.predict_proba(live_row)[0, 1])
    except Exception:
        next_prob = float(model_full.predict(live_row)[0])

    imp_vals = _importance_vals(model_full, name)
    res = _assemble(name, y_pred, proba, y_te.values, r_te, X_te.index,
                    next_prob, ret_next, last_close, feature_cols, imp_vals)
    raw = {"proba": np.asarray(proba), "next_prob": next_prob,
           "y_te": y_te.values, "r_te": r_te, "test_index": X_te.index,
           "imp_vals": imp_vals}
    return res, raw


# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINES
# ─────────────────────────────────────────────────────────────────────────────
def _baselines(ret_next: pd.Series, split: int, y_true: np.ndarray) -> dict:
    r_te = ret_next.iloc[split:]

    # Buy & hold — always long
    bh_equity = (1.0 + r_te).cumprod()
    bh_return = float(bh_equity.iloc[-1] - 1.0) if len(bh_equity) else 0.0

    # Random guess — accuracy on direction (deterministic seed for stability)
    rng       = np.random.default_rng(42)
    rand_pred = rng.integers(0, 2, size=len(y_true))
    rand_acc  = float(accuracy_score(y_true, rand_pred))

    # Majority-class (always predict "up") baseline accuracy
    up_rate   = float(np.mean(y_true))
    base_acc  = max(up_rate, 1 - up_rate)

    return {
        "buyhold_equity": bh_equity,
        "buyhold_return": bh_return,
        "random_acc":     rand_acc * 100,
        "majority_acc":   base_acc * 100,
        "up_rate":        up_rate * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HERO — Monte Carlo simulation of a stock (GBM)
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_paths(close: pd.Series, n_paths: int = 100, n_steps: int = 100,
                      seed: int = 7) -> np.ndarray:
    """Geometric-Brownian-Motion simulation calibrated to the stock's own
    historical daily drift & volatility. Returns an (n_steps+1, n_paths) array."""
    ret = close.pct_change().dropna()
    mu  = float(ret.mean())
    sig = float(ret.std())
    s0  = float(close.iloc[-1])

    rng    = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu - 0.5 * sig ** 2, scale=sig, size=(n_steps, n_paths))
    logpay = np.vstack([np.zeros((1, n_paths)), np.cumsum(shocks, axis=0)])
    return s0 * np.exp(logpay)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_movement_analysis(ohlcv: pd.DataFrame,
                          models: Optional[list[str]] = None,
                          test_frac: float = 0.25,
                          ticker: Optional[str] = None,
                          use_fundamentals: bool = True) -> dict:
    """Full up/down pipeline. Returns a dict consumed by the Streamlit tab.

    When ``ticker`` is given and ``use_fundamentals`` is True, the model also
    trains on backtestable earnings-cycle + market-regime (Nifty/VIX) features.
    """
    if models is None:
        models = list(BASE_MODELS)
    # The ensemble is derived, not trained directly — strip it from the base list.
    base_models = [m for m in models if m != "Ensemble"] or list(BASE_MODELS)

    feat = engineer_features(ohlcv, ticker=ticker, use_fundamentals=use_fundamentals)
    feature_cols = feat.attrs.get("feature_cols", list(PRICE_FEATURE_NAMES))

    # Live forecast row = most-recent day with valid features (target unknown).
    live_row   = feat[feature_cols].iloc[[-1]]
    last_close = float(feat["close"].iloc[-1])

    # Labelled data (drop the final, unlabelled row) for training + backtesting.
    labelled = feat.dropna(subset=["target", "ret_next"])
    if len(labelled) < 120:
        raise ValueError(
            f"Only {len(labelled)} usable rows after feature engineering — "
            f"need ≥120. Try a longer history window.")

    X        = labelled[feature_cols]
    y        = labelled["target"].astype(int)
    ret_next = labelled["ret_next"]

    split = int(len(labelled) * (1 - test_frac))
    split = max(80, min(split, len(labelled) - 30))   # keep both sides usable

    results: dict[str, MovementResult] = {}
    raws:    dict[str, dict] = {}
    for name in base_models:
        try:
            res, raw       = _train_one(name, X, y, ret_next, split,
                                        live_row, last_close, feature_cols)
            results[name]  = res
            raws[name]     = raw
        except Exception as e:                        # pragma: no cover
            results[name] = MovementResult(
                name=name, test_dates=X.index[split:], y_true=y.iloc[split:].values,
                y_pred=np.array([]), proba=np.array([]),
                accuracy=0, precision=0, recall=0, f1=0,
                confusion=np.zeros((2, 2), int),
                equity=pd.Series(dtype=float), strat_return=0, strat_sharpe=0,
                win_rate=0, next_up_prob=0.5, next_signal="—", error=str(e))

    # ── Soft-voting ensemble: average the base models' probabilities ──────────
    if len(raws) >= 2:
        te_proba  = np.mean([raws[n]["proba"] for n in raws], axis=0)
        next_prob = float(np.mean([raws[n]["next_prob"] for n in raws]))
        imps      = [raws[n]["imp_vals"] for n in raws if raws[n]["imp_vals"] is not None]
        imp_avg   = np.mean(imps, axis=0) if imps else None
        any_raw   = next(iter(raws.values()))
        results["Ensemble"] = _assemble(
            "Ensemble", (te_proba >= 0.5).astype(int), te_proba,
            any_raw["y_te"], any_raw["r_te"], any_raw["test_index"],
            next_prob, ret_next, last_close, feature_cols, imp_avg)

    ok = {k: v for k, v in results.items() if v.error is None}
    if not ok:
        raise ValueError("All classifiers failed to train.")

    # Prefer the ensemble for the headline call (most stable); otherwise rank by
    # a stability-aware score: accuracy blended with F1.
    def _score(r: MovementResult) -> float:
        return 0.5 * r.accuracy + 0.5 * r.f1
    if "Ensemble" in ok:
        best_name = "Ensemble"
    else:
        best_name = max(ok, key=lambda k: _score(ok[k]))
    base = _baselines(ret_next, split, ok[best_name].y_true)

    return {
        "results":    results,
        "best_name":  best_name,
        "baselines":  base,
        "split_date": labelled.index[split],
        "n_train":    split,
        "n_test":     len(labelled) - split,
        "test_dates": ok[best_name].test_dates,
        "close":      ohlcv["Close"].astype(float),
        "feature_cols": feature_cols,
        "n_features":   len(feature_cols),
    }


# Base learners actually trained (gradient-boosted trees + linear baseline).
BASE_MODELS = (["LightGBM"] if _HAS_LGBM else []) \
            + (["XGBoost"] if _HAS_XGB else []) \
            + ["Logistic Regression"]
# What the UI offers to toggle; "Ensemble" is derived from the base learners.
MOVEMENT_MODELS = ["Ensemble"] + BASE_MODELS
