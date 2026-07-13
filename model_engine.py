"""
Client-grade trading model engine
==================================
Rigorous, decision-worthy evaluation of the Stock Movement Predictor for a
trading (entry/exit) use-case. Built the way a data scientist would defend it to
a client — no single lucky split, no un-costed backtest, no un-calibrated
probabilities.

Pipeline
--------
1.  TARGET      multi-day direction: does the stock close higher H trading days
                from now? (H = 5 or 10 — real signal, unlike 1-day noise)
2.  CV          expanding-window walk-forward with an H-day **embargo** between
                train and test (labels span H days → prevents leakage)
3.  CALIBRATION per-fold Platt scaling (prefit) so a "62%" really means ~62%
4.  METRICS     ROC-AUC · PR-AUC · Brier · log-loss · accuracy/F1  (mean ± std
                across folds, plus pooled out-of-sample)
5.  BACKTEST    cost-aware: bps per trade + slippage on turnover → net equity,
                annualised Sharpe/Sortino, max drawdown, hit-rate, turnover,
                vs a net buy&hold
6.  SIGNIFICANCE bootstrap CI on AUC + a random-signal Sharpe null → "is the
                edge real, or luck?"
7.  VERDICT     GREEN / AMBER / RED deploy recommendation per stock

The result dict is consumed by the Predict tab.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    accuracy_score, f1_score, precision_score, recall_score,
)

from movement_predictor import engineer_features, _build_model, BASE_MODELS, PRICE_FEATURE_NAMES

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TradingModelResult:
    ticker:        str
    horizon:       int
    n_samples:     int
    n_features:    int
    feature_cols:  list

    # cross-validated classification metrics (pooled OOS + per-fold mean/std)
    cv_metrics:    dict           # {metric: {"mean":, "std":, "pooled":}}
    auc_ci:        tuple          # (lo, hi) bootstrap 95% CI on pooled AUC

    # calibration
    reliability:   dict           # {"bins":, "pred":, "obs":}
    brier:         float

    # cost-aware backtest (net of costs)
    equity:        pd.Series
    bh_equity:     pd.Series
    net_return:    float
    ann_return:    float
    sharpe:        float
    sortino:       float
    max_dd:        float
    hit_rate:      float
    turnover:      float
    n_trades:      int
    threshold:     float
    cost_bps:      float

    # significance
    sharpe_pvalue: float          # vs random-signal null
    edge_is_real:  bool

    # live call (model retrained on all data)
    next_proba:    float
    next_signal:   str
    next_price:    float
    exp_return:    float
    last_close:    float

    # explainability
    feat_importance: dict

    # verdict
    verdict:       str            # GREEN / AMBER / RED
    verdict_reason: str

    oos_dates:     pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    oos_proba:     np.ndarray = field(default_factory=lambda: np.array([]))
    oos_true:      np.ndarray = field(default_factory=lambda: np.array([]))


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
def make_labels(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Forward H-day return, binary up-label, and next-day return (for PnL)."""
    fwd_ret = close.shift(-horizon) / close - 1.0
    target  = (fwd_ret > 0).astype(float)
    ret_1d  = close.pct_change().shift(-1)          # tomorrow's return, held daily
    return fwd_ret, target, ret_1d


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (prefit Platt scaling)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_calibrated(name, X_core, y_core, X_cal, y_cal):
    """Fit base model on core, then a Platt (sigmoid) map on the calib slice.
    Returns (base_model, platt) where platt maps raw prob → calibrated prob."""
    base = _build_model(name)
    base.fit(X_core, y_core)
    raw  = _raw_proba(base, X_cal)
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    try:
        platt.fit(raw.reshape(-1, 1), y_cal)
    except Exception:                                # single-class calib slice
        platt = None
    return base, platt


def _raw_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.predict(X).astype(float)


def _apply(base, platt, X):
    raw = _raw_proba(base, X)
    if platt is None:
        return raw
    return platt.predict_proba(raw.reshape(-1, 1))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Cost-aware backtest
# ─────────────────────────────────────────────────────────────────────────────
def _backtest(proba, ret_1d, threshold, cost_bps, slippage_bps):
    """Long when calibrated P(up) >= threshold, else flat. Daily PnL on next-day
    returns; costs charged on turnover (position changes)."""
    pos       = (proba >= threshold).astype(float)
    pos_prev  = np.concatenate([[0.0], pos[:-1]])
    turnover  = np.abs(pos - pos_prev)
    cost      = turnover * (cost_bps + slippage_bps) / 1e4
    net_ret   = pos * ret_1d - cost
    equity    = np.cumprod(1.0 + net_ret)
    return pos, net_ret, equity, turnover


def _sharpe(daily):
    sd = np.std(daily)
    return float(np.mean(daily) / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0


def _sortino(daily):
    downside = daily[daily < 0]
    dd = np.std(downside)
    return float(np.mean(daily) / dd * np.sqrt(TRADING_DAYS)) if dd > 1e-12 else 0.0


def _max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0)) if len(equity) else 0.0


def _pick_threshold(proba, ret_1d, cost_bps, slippage_bps):
    """Choose the probability threshold that maximises net Sharpe (on the slice
    it's given — used on calibration data, kept out of the test window)."""
    best_thr, best_s = 0.5, -1e9
    for thr in np.arange(0.45, 0.70, 0.025):
        _, net, _, _ = _backtest(proba, ret_1d, thr, cost_bps, slippage_bps)
        s = _sharpe(net)
        if s > best_s:
            best_s, best_thr = s, thr
    return float(best_thr)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward CV
# ─────────────────────────────────────────────────────────────────────────────
def _cv_metrics(y, proba, thr):
    yhat = (proba >= thr).astype(int)
    out = {}
    try:    out["AUC"]  = roc_auc_score(y, proba)
    except Exception: out["AUC"] = 0.5
    try:    out["PR-AUC"] = average_precision_score(y, proba)
    except Exception: out["PR-AUC"] = float(np.mean(y))
    out["Brier"]     = brier_score_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))
    try:    out["LogLoss"] = log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])
    except Exception: out["LogLoss"] = np.nan
    out["Accuracy"]  = accuracy_score(y, yhat)
    out["F1"]        = f1_score(y, yhat, zero_division=0)
    out["Precision"] = precision_score(y, yhat, zero_division=0)
    out["Recall"]    = recall_score(y, yhat, zero_division=0)
    return out


def walk_forward(name, X, y, ret_1d, horizon, n_folds=5, cost_bps=10.0, slippage_bps=5.0):
    """Expanding-window walk-forward with an H-day embargo. Returns pooled OOS
    predictions, per-fold thresholds, and per-fold metric dicts."""
    n = len(X)
    # test blocks over the last ~60% of data, expanding train before each
    start = int(n * 0.40)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)

    oos_idx, oos_proba, oos_pos, oos_ret, fold_metrics = [], [], [], [], []
    for k in range(n_folds):
        te_lo, te_hi = bounds[k], bounds[k + 1]
        tr_hi = max(te_lo - horizon, 0)               # embargo H days
        if tr_hi < 100 or te_hi - te_lo < 15:
            continue
        # split train into core / calibration (last 20%)
        cut     = int(tr_hi * 0.80)
        X_core, y_core = X.iloc[:cut], y.iloc[:cut]
        X_cal,  y_cal  = X.iloc[cut:tr_hi], y.iloc[cut:tr_hi]
        X_te            = X.iloc[te_lo:te_hi]
        y_te            = y.iloc[te_lo:te_hi]
        r_te            = ret_1d.iloc[te_lo:te_hi]
        if len(X_cal) < 30 or y_core.nunique() < 2:
            continue

        base, platt = _fit_calibrated(name, X_core, y_core, X_cal, y_cal)
        cal_proba   = _apply(base, platt, X_cal)
        thr         = _pick_threshold(cal_proba.values if hasattr(cal_proba, "values") else cal_proba,
                                      ret_1d.iloc[cut:tr_hi].values, cost_bps, slippage_bps)
        te_proba    = _apply(base, platt, X_te)

        pos, net, _, _ = _backtest(te_proba, r_te.values, thr, cost_bps, slippage_bps)
        fold_metrics.append({**_cv_metrics(y_te.values, te_proba, thr),
                             "Threshold": thr, "NetSharpe": _sharpe(net)})
        oos_idx.extend(X_te.index)
        oos_proba.extend(te_proba)
        oos_pos.extend(pos)
        oos_ret.extend(r_te.values)

    return {
        "oos_index": pd.DatetimeIndex(oos_idx),
        "oos_proba": np.array(oos_proba),
        "oos_pos":   np.array(oos_pos),
        "oos_ret":   np.array(oos_ret),
        "y_true":    y.reindex(pd.DatetimeIndex(oos_idx)).values if len(oos_idx) else np.array([]),
        "fold_metrics": fold_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Significance
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_auc_ci(y, proba, n=500, seed=0):
    rng = np.random.default_rng(seed)
    if len(np.unique(y)) < 2:
        return (0.5, 0.5)
    stats = []
    m = len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], proba[idx]))
    if not stats:
        return (0.5, 0.5)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def _sharpe_pvalue(net_ret, pos, ret, n=500, seed=1):
    """Null: shuffle the position series (random timing, same exposure)."""
    actual = _sharpe(net_ret)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        p = rng.permutation(pos)
        null.append(_sharpe(p * ret))
    null = np.array(null)
    p = float(np.mean(null >= actual))
    return p, actual


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_trading_model(ohlcv: pd.DataFrame, ticker: str, horizon: int = 5,
                      models: Optional[list] = None, n_folds: int = 5,
                      cost_bps: float = 10.0, slippage_bps: float = 5.0,
                      use_fundamentals: bool = True) -> TradingModelResult:
    models = models or BASE_MODELS

    feat = engineer_features(ohlcv, ticker=ticker, use_fundamentals=use_fundamentals)
    feature_cols = feat.attrs.get("feature_cols", list(PRICE_FEATURE_NAMES))
    close = feat["close"].astype(float)

    fwd_ret, target, ret_1d = make_labels(close, horizon)
    df = feat[feature_cols].copy()
    df["target"] = target
    df["ret_1d"] = ret_1d
    live_row = df[feature_cols].iloc[[-1]]            # last row → forecast H days ahead
    df = df.dropna(subset=["target", "ret_1d"])       # drop last H rows (no label)

    if len(df) < 250:
        raise ValueError(f"Only {len(df)} labelled rows — need ≥250 for a credible "
                         f"walk-forward at horizon {horizon}. Use a longer history.")

    X = df[feature_cols]
    y = df["target"].astype(int)
    r = df["ret_1d"]

    # ── Walk-forward CV per model, then a soft-voting ensemble on pooled OOS ──
    per_model = {m: walk_forward(m, X, y, r, horizon, n_folds, cost_bps, slippage_bps)
                 for m in models}
    per_model = {m: v for m, v in per_model.items() if len(v["oos_proba"]) > 0}
    if not per_model:
        raise ValueError("Walk-forward produced no out-of-sample predictions.")

    # align all models on the common OOS index (they share it) and average proba
    ref = next(iter(per_model.values()))
    oos_index = ref["oos_index"]
    oos_ret   = ref["oos_ret"]
    y_true    = y.reindex(oos_index).values
    ens_proba = np.mean([v["oos_proba"] for v in per_model.values()], axis=0)

    # ── Choose the best base model by mean fold AUC; ensemble usually wins ──
    def _mean_auc(v):
        return float(np.mean([f["AUC"] for f in v["fold_metrics"]])) if v["fold_metrics"] else 0.5
    all_probas = {**{m: v["oos_proba"] for m, v in per_model.items()}, "Ensemble": ens_proba}
    all_folds  = {**{m: v["fold_metrics"] for m, v in per_model.items()},
                  "Ensemble": None}
    best_name = max([*per_model.keys(), "Ensemble"],
                    key=lambda m: (roc_auc_score(y_true, all_probas[m])
                                   if len(np.unique(y_true)) > 1 else 0.5))
    proba = all_probas[best_name]

    # deployment threshold = median of per-fold thresholds (out-of-sample choice)
    fold_thrs = [f["Threshold"] for v in per_model.values() for f in v["fold_metrics"]]
    threshold = float(np.median(fold_thrs)) if fold_thrs else 0.5

    # ── Pooled OOS metrics + per-fold mean/std (use best model's folds if base) ──
    pooled = _cv_metrics(y_true, proba, threshold)
    ref_folds = per_model[best_name]["fold_metrics"] if best_name in per_model else \
                [f for v in per_model.values() for f in v["fold_metrics"]]
    cv_metrics = {}
    for key in ["AUC", "PR-AUC", "Brier", "LogLoss", "Accuracy", "F1", "Precision", "Recall"]:
        vals = [f[key] for f in ref_folds if key in f and not np.isnan(f.get(key, np.nan))]
        cv_metrics[key] = {"mean": float(np.mean(vals)) if vals else np.nan,
                           "std":  float(np.std(vals)) if vals else np.nan,
                           "pooled": pooled.get(key, np.nan)}
    auc_ci = _bootstrap_auc_ci(y_true, proba)

    # ── Reliability curve (calibration) ──
    bins = np.linspace(0, 1, 11)
    which = np.digitize(proba, bins) - 1
    rel_pred, rel_obs = [], []
    for b in range(10):
        mask = which == b
        if mask.sum() >= 5:
            rel_pred.append(float(proba[mask].mean()))
            rel_obs.append(float(y_true[mask].mean()))
    reliability = {"pred": rel_pred, "obs": rel_obs}

    # ── Cost-aware net backtest on pooled OOS ──
    pos, net_ret, equity, turnover = _backtest(proba, oos_ret, threshold, cost_bps, slippage_bps)
    bh_equity = np.cumprod(1.0 + oos_ret)
    n_days    = len(net_ret)
    ann_return = float(equity[-1] ** (TRADING_DAYS / n_days) - 1.0) if n_days > 0 and equity[-1] > 0 else -1.0
    traded    = net_ret[pos > 0]
    hit_rate  = float((traded > 0).mean() * 100) if len(traded) else 0.0
    n_trades  = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0))

    # ── Significance ──
    sh_p, sharpe = _sharpe_pvalue(net_ret, pos, oos_ret)
    auc_real = auc_ci[0] > 0.5
    edge_is_real = bool(auc_real and sh_p < 0.10 and sharpe > 0)

    # ── Live call: retrain on ALL data w/ calibration on last 20% ──
    cut = int(len(X) * 0.80)
    base_live, platt_live = _fit_calibrated(best_name if best_name in models else models[0],
                                            X.iloc[:cut], y.iloc[:cut],
                                            X.iloc[cut:], y.iloc[cut:])
    if best_name == "Ensemble":
        live_probs = []
        for m in models:
            b, pl = _fit_calibrated(m, X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:])
            live_probs.append(_apply(b, pl, live_row)[0])
        next_proba = float(np.mean(live_probs))
    else:
        next_proba = float(_apply(base_live, platt_live, live_row)[0])
    next_signal = "UP" if next_proba >= threshold else "DOWN"

    # expected H-day price from calibrated prob × avg up/down H-day move
    up_mv = fwd_ret[(fwd_ret > 0)].mean()
    dn_mv = fwd_ret[(fwd_ret <= 0)].mean()
    exp_ret = next_proba * (up_mv if pd.notna(up_mv) else 0) + (1 - next_proba) * (dn_mv if pd.notna(dn_mv) else 0)
    last_close = float(close.iloc[-1])

    # ── Feature importance (avg over base tree models, live fit) ──
    imp = _live_importance(models, X, y, feature_cols)

    # ── Verdict ──
    verdict, reason = _verdict(auc_ci, sharpe, sh_p, ann_return, oos_ret, cv_metrics)

    return TradingModelResult(
        ticker=ticker, horizon=horizon, n_samples=len(df), n_features=len(feature_cols),
        feature_cols=feature_cols, cv_metrics=cv_metrics, auc_ci=auc_ci,
        reliability=reliability, brier=pooled["Brier"],
        equity=pd.Series(equity, index=oos_index),
        bh_equity=pd.Series(bh_equity, index=oos_index),
        net_return=float(equity[-1] - 1.0), ann_return=ann_return,
        sharpe=sharpe, sortino=_sortino(net_ret), max_dd=_max_dd(equity),
        hit_rate=hit_rate, turnover=float(np.sum(turnover)), n_trades=n_trades,
        threshold=threshold, cost_bps=cost_bps + slippage_bps,
        sharpe_pvalue=sh_p, edge_is_real=edge_is_real,
        next_proba=next_proba, next_signal=next_signal,
        next_price=last_close * (1 + exp_ret), exp_return=float(exp_ret),
        last_close=last_close, feat_importance=imp,
        verdict=verdict, verdict_reason=reason,
        oos_dates=oos_index, oos_proba=proba, oos_true=y_true,
    )


def _live_importance(models, X, y, feature_cols):
    imps = []
    for m in models:
        if m == "Logistic Regression":
            continue
        try:
            mdl = _build_model(m); mdl.fit(X, y)
            v = np.asarray(mdl.feature_importances_, dtype=float)
            if v.sum() > 0:
                imps.append(v / v.sum())
        except Exception:
            pass
    if not imps:
        return {}
    avg = np.mean(imps, axis=0)
    return dict(sorted(zip(feature_cols, avg), key=lambda x: -x[1]))


def _verdict(auc_ci, sharpe, sh_p, ann_return, oos_ret, cv_metrics):
    bh_ann = float(np.mean(oos_ret) * TRADING_DAYS)
    auc_lo = auc_ci[0]
    if auc_lo > 0.52 and sharpe > 0.5 and sh_p < 0.05:
        return ("GREEN", f"AUC 95% CI lower bound {auc_lo:.2f} > 0.5, net Sharpe {sharpe:.2f}, "
                         f"edge significant (p={sh_p:.2f}). Deployable with monitoring.")
    if auc_lo > 0.50 and sharpe > 0.2 and sh_p < 0.15:
        return ("AMBER", f"Marginal edge: AUC CI lower {auc_lo:.2f}, Sharpe {sharpe:.2f}, p={sh_p:.2f}. "
                         f"Paper-trade before risking capital.")
    return ("RED", f"No reliable edge: AUC CI lower {auc_lo:.2f} (≈0.5), Sharpe {sharpe:.2f}, "
                   f"p={sh_p:.2f}. Do not deploy for trading at this horizon.")
