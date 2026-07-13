"""
Pooled cross-sectional trading model
=====================================
The client-grade core. Instead of predicting each stock in isolation (which our
walk-forward proved has no reliable edge), we train ONE model on the whole
Nifty-50 panel — ~25k rows — so it learns a *general* cross-sectional pattern.
Each period it ranks the universe by calibrated P(up over H days); we trade the
spread (long the strongest, short the weakest) net of costs.

Why this is defensible to a client
----------------------------------
• Pooled panel → far more signal than one noisy ticker (AUC CI lower bound > 0.5).
• Walk-forward by DATE with an H-day embargo → no look-ahead, no leakage.
• Platt-calibrated probabilities → a "58%" really means ~58%.
• Long/short basket backtest **net of costs & slippage** → economically honest.
• Bootstrap CI on AUC + random-timing Sharpe null → "is the edge real?"
• GREEN / AMBER / RED verdict → a clear deploy decision.

Entry point: run_universe_model(...).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from data_fetcher import fetch_ohlcv
from movement_predictor import engineer_features, _build_model, BASE_MODELS, PRICE_FEATURE_NAMES
from model_engine import (
    make_labels, _fit_calibrated, _apply, _backtest, _sharpe, _sortino, _max_dd,
    _bootstrap_auc_ci, _cv_metrics, TRADING_DAYS,
)

warnings.filterwarnings("ignore")


@dataclass
class UniverseResult:
    horizon:       int
    n_stocks:      int
    n_rows:        int
    n_features:    int
    feature_cols:  list

    # model quality (pooled OOS)
    auc:           float
    auc_ci:        tuple
    brier:         float
    accuracy:      float
    fold_aucs:     list
    reliability:   dict

    # cross-sectional long/short backtest (net of costs)
    equity:        pd.Series
    bench_equity:  pd.Series
    ann_return:    float
    sharpe:        float
    sortino:       float
    max_dd:        float
    hit_rate:      float
    bench_ann:     float
    n_periods:     int
    top_frac:      float
    cost_bps:      float
    long_short:    bool
    sharpe_pvalue: float

    # today's ranking (live) — DataFrame: name, ticker, proba, signal
    ranking:       pd.DataFrame

    # per-stock OOS slice (for the deep-dive): {ticker: {...}}
    per_stock:     dict

    feat_importance: dict
    verdict:       str
    verdict_reason: str
    oos:           pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the panel
# ─────────────────────────────────────────────────────────────────────────────
def build_panel(universe: dict, period: str, horizon: int,
                use_fundamentals: bool = True, progress_cb=None):
    """Fetch every stock, engineer features, label H-day forward direction, and
    stack into one long panel. Also returns the latest (unlabelled) feature row
    per stock for live ranking."""
    frames, live_rows, feature_cols = [], {}, None
    names = list(universe.items())
    for i, (name, tk) in enumerate(names):
        if progress_cb:
            progress_cb(i, len(names), name)
        try:
            df = fetch_ohlcv(tk, period=period)
            if df is None or df.empty or len(df) < 400:
                continue
            feat = engineer_features(df, ticker=tk, use_fundamentals=use_fundamentals)
            cols = feat.attrs.get("feature_cols", list(PRICE_FEATURE_NAMES))
            feature_cols = cols
            close = feat["close"].astype(float)
            fwd, tar, r1 = make_labels(close, horizon)
            f = feat[cols].copy()
            f["target"] = tar
            f["fwd_ret"] = fwd
            f["ret_1d"] = r1
            f["name"] = name
            f["ticker"] = tk
            f["date"] = feat.index
            live_rows[tk] = {"name": name, "row": feat[cols].iloc[[-1]],
                             "last_close": float(close.iloc[-1])}
            frames.append(f.dropna(subset=["target", "fwd_ret", "ret_1d"]))
        except Exception:
            continue
    if not frames:
        raise ValueError("Could not build a panel — no stock data available.")
    panel = pd.concat(frames).sort_values("date").reset_index(drop=True)
    return panel, feature_cols, live_rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. Walk-forward over DATES (pooled), calibrated
# ─────────────────────────────────────────────────────────────────────────────
def pooled_walk_forward(panel, feature_cols, models, horizon,
                        n_folds=4, embargo_days=None):
    """Expanding-window walk-forward by calendar date with an H-day embargo.
    Trains the pooled model on all stocks up to each cut, calibrates, and
    predicts the next block. Returns pooled OOS predictions (one row per
    stock-date) plus per-fold AUCs."""
    embargo_days = embargo_days or horizon
    dates = np.array(sorted(panel["date"].unique()))
    nd = len(dates)
    start = int(nd * 0.45)
    bounds = np.linspace(start, nd, n_folds + 1).astype(int)

    oos_parts, fold_aucs = [], []
    for k in range(n_folds):
        te_lo, te_hi = bounds[k], bounds[k + 1]
        if te_hi - te_lo < 5:
            continue
        test_dates = dates[te_lo:te_hi]
        train_end  = dates[max(te_lo - embargo_days, 0)]
        tr = panel[panel["date"] < train_end]
        te = panel[panel["date"].isin(test_dates)]
        if len(tr) < 2000 or len(te) < 200:
            continue
        # chronological calibration slice = last 15% of train dates
        tr_dates = np.array(sorted(tr["date"].unique()))
        cal_start = tr_dates[int(len(tr_dates) * 0.85)]
        core = tr[tr["date"] < cal_start]
        cal  = tr[tr["date"] >= cal_start]
        if len(core) < 1000 or len(cal) < 200:
            core, cal = tr, tr

        te_probas = []
        for m in models:
            base, platt = _fit_calibrated(m, core[feature_cols], core["target"].astype(int),
                                          cal[feature_cols], cal["target"].astype(int))
            te_probas.append(_apply(base, platt, te[feature_cols]))
        ens = np.mean(te_probas, axis=0)
        part = te[["date", "name", "ticker", "target", "fwd_ret", "ret_1d"]].copy()
        part["proba"] = ens
        oos_parts.append(part)
        try:
            fold_aucs.append(roc_auc_score(part["target"].astype(int), ens))
        except Exception:
            pass

    if not oos_parts:
        raise ValueError("Walk-forward produced no OOS predictions (need longer history).")
    oos = pd.concat(oos_parts).reset_index(drop=True)
    return oos, fold_aucs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-sectional long/short backtest (net of costs)
# ─────────────────────────────────────────────────────────────────────────────
def cross_sectional_backtest(oos, horizon, top_frac=0.2, cost_bps=10.0,
                             slippage_bps=5.0, long_short=True, min_names=8):
    """Rebalance every H days: rank by proba, long the top fraction, short the
    bottom fraction (or long-only). Period return uses realised H-day forward
    returns; costs charged on full rotation each rebalance."""
    dates = np.array(sorted(oos["date"].unique()))
    rebal = dates[::horizon]                       # non-overlapping H-day periods
    per_ret, bench_ret, hits = [], [], []
    cost = 2 * (cost_bps + slippage_bps) / 1e4     # enter+exit
    for d in rebal:
        day = oos[oos["date"] == d]
        if len(day) < min_names:
            continue
        day = day.sort_values("proba", ascending=False)
        k = max(int(len(day) * top_frac), 1)
        longs = day.head(k)
        long_r = float(longs["fwd_ret"].mean())
        if long_short:
            shorts = day.tail(k)
            gross = long_r - float(shorts["fwd_ret"].mean())
        else:
            gross = long_r
        per_ret.append(gross - cost)
        bench_ret.append(float(day["fwd_ret"].mean()))
        hits.append(1.0 if gross > 0 else 0.0)

    per_ret = np.array(per_ret); bench_ret = np.array(bench_ret)
    if len(per_ret) == 0:
        raise ValueError("No rebalance periods with enough names.")
    equity = np.cumprod(1.0 + per_ret)
    bench_eq = np.cumprod(1.0 + bench_ret)
    ppy = TRADING_DAYS / horizon                    # periods per year
    ann = float(equity[-1] ** (ppy / len(per_ret)) - 1.0) if equity[-1] > 0 else -1.0
    bench_ann = float(bench_eq[-1] ** (ppy / len(bench_ret)) - 1.0) if bench_eq[-1] > 0 else -1.0
    sd = per_ret.std()
    sharpe = float(per_ret.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0
    dn = per_ret[per_ret < 0].std()
    sortino = float(per_ret.mean() / dn * np.sqrt(ppy)) if dn > 1e-12 else 0.0
    rebal_dates = rebal[:len(per_ret)]
    return {
        "equity": pd.Series(equity, index=rebal_dates),
        "bench_equity": pd.Series(bench_eq, index=rebal_dates),
        "ann": ann, "bench_ann": bench_ann, "sharpe": sharpe, "sortino": sortino,
        "max_dd": _max_dd(equity), "hit_rate": float(np.mean(hits) * 100),
        "n_periods": len(per_ret), "per_ret": per_ret,
    }


def _sharpe_pval_periods(per_ret, n=1000, seed=3):
    """Null: random sign flips of period returns (random timing)."""
    ppy_adj = np.sqrt(len(per_ret)) if len(per_ret) else 1
    actual = per_ret.mean() / (per_ret.std() + 1e-12) * np.sqrt(len(per_ret))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        signs = rng.choice([-1, 1], size=len(per_ret))
        x = per_ret * signs
        null.append(x.mean() / (x.std() + 1e-12) * np.sqrt(len(per_ret)))
    return float(np.mean(np.array(null) >= actual))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_universe_model(universe: dict, period: str = "6y", horizon: int = 10,
                       models: Optional[list] = None, top_frac: float = 0.2,
                       long_short: bool = True, cost_bps: float = 10.0,
                       slippage_bps: float = 5.0, n_folds: int = 4,
                       use_fundamentals: bool = True, progress_cb=None) -> UniverseResult:
    models = models or BASE_MODELS

    panel, feature_cols, live_rows = build_panel(universe, period, horizon,
                                                 use_fundamentals, progress_cb)
    oos, fold_aucs = pooled_walk_forward(panel, feature_cols, models, horizon, n_folds)

    y = oos["target"].astype(int).values
    p = oos["proba"].values
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    auc_ci = _bootstrap_auc_ci(y, p)
    m = _cv_metrics(y, p, 0.5)

    # reliability curve
    bins = np.linspace(0, 1, 11); which = np.digitize(p, bins) - 1
    rel_pred, rel_obs = [], []
    for b in range(10):
        mask = which == b
        if mask.sum() >= 20:
            rel_pred.append(float(p[mask].mean())); rel_obs.append(float(y[mask].mean()))
    reliability = {"pred": rel_pred, "obs": rel_obs}

    # backtest
    bt = cross_sectional_backtest(oos, horizon, top_frac, cost_bps, slippage_bps, long_short)
    sh_p = _sharpe_pval_periods(bt["per_ret"])

    # per-stock OOS slice
    per_stock = {}
    for tk, g in oos.groupby("ticker"):
        yy = g["target"].astype(int).values
        if len(yy) < 20 or len(np.unique(yy)) < 2:
            continue
        per_stock[tk] = {
            "name": g["name"].iloc[0],
            "auc": float(roc_auc_score(yy, g["proba"].values)),
            "n": int(len(yy)),
            "hit": float(((g["proba"] >= 0.5).astype(int).values == yy).mean() * 100),
        }

    # ── Live ranking: retrain pooled on ALL data (calibrated), score latest row ──
    dates = np.array(sorted(panel["date"].unique()))
    cal_start = dates[int(len(dates) * 0.85)]
    core = panel[panel["date"] < cal_start]; cal = panel[panel["date"] >= cal_start]
    if len(core) < 1000 or len(cal) < 200:
        core = cal = panel
    live_models = [_fit_calibrated(mm, core[feature_cols], core["target"].astype(int),
                                   cal[feature_cols], cal["target"].astype(int)) for mm in models]
    rank_rows = []
    for tk, info in live_rows.items():
        probs = [_apply(b, pl, info["row"])[0] for (b, pl) in live_models]
        pr = float(np.mean(probs))
        rank_rows.append({"name": info["name"], "ticker": tk, "proba": pr,
                          "last_close": info["last_close"]})
    ranking = pd.DataFrame(rank_rows).sort_values("proba", ascending=False).reset_index(drop=True)
    k = max(int(len(ranking) * top_frac), 1)
    ranking["bucket"] = "Hold"
    ranking.iloc[:k, ranking.columns.get_loc("bucket")] = "LONG"
    if long_short:
        ranking.iloc[-k:, ranking.columns.get_loc("bucket")] = "SHORT"

    # feature importance (pooled, tree models)
    imps = []
    for mm in models:
        if mm == "Logistic Regression":
            continue
        try:
            mdl = _build_model(mm); mdl.fit(panel[feature_cols], panel["target"].astype(int))
            v = np.asarray(mdl.feature_importances_, float)
            if v.sum() > 0:
                imps.append(v / v.sum())
        except Exception:
            pass
    feat_importance = dict(sorted(zip(feature_cols, np.mean(imps, axis=0)),
                                  key=lambda x: -x[1])) if imps else {}

    verdict, reason = _verdict(auc_ci, bt["sharpe"], sh_p, bt["ann"], bt["bench_ann"])

    return UniverseResult(
        horizon=horizon, n_stocks=int(panel["ticker"].nunique()), n_rows=int(len(panel)),
        n_features=len(feature_cols), feature_cols=feature_cols,
        auc=auc, auc_ci=auc_ci, brier=m["Brier"], accuracy=m["Accuracy"],
        fold_aucs=fold_aucs, reliability=reliability,
        equity=bt["equity"], bench_equity=bt["bench_equity"],
        ann_return=bt["ann"], sharpe=bt["sharpe"], sortino=bt["sortino"],
        max_dd=bt["max_dd"], hit_rate=bt["hit_rate"], bench_ann=bt["bench_ann"],
        n_periods=bt["n_periods"], top_frac=top_frac, cost_bps=cost_bps + slippage_bps,
        long_short=long_short, sharpe_pvalue=sh_p,
        ranking=ranking, per_stock=per_stock, feat_importance=feat_importance,
        verdict=verdict, verdict_reason=reason, oos=oos,
    )


def _verdict(auc_ci, sharpe, sh_p, ann, bench_ann):
    lo = auc_ci[0]
    if lo > 0.505 and sharpe > 0.6 and sh_p < 0.05 and ann > bench_ann:
        return ("GREEN", f"AUC 95% CI lower {lo:.3f} > 0.5, net Sharpe {sharpe:.2f} beats benchmark "
                         f"({ann*100:+.0f}% vs {bench_ann*100:+.0f}%), edge significant (p={sh_p:.2f}). "
                         f"Deployable with live monitoring & risk limits.")
    if lo > 0.500 and sharpe > 0.25 and sh_p < 0.15:
        return ("AMBER", f"Small but real edge: AUC CI lower {lo:.3f}, net Sharpe {sharpe:.2f}, "
                         f"p={sh_p:.2f}. Paper-trade and size conservatively before deploying capital.")
    return ("RED", f"No dependable edge after costs: AUC CI lower {lo:.3f}, Sharpe {sharpe:.2f}, "
                   f"p={sh_p:.2f}. Do not trade this configuration.")
