"""
War Room Tab — UI renderer for the 🤖 War Room tab.
===================================================
Renders 6 advanced prediction models side-by-side.
Call render_war_room_tab(prices, company_list, universe) from app.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import fetch_price_data
from war_room_models import (
    WAR_ROOM_MODELS, MODEL_ICONS, run_all_war_room,
)

# Groww colour palette (mirrors app.py constants)
_GW_GREEN  = "#00D09C"
_GW_RED    = "#FF5370"
_GW_PURPLE = "#5367FF"
_GW_NAVY   = "#1B2236"
_GW_GRAY   = "#F6F7F8"

_MODEL_COLORS: dict[str, str] = {
    "Stacked Ensemble ⭐": "#5367FF",
    "CatBoost":            "#F97316",
    "Extra Trees":         "#10B981",
    "SVR (RBF)":           "#8B5CF6",
    "Prophet":             "#00D09C",
    "LSTM":                "#EF4444",
    "Monte Carlo":         "#6B7280",
}


# ─────────────────────────────────────────────────────────────────────────────
# Small HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_card_html(name: str, icon: str, result, color: str) -> str:
    """Return HTML for a single model result card."""
    if result is None:
        return f"""
        <div style="background:#fff;border-radius:12px;padding:14px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);
                    border-left:4px solid #E5E7EB;opacity:0.5;">
            <div style="font-size:0.75rem;color:#9CA3AF;">{icon} {name}</div>
            <div style="font-size:0.82rem;color:#9CA3AF;margin-top:4px;">Not run</div>
        </div>"""

    if result.error:
        return f"""
        <div style="background:#fff;border-radius:12px;padding:14px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);
                    border-left:4px solid {_GW_RED};">
            <div style="font-size:0.78rem;font-weight:600;color:{_GW_NAVY};">{icon} {name}</div>
            <div style="font-size:0.70rem;color:{_GW_RED};margin-top:4px;">
                ⚠️ {result.error[:90]}
            </div>
        </div>"""

    fc = result.forecast
    if fc is None or fc.empty:
        return f"""
        <div style="background:#fff;border-radius:12px;padding:14px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);
                    border-left:4px solid {color};">
            <div style="font-size:0.78rem;font-weight:600;color:{_GW_NAVY};">{icon} {name}</div>
            <div style="font-size:0.72rem;color:#9CA3AF;">No forecast data</div>
        </div>"""

    chg_pct   = (fc.iloc[-1] - fc.iloc[0]) / (abs(fc.iloc[0]) + 1e-8) * 100
    dir_color = _GW_GREEN if chg_pct >= 0 else _GW_RED
    dir_arrow = "▲" if chg_pct >= 0 else "▼"
    metrics   = result.metrics or {}
    dir_acc   = metrics.get("Directional Acc %", "—")
    mape      = metrics.get("MAPE % (price)", "—")
    dir_acc_s = f"{dir_acc}%" if isinstance(dir_acc, (int, float)) else "—"
    mape_s    = f"{mape}%"   if isinstance(mape,    (int, float)) else "—"

    return f"""
    <div style="background:#fff;border-radius:12px;padding:14px;
                box-shadow:0 1px 6px rgba(0,0,0,0.07);
                border-left:4px solid {color};">
        <div style="font-size:0.78rem;font-weight:600;color:{_GW_NAVY};">{icon} {name}</div>
        <div style="font-size:1.0rem;font-weight:700;margin:4px 0;color:{_GW_NAVY};">
            ₹{fc.iloc[-1]:,.2f}
            &nbsp;<span style="color:{dir_color};">{dir_arrow} {abs(chg_pct):.1f}%</span>
        </div>
        <div style="font-size:0.70rem;color:#6B7280;">
            Dir Acc: <b>{dir_acc_s}</b> &nbsp;·&nbsp; MAPE: <b>{mape_s}</b>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_war_room_tab(
    prices: pd.DataFrame,
    company_list: list[str],
    universe: dict[str, str],
) -> None:
    """Render the full War Room tab. Called from app.py."""

    st.markdown(
        '<div class="gw-section-title">🤖 War Room — Advanced Prediction Models</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="gw-info-box">
            6 advanced ML/DL models: <b>Stacked Ensemble ⭐ · CatBoost · Extra Trees ·
            SVR (RBF) · Prophet + RSI · LSTM</b><br>
            <span style="color:#9CA3AF;font-size:0.75rem;">
            Bootstrap 90% confidence bands · Walk-forward validation ·
            LSTM cached between reruns · Indian market holidays baked in
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    with c1:
        wr_company = st.selectbox("Stock", company_list, index=0, key="wr_company")
    with c2:
        wr_horizon = st.slider(
            "Forecast horizon (trading days)",
            min_value=5, max_value=90, value=30, step=5,
            key="wr_horizon",
        )
    with c3:
        wr_inc_mc = st.checkbox("Monte Carlo", value=True, key="wr_mc")
    with c4:
        wr_run = st.button("🚀 Run War Room", key="wr_run", use_container_width=True)

    # ── Decide whether cached results can be reused ──────────────────────────
    _cached = (
        "wr_results" in st.session_state
        and st.session_state.get("wr_last_company") == wr_company
        and st.session_state.get("wr_last_horizon") == wr_horizon
    )

    if not wr_run and not _cached:
        st.markdown(
            """
            <div class="gw-card" style="text-align:center;padding:40px;
                                        color:#9CA3AF;font-size:0.88rem;">
                Select a stock and click <b>🚀 Run War Room</b> to run all 6 advanced models.<br>
                <span style="font-size:0.78rem;">
                    ⚠️ LSTM requires TensorFlow — first run may take 60–120 s.
                    All other models finish in ~10–30 s.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Run or restore ────────────────────────────────────────────────────────
    if wr_run:
        with st.spinner(f"Fetching 5-year data for {wr_company}…"):
            ticker = universe.get(wr_company, wr_company)
            raw = fetch_price_data([ticker], period="5y")
            if raw.empty:
                st.error(f"No price data returned for {wr_company}.")
                return
            wr_prices = raw.iloc[:, 0].dropna()
            wr_prices.name = wr_company

        progress = st.progress(0, text="Starting War Room models…")
        results: dict = {}
        model_list = list(WAR_ROOM_MODELS.items())
        for idx, (name, fn) in enumerate(model_list):
            progress.progress(
                int((idx) / len(model_list) * 90),
                text=f"Running {name}…",
            )
            kw = {"cache_key": f"wr_{wr_company}_{len(wr_prices)}"} if name == "LSTM" else {}
            results[name] = fn(wr_prices, horizon=wr_horizon, **kw)

        if wr_inc_mc:
            progress.progress(92, text="Running Monte Carlo…")
            from war_room_models import monte_carlo_forecast
            results["Monte Carlo"] = monte_carlo_forecast(wr_prices, horizon=wr_horizon)

        progress.progress(100, text="Done!")
        progress.empty()

        st.session_state["wr_results"]      = results
        st.session_state["wr_prices"]       = wr_prices
        st.session_state["wr_last_company"] = wr_company
        st.session_state["wr_last_horizon"] = wr_horizon

    results   = st.session_state["wr_results"]
    wr_prices = st.session_state.get("wr_prices", pd.Series())

    # ── Model snapshot cards (2 rows × 3 cols) ───────────────────────────────
    st.markdown("#### Model Snapshots")
    model_names = list(WAR_ROOM_MODELS.keys())
    for row_start in range(0, len(model_names), 3):
        row_names = model_names[row_start: row_start + 3]
        cols = st.columns(3)
        for col, name in zip(cols, row_names):
            icon  = MODEL_ICONS.get(name, "🔷")
            color = _MODEL_COLORS.get(name, _GW_PURPLE)
            with col:
                st.markdown(
                    _model_card_html(name, icon, results.get(name), color),
                    unsafe_allow_html=True,
                )

    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

    # ── Combined forecast chart ───────────────────────────────────────────────
    st.markdown("#### Combined Forecast Chart")
    fig_combo = go.Figure()

    if not wr_prices.empty:
        hist = wr_prices.iloc[-90:]
        fig_combo.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            mode="lines", name="Historical",
            line=dict(color=_GW_NAVY, width=2.5),
        ))
        _today_x = str(wr_prices.index[-1].date())
        fig_combo.add_shape(
            type="line", x0=_today_x, x1=_today_x, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="#9CA3AF", width=1.5, dash="dash"),
        )
        fig_combo.add_annotation(
            x=_today_x, y=0.97, yref="paper", text="Today",
            showarrow=False, font=dict(color="#9CA3AF", size=10), xanchor="left",
        )

    for name, res in results.items():
        if res is None or res.error or res.forecast is None or res.forecast.empty:
            continue
        color = _MODEL_COLORS.get(name, "#888")
        fc = res.forecast
        fig_combo.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            mode="lines", name=name,
            line=dict(
                color=color, width=1.8,
                dash="dot" if name == "Monte Carlo" else "solid",
            ),
            opacity=0.9,
        ))
        if (
            res.upper_bound is not None and not res.upper_bound.empty
            and res.lower_bound is not None and not res.lower_bound.empty
        ):
            x_band = list(fc.index) + list(fc.index[::-1])
            y_band = list(res.upper_bound.values) + list(res.lower_bound.values[::-1])
            fig_combo.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill="toself", fillcolor=color,
                opacity=0.06, line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))

    fig_combo.update_layout(
        height=420,
        paper_bgcolor="white", plot_bgcolor=_GW_GRAY,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=12, color=_GW_NAVY),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=10),
        ),
        xaxis=dict(gridcolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB", tickprefix="₹"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # ── Individual CI charts (3 per row) ─────────────────────────────────────
    st.markdown("#### Individual Forecast Detail")
    valid = [
        (n, r) for n, r in results.items()
        if r and not r.error and r.forecast is not None and not r.forecast.empty
    ]
    for chunk_start in range(0, len(valid), 3):
        chunk = valid[chunk_start: chunk_start + 3]
        cols  = st.columns(3)
        for col, (name, res) in zip(cols, chunk):
            color = _MODEL_COLORS.get(name, _GW_PURPLE)
            icon  = MODEL_ICONS.get(name, "🔷")
            fc    = res.forecast

            fig_ind = go.Figure()
            if not wr_prices.empty:
                h60 = wr_prices.iloc[-60:]
                fig_ind.add_trace(go.Scatter(
                    x=h60.index, y=h60.values,
                    mode="lines", showlegend=False,
                    line=dict(color=_GW_NAVY, width=1.5),
                ))
            fig_ind.add_trace(go.Scatter(
                x=fc.index, y=fc.values,
                mode="lines", showlegend=False,
                line=dict(color=color, width=2),
            ))
            if (
                res.upper_bound is not None and not res.upper_bound.empty
                and res.lower_bound is not None and not res.lower_bound.empty
            ):
                x_b = list(fc.index) + list(fc.index[::-1])
                y_b = list(res.upper_bound.values) + list(res.lower_bound.values[::-1])
                fig_ind.add_trace(go.Scatter(
                    x=x_b, y=y_b, fill="toself",
                    fillcolor=color, opacity=0.14,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
            fig_ind.update_layout(
                title=dict(text=f"{icon} {name}", font=dict(size=11, color=_GW_NAVY)),
                height=220,
                paper_bgcolor="white", plot_bgcolor=_GW_GRAY,
                margin=dict(l=8, r=8, t=30, b=8),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(
                    showgrid=True, gridcolor="#E5E7EB",
                    tickprefix="₹", tickfont=dict(size=9),
                ),
                hovermode="x unified",
            )
            with col:
                st.plotly_chart(fig_ind, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### Model Comparison Table")
    rows = []
    for name, res in results.items():
        icon = MODEL_ICONS.get(name, "🔷")
        if res is None or res.error:
            rows.append({
                "Model": f"{icon} {name}", "Status": "❌ Error",
                "End Price": "—", "Forecast Δ%": "—",
                "Dir Acc %": "—", "MAPE %": "—", "Signal": "—",
            })
            continue
        fc = res.forecast
        if fc is None or fc.empty:
            continue
        chg_pct = (fc.iloc[-1] - fc.iloc[0]) / (abs(fc.iloc[0]) + 1e-8) * 100
        metrics = res.metrics or {}
        da  = metrics.get("Directional Acc %", "—")
        mp  = metrics.get("MAPE % (price)", "—")
        sig = "BUY  ▲" if chg_pct >= 1.0 else ("SELL ▼" if chg_pct <= -1.0 else "HOLD —")
        rows.append({
            "Model":       f"{icon} {name}",
            "Status":      "✅ OK",
            "End Price":   f"₹{fc.iloc[-1]:,.2f}",
            "Forecast Δ%": f"{chg_pct:+.2f}%",
            "Dir Acc %":   f"{da}%" if isinstance(da, (int, float)) else "—",
            "MAPE %":      f"{mp}%" if isinstance(mp, (int, float)) else "—",
            "Signal":      sig,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── War Room Consensus banner ─────────────────────────────────────────────
    buy_c = sell_c = hold_c = 0
    for name, res in results.items():
        if res is None or res.error or res.forecast is None or res.forecast.empty:
            continue
        chg = (res.forecast.iloc[-1] - res.forecast.iloc[0]) / (abs(res.forecast.iloc[0]) + 1e-8) * 100
        if chg >= 1.0:
            buy_c += 1
        elif chg <= -1.0:
            sell_c += 1
        else:
            hold_c += 1

    total = buy_c + sell_c + hold_c
    if total > 0:
        if buy_c > sell_c and buy_c >= hold_c:
            label, clr = "BULLISH", _GW_GREEN
        elif sell_c > buy_c and sell_c >= hold_c:
            label, clr = "BEARISH", _GW_RED
        else:
            label, clr = "NEUTRAL", "#F59E0B"

        st.markdown(f"""
        <div style="background:#fff;border-radius:12px;padding:20px;
                    text-align:center;margin-top:12px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07);">
            <div style="font-size:0.78rem;color:#9CA3AF;letter-spacing:0.08em;">
                WAR ROOM CONSENSUS
            </div>
            <div style="font-size:1.9rem;font-weight:700;color:{clr};margin:4px 0;">
                {label}
            </div>
            <div style="font-size:0.82rem;color:#9CA3AF;">
                {buy_c} BUY &nbsp;·&nbsp; {sell_c} SELL &nbsp;·&nbsp;
                {hold_c} HOLD &nbsp;out of&nbsp; {total} models
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Monte Carlo end-price distribution ───────────────────────────────────
    mc_res = results.get("Monte Carlo")
    if mc_res and not mc_res.error and mc_res.forecast is not None and not mc_res.forecast.empty:
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        st.markdown("#### Monte Carlo End-Price Distribution")

        mc_fc    = mc_res.forecast
        mc_end   = float(mc_fc.iloc[-1])
        mc_upper = (
            float(mc_res.upper_bound.iloc[-1])
            if mc_res.upper_bound is not None and not mc_res.upper_bound.empty
            else mc_end * 1.10
        )
        mc_lower = (
            float(mc_res.lower_bound.iloc[-1])
            if mc_res.lower_bound is not None and not mc_res.lower_bound.empty
            else mc_end * 0.90
        )

        # Approximate distribution from CI bounds
        std_approx = max((mc_upper - mc_lower) / (2 * 1.645), mc_end * 0.005)
        rng        = np.random.default_rng(42)
        sim_ends   = rng.normal(mc_end, std_approx, 3000)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=sim_ends, nbinsx=60,
            marker_color=_GW_PURPLE, opacity=0.75,
        ))
        for _vx, _vc, _vd, _vt, _vxa in [
            (mc_end,   _GW_GREEN, "solid", f"Median ₹{mc_end:,.1f}",   "left"),
            (mc_lower, _GW_RED,   "dash",  f"5th pct ₹{mc_lower:,.1f}", "right"),
            (mc_upper, _GW_GREEN, "dash",  f"95th pct ₹{mc_upper:,.1f}", "left"),
        ]:
            fig_mc.add_shape(
                type="line", x0=_vx, x1=_vx, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=_vc, width=1.5, dash=_vd),
            )
            fig_mc.add_annotation(
                x=_vx, y=0.95, yref="paper", text=_vt,
                showarrow=False, font=dict(color=_vc, size=9), xanchor=_vxa,
            )
        fig_mc.update_layout(
            height=270,
            paper_bgcolor="white", plot_bgcolor=_GW_GRAY,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=11, color=_GW_NAVY),
            xaxis=dict(title="End Price (₹)", gridcolor="#E5E7EB"),
            yaxis=dict(title="Simulated paths", gridcolor="#E5E7EB"),
            showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown(
        '<div style="font-size:0.72rem;color:#9CA3AF;text-align:center;margin-top:8px;">'
        "Not financial advice. Model forecasts are probabilistic — use for research only."
        "</div>",
        unsafe_allow_html=True,
    )
