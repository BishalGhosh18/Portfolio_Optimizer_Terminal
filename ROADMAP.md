# 🗺️ Project Roadmap

How the **Nifty-50 Portfolio Optimizer & Movement Predictor** grew — from a portfolio-analytics dashboard into a fundamentals-aware stock movement predictor — and where it's headed next.

**Project started:** 15 March 2026 · **Current phase:** Phase 4 (July 2026)

> Legend: ✅ Done · 🟡 In progress · ⏳ Planned

### Two-minute overviews

**For everyone** — what the tool does, in plain words:

![Non-technical overview](docs/non-technical-overview.svg)

**For engineers** — how it's built:

![Technical overview](docs/technical-overview.svg)

---

## Phase Overview

```mermaid
graph LR
    P0["🧱 Phase 0<br/>Foundation<br/>✅"] --> P1["🖥️ Phase 1<br/>Market Terminal<br/>✅"]
    P1 --> P2["📉 Phase 2<br/>Price Prediction v5<br/>✅ (legacy)"]
    P2 --> P3["🎯 Phase 3<br/>Movement Predictor<br/>✅"]
    P3 --> P4["🌐 Phase 4<br/>Context Intelligence<br/>+ Nifty-50 + Forecast<br/>✅"]
    P4 --> P5["🚀 Phase 5<br/>Signal Depth<br/>🟡 / ⏳"]

    style P0 fill:#00D09C22,stroke:#00D09C
    style P1 fill:#00D09C22,stroke:#00D09C
    style P2 fill:#9CA3AF22,stroke:#9CA3AF
    style P3 fill:#00D09C22,stroke:#00D09C
    style P4 fill:#5367FF22,stroke:#5367FF
    style P5 fill:#F59E0B22,stroke:#F59E0B
```

---

## Timeline

```mermaid
timeline
    title Evolution of the Project
    Phase 0 · Foundation : Live price feed : Risk analytics (VaR, Sharpe, drawdown) : Portfolio optimisation (5 strategies) : Interactive charts
    Phase 1 · Terminal : 7-panel market terminal : Indices, order book, heatmap : News feed + TextBlob sentiment
    Phase 2 · Prediction v5 : Walk-forward backtest engine : Prophet / SARIMA / Theta / XGB / LGB / LSTM : Conformal confidence bands : Ensemble + naive benchmark
    Phase 3 · Movement Predictor : Direction (up/down) classification : Feature engineering + backtest : Accuracy / precision / recall / F1 : Strategy vs buy&hold / random
    Phase 4 · Context Intelligence : Earnings + market-regime features : Volume buy-sell pressure (OBV/MFI/A-D) : News + Screener.in + analyst overlay : LGBM+XGB+LogReg ensemble : Long-term price forecast (1-4 mo) : Nifty-50 across whole app
    Phase 5 · Signal Depth : Multi-day horizon : Universe expansion : Probability calibration : Alerts & backtest exports
```

---

## Phase Detail

### 🧱 Phase 0 — Foundation ✅
The portfolio-analytics core.
- `data_fetcher.py` · `risk_engine.py` · `optimizer.py`
- Live Feed, Charts, Risk, Optimizer tabs
- Groww-style UI, auto-refresh

### 🖥️ Phase 1 — Market Terminal ✅
- `terminal_tab.py` · `terminal_utils.py`
- 7 panels: indices, TA chart, order book, news, geo map, strategy signals, sector heatmap
- News + TextBlob sentiment (NewsAPI → RSS fallback)

### 📉 Phase 2 — Price Prediction v5 ✅ *(now legacy)*
- `predictor.py`: one walk-forward backtest shared by all models
- Prophet · Holt-Winters · SARIMA · Theta · XGBoost · LightGBM · LSTM · Monte Carlo + Ensemble
- Split-conformal confidence bands, skill-vs-naive scoring
- *Superseded in the Predict tab by Phase 3–4; still supplies Charts-tab indicators.*

### 🎯 Phase 3 — Stock Movement Predictor ✅
- `movement_predictor.py`: predict **next-day direction**, not exact price
- Feature engineering → classifier → **walk-forward backtest** → evaluation
- Metrics: accuracy, precision, recall, F1 + trading backtest vs buy&hold / random guess
- Honest framing: 1-day direction accuracy hovers near 50% and the UI says so

### 🌐 Phase 4 — Context Intelligence + Nifty-50 + Forecast ✅
The current release. Prediction stops being "just past prices."
- **Backtestable features** added: earnings-cycle, market regime (Nifty + India VIX), **volume buy-sell pressure** (OBV / MFI / A-D)
- **Model upgrade**: LightGBM + XGBoost + Logistic Regression → **soft-voting ensemble** (dropped RandomForest)
- **Live context overlay** (`fundamentals.py`): per-stock news, Screener.in fundamentals, analyst targets → a bias that nudges today's call
- **Long-term price forecast** (`price_forecast.py`): Monte-Carlo GBM, dampened drift, analyst anchor → 1/2/3/4-month median + 50%/90% bands
- **Nifty-50** is now the whole app's universe
- **Three headline outputs**: Movement · Future Price · Accuracy

### 🚀 Phase 5 — Signal Depth 🟡 / ⏳
- ⏳ **Multi-day horizon** (3/5-day direction) — the biggest lever to push accuracy above baseline
- ⏳ **Universe expansion** beyond Nifty 50 (Next 50 / midcaps)
- ⏳ **Probability calibration** (isotonic / Platt) for trustworthy confidence %
- ⏳ **Delivery volume & FII/DII flows** as features (where obtainable)
- ⏳ **Alerts & exports** — signal notifications, backtest CSV/report download
- ⏳ **Sector / peer momentum** features

---

## System Architecture

```mermaid
flowchart TD
    subgraph Sources["📡 Data Sources"]
        YF["Yahoo Finance<br/>prices · OHLCV · earnings<br/>news · analyst targets · Nifty/VIX"]
        SCR["Screener.in<br/>P/E · ROCE · ROE · pros/cons"]
        NEWS["NewsAPI / RSS<br/>market news"]
    end

    subgraph Data["🗂️ Data Layer"]
        DF["data_fetcher.py"]
        FN["fundamentals.py<br/>NIFTY_50 · earnings · regime<br/>news · screener · analysts"]
    end

    subgraph Engines["⚙️ Analytics & ML Engines"]
        RISK["risk_engine.py"]
        OPT["optimizer.py"]
        MOVE["movement_predictor.py<br/>features → ensemble → backtest"]
        FC["price_forecast.py<br/>Monte-Carlo 1-4 mo"]
        PRED["predictor.py<br/>(legacy · TA indicators)"]
    end

    subgraph UI["🖥️ Streamlit UI — app.py"]
        T1["Live Feed"]
        T2["Terminal"]
        T3["Charts"]
        T4["Risk"]
        T5["Optimizer"]
        T6["🔮 Predict"]
        T7["Insights"]
    end

    YF --> DF
    YF --> FN
    SCR --> FN
    NEWS --> T2

    DF --> RISK & OPT & MOVE & FC & PRED
    FN --> MOVE & FC

    RISK --> T4
    OPT --> T5
    PRED --> T3
    MOVE --> T6
    FC --> T6
    DF --> T1 & T2
```

---

## Movement Predictor Pipeline

```mermaid
flowchart TD
    A["📥 Pull OHLCV + fundamentals<br/>(Nifty-50 stock)"] --> B

    subgraph B["🔧 Feature Engineering (~27 features)"]
        B1["Price / technical<br/>returns · MAs · RSI · MACD · momentum"]
        B2["Volume pressure<br/>OBV · MFI · A/D"]
        B3["Earnings cycle<br/>days to/since · last surprise"]
        B4["Market regime<br/>Nifty returns · India VIX"]
    end

    B --> C["✂️ Chronological split<br/>(most-recent slice held out)"]
    C --> D1["🌳 LightGBM"]
    C --> D2["🌳 XGBoost"]
    C --> D3["📏 Logistic Regression"]
    D1 & D2 & D3 --> E["🤝 Soft-Voting Ensemble<br/>(avg probability)"]

    E --> F["📊 Walk-forward backtest<br/>accuracy · precision · recall · F1<br/>strategy vs buy&hold / random"]

    G["🌐 Live context (NOT backtested)<br/>news · Screener.in · analysts · earnings"] --> H
    E --> H["🎯 Combined call<br/>model prob ± context bias"]

    F --> OUT1["① Movement<br/>▲/▼ + conviction"]
    H --> OUT1
    F --> OUT3["③ Accuracy<br/>backtested vs baseline"]

    A --> I["🔮 price_forecast.py<br/>Monte-Carlo GBM<br/>dampened drift + analyst anchor"]
    I --> OUT2["② Future Price<br/>1-4 mo median + range"]

    style B1 fill:#5367FF15
    style B2 fill:#00D09C15
    style B3 fill:#F59E0B15
    style B4 fill:#EC489915
    style OUT1 fill:#00D09C22,stroke:#00D09C
    style OUT2 fill:#5367FF22,stroke:#5367FF
    style OUT3 fill:#F59E0B22,stroke:#F59E0B
```

---

## Signal Honesty Model

A core design principle: every signal is tagged by whether it can be **validated**.

```mermaid
flowchart LR
    subgraph BT["✅ Backtestable — feeds the ML model"]
        direction TB
        b1["Price / technical"]
        b2["Volume pressure"]
        b3["Earnings cycle"]
        b4["Market regime"]
    end
    subgraph OV["⚠️ Live overlay — NOT backtested"]
        direction TB
        o1["News sentiment"]
        o2["Screener.in fundamentals"]
        o3["Analyst view"]
    end
    BT --> M["Walk-forward<br/>backtested accuracy"]
    OV --> N["Context bias<br/>(±10 pts nudge)"]
    M --> CALL["🎯 Final call<br/>honestly labelled"]
    N --> CALL
```

---

_Have an idea for a phase? The universe, features and models are all modular — see [README.md](README.md) for the module map._
