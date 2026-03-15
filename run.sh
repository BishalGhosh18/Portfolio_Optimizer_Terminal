#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NSE/BSE Portfolio Optimizer — launcher
# Works with: conda env, plain venv, or system Python
# Usage:  bash run.sh [port]          e.g.  bash run.sh 8502
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="portfolio_optimizer"

# ── 1. Find Python & Streamlit ───────────────────────────────────────────────

find_streamlit() {
    # Priority: active venv/conda → local .venv → named conda env → system
    if command -v streamlit &>/dev/null; then
        echo "$(command -v streamlit)"
        return
    fi
    # Local .venv
    if [ -f "$SCRIPT_DIR/.venv/bin/streamlit" ]; then
        echo "$SCRIPT_DIR/.venv/bin/streamlit"
        return
    fi
    # Common conda locations
    for BASE in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/opt/anaconda3" \
                "/opt/anaconda3" "/opt/miniconda3" "/usr/local/anaconda3"; do
        if [ -f "$BASE/envs/$ENV_NAME/bin/streamlit" ]; then
            echo "$BASE/envs/$ENV_NAME/bin/streamlit"
            return
        fi
    done
    echo ""
}

find_python() {
    for BASE in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/opt/anaconda3" \
                "/opt/anaconda3" "/opt/miniconda3" "/usr/local/anaconda3"; do
        if [ -f "$BASE/envs/$ENV_NAME/bin/python" ]; then
            echo "$BASE/envs/$ENV_NAME/bin/python"
            return
        fi
    done
    if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
        echo "$SCRIPT_DIR/.venv/bin/python"
        return
    fi
    command -v python3 || command -v python
}

STREAMLIT=$(find_streamlit)

# ── 2. Auto-create environment if nothing found ──────────────────────────────

if [ -z "$STREAMLIT" ]; then
    echo "No environment found. Setting up now..."

    # Try conda first
    CONDA_BIN=""
    for BASE in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/opt/anaconda3" \
                "/opt/anaconda3" "/opt/miniconda3" "/usr/local/anaconda3"; do
        if [ -f "$BASE/bin/conda" ]; then
            CONDA_BIN="$BASE/bin/conda"
            break
        fi
    done

    if [ -n "$CONDA_BIN" ]; then
        echo "Creating conda environment '$ENV_NAME' (Python 3.11)..."
        "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 -y
        # Find the newly created env
        CONDA_PREFIX="$(dirname "$CONDA_BIN")/../envs/$ENV_NAME"
        "$CONDA_PREFIX/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
        STREAMLIT="$CONDA_PREFIX/bin/streamlit"
    else
        echo "conda not found. Creating Python venv in .venv/ ..."
        python3 -m venv "$SCRIPT_DIR/.venv"
        "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
        "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
        STREAMLIT="$SCRIPT_DIR/.venv/bin/streamlit"
    fi
fi

# ── 3. Launch ────────────────────────────────────────────────────────────────

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   NSE/BSE PORTFOLIO OPTIMIZER                ║"
echo "  ║   http://localhost:$PORT                       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

cd "$SCRIPT_DIR"
"$STREAMLIT" run app.py \
    --server.port "$PORT" \
    --server.headless false \
    --browser.gatherUsageStats false
