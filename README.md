# PALF - Pattern-Aware Learning Framework

Adaptive training framework for time-series forecasting.

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e .
    pip install pytest

## Run

    pytest -q
    streamlit run app/app.py --server.fileWatcherType none