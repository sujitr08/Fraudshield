#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! python -m streamlit --version >/dev/null 2>&1; then
  echo "Streamlit is not installed. Install dependencies with: python -m pip install -r requirements.txt" >&2
  exit 1
fi
python -m streamlit run app.py --server.headless false
