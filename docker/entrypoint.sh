#!/bin/sh
# Start one of the two services. PORT is set by the platform (Azure, compose, ...).
set -eu

case "${1:-api}" in
  api)
    exec uvicorn paper_trading.api:create_app --factory \
      --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
    ;;
  dashboard)
    exec streamlit run paper_trading/app.py \
      --server.address=0.0.0.0 --server.port="${PORT:-8501}" \
      --server.headless=true --browser.gatherUsageStats=false \
      --server.fileWatcherType=none
    ;;
  *)
    exec "$@"
    ;;
esac
