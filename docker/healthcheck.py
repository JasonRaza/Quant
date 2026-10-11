"""Container health probe: exit 0 if the service answers on PORT/HEALTH_PATH.

The API serves /health; the Streamlit app serves /_stcore/health.
"""

import os
import sys
import urllib.request

url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}{os.environ.get('HEALTH_PATH', '/health')}"
try:
    with urllib.request.urlopen(url, timeout=4) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
