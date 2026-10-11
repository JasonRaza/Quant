# syntax=docker/dockerfile:1
# One image, two services: the paper-trading REST API (default) and the Streamlit app.
#   docker run -p 8000:8000 quant                      -> API, docs at /docs
#   docker run -p 8501:8501 -e PORT=8501 quant dashboard -> Streamlit app

ARG PYTHON_VERSION=3.12

# ---- builder: resolve and install dependencies into a virtualenv ----------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
# Only the metadata: dependency layers stay cached until pyproject.toml changes.
COPY pyproject.toml README.md LICENSE ./
# Install the "paper" extra (API + Streamlit + research modules), then drop the
# project itself: the code runs from /app so config/ and data/snapshot/ resolve.
# pip itself is removed too: nothing is installed at runtime (smaller attack surface).
RUN pip install ".[paper]" \
    && pip uninstall --yes quant-portfolio \
    && pip uninstall --yes pip

# ---- runtime ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PORT=8000 \
    HEALTH_PATH=/health \
    QUANT_PAPER_DB=/app/var/paper_trading.sqlite

# The base image ships its own pip; the app never installs anything at runtime.
RUN python -m pip uninstall --yes pip \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin quant

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY shared/ shared/
COPY data_pipeline/ data_pipeline/
COPY backtester/ backtester/
COPY factors/ factors/
COPY portfolio/ portfolio/
COPY ml_alpha/ ml_alpha/
COPY paper_trading/ paper_trading/
COPY config/ config/
COPY data/snapshot/ data/snapshot/
COPY .streamlit/ .streamlit/
COPY docker/ docker/

# The only writable location: the paper account database (mount a volume here).
RUN mkdir -p /app/var && chown quant:quant /app/var

USER quant
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]

ENTRYPOINT ["sh", "docker/entrypoint.sh"]
CMD ["api"]
