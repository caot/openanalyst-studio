# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="OpenAnalyst Studio" \
      org.opencontainers.image.description="Chat-with-your-data: Streamlit + LangChain" \
      org.opencontainers.image.url="https://github.com/<owner>/openanalyst-studio" \
      org.opencontainers.image.source="https://github.com/<owner>/openanalyst-studio"

# OS deps (minimal; keep wheels fast)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

# Create isolated venv
ENV VENV_PATH=/opt/venv
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

# Install Python deps first to maximize layer cache
# (ensure you have a requirements.txt at repo root)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Non-root user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# App env (override at run-time as needed)
# OPENAI_API_KEY is read by your code; set it via -e at runtime.
ENV APP_FILE="app.py"

EXPOSE 8501

# Lightweight healthcheck (TCP connect)
HEALTHCHECK --interval=30s --timeout=3s --retries=10 \
  CMD python - <<'PY'\nimport socket,sys\ns=socket.socket();s.settimeout(2)\ntry:\n s.connect(('127.0.0.1',8501))\nexcept Exception as e:\n print(e);sys.exit(1)\nfinally:\n s.close()\nPY

# Start Streamlit
CMD ["bash","-lc","streamlit run ${APP_FILE}"]
