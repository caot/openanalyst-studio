# OpenAnalyst Studio — Streamlit + LangChain Agent

A production-ready Streamlit app that lets you **chat with your data**. It uses a **LangChain agent** with **typed, schema-validated tools** to generate insights and Plotly charts, plus robust logging, retries, tests, CI, and containerization.

## Features
- **Chat with CSV/Excel**: automatic cleaning, column classification, stats & visuals
- **Tooling with types**: `BaseTool` subclasses + Pydantic I/O (`AnalysisTool`, `VisualizationTool`)
- **Stable LLM wrapper**: `get_llm()` with secrets/env precedence, JSON-mode support, streaming option
- **Reliability**: tenacity retries, structured logs with `structlog`, defensive parsing
- **Quality gates**: Ruff/Black/Mypy, pytest + coverage
- **CI/CD**: GitHub Actions for lint/test, container build (GHCR), releases, CodeQL, Trivy, optional Snyk
- **Containerized**: Containerfile for Docker/Podman; non-root user; healthcheck
- **Optional local models**: call an Ollama server via `OLLAMA_BASE_URL`

## Repo layout (key files)
```
.
├─ openanalyst.py                          # Streamlit entrypoint
├─ tools/
│  ├─ __init__.py                  # agent wiring (lazy imports)
│  ├─ llm.py                       # LLM factory (JSON mode, streaming)
│  ├─ schemas.py                   # Pydantic I/O models
│  ├─ analysis_tool.py             # AnalysisTool (JSON mode + streaming)
│  ├─ visualization_tool.py        # VisualizationTool (returns ChartSpec)
├─ utils/
│  └─ logging_utils.py             # structlog config + timed() context
├─ tests/
│  ├─ test_analysis_tool.py        # unit tests (fake LLMs)
│  └─ test_visualization_tool.py
├─ Containerfile                   # production container
├─ requirements.txt
└─ .github/workflows/              # CI/CD (CI, container, release, security)
```

## Quick start (local dev)
```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI key (or use .streamlit/secrets.toml)
export OPENAI_API_KEY=sk-...

# Run the app
streamlit run openanalyst.py
# → http://localhost:8501
```

**Secrets via Streamlit (optional)**
```
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = "0.2"
OPENAI_TIMEOUT = "60"
```

## Run with Docker/Podman
```bash
# Build
podman build -t openanalyst-studio:dev -f Containerfile .
# docker build -t openanalyst-studio:dev -f Containerfile .

# Run
podman run --rm -p 8501:8501   -e OPENAI_API_KEY="$OPENAI_API_KEY"   -e APP_FILE="openanalyst.py"   --name openanalyst-studio   openanalyst-studio:dev
# → http://localhost:8501
```

**Mount Streamlit secrets**
```bash
podman run --rm -p 8501:8501   -e OPENAI_API_KEY="$OPENAI_API_KEY"   -v "$PWD/.streamlit:/app/.streamlit:ro,Z"   openanalyst-studio:dev
```

**Call a host Ollama server (optional)**
```bash
# macOS/Windows
docker run --rm -p 8501:8501   -e OPENAI_API_KEY="$OPENAI_API_KEY"   -e OLLAMA_BASE_URL="http://host.docker.internal:11434"   openanalyst-studio:dev
```

## Tests & quality
```bash
# Unit tests
pytest -q

# Lint & type check
ruff check .
black --check .
mypy .
```

## GitHub Actions (CI/CD)
- **CI**: lint (Ruff/Black), mypy, pytest + coverage (Codecov if `CODECOV_TOKEN` set)
- **Container**: build & push to GHCR (`ghcr.io/<owner>/openanalyst-studio:latest`) on main/tags
- **Release Please**: automated changelog & GitHub releases
- **CodeQL**: static analysis
- **Trivy**: filesystem + image scanning (uploads SARIF to Security tab)
- **Snyk** (optional): guarded by `SNYK_TOKEN` secret  
  _Note_: SARIF uploads on forked PRs are skipped (token is read-only).

## Configuration
Environment variables (or `secrets.toml`):
- `OPENAI_API_KEY` *(required for OpenAI models)*
- `MODEL_NAME` (default `gpt-4o-mini`)
- `TEMPERATURE` (default `0.2`)
- `OPENAI_TIMEOUT` (default `60`)
- `OLLAMA_BASE_URL` *(optional, e.g., `http://localhost:11434`)*

## Troubleshooting
- **Timeouts / slow first token**: enable streaming (`streaming=True` in LLM) or increase `OPENAI_TIMEOUT`.
- **“argument of type 'NoneType' is not iterable”**: fixed by defensive parsing and ensuring `model_kwargs` is always a dict.
- **No stack traces**: ensure `configure_logging()` runs once; log with `.exception(...)`.
- **Streamlit keys**: use stable keys derived from message IDs or spec hashes.

## License
MIT License
