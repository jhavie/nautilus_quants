# === Builder stage ===
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Layer cache: install dependencies first
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Install project
COPY src/ ./src/
RUN uv sync --frozen --no-editable

# === Runtime stage ===
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 procps && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Deterministic floating-point: force OpenBLAS single-thread
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

COPY src/ ./src/
COPY config/ ./config/
RUN mkdir -p /app/logs /app/data

RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD pgrep -f "nautilus_quants" || exit 1

# Default entrypoint: live trading CLI
ENTRYPOINT ["python", "-m", "nautilus_quants.live"]
CMD ["--help"]
