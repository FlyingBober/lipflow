FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    python3-tk \
    xclip \
    wl-clipboard \
    v4l-utils \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-tk \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/bin/python3.12 && uv sync --frozen

# Copy source code and helper scripts
COPY lipflow/ ./lipflow/
COPY espnet/ ./espnet/
COPY scripts/ ./scripts/
COPY README.md LICENSE NOTICE ./

# Re-install package in editable mode
RUN uv pip install -e . --no-deps

# Create directories for models and persistent user data
RUN mkdir -p /app/models /root/.local/share/lipflow

# Default entrypoint runs lipflow inside virtualenv
ENTRYPOINT ["uv", "run", "--no-sync", "lipflow"]
CMD ["doctor"]
