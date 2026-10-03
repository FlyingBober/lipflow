#!/usr/bin/env bash
# Helper script to run Lipflow inside Docker on Linux with GUI, webcam and GPU pass-through.
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if docker is installed
if ! command -v docker >/dev/null; then
    echo "Error: docker is not installed or not in PATH."
    exit 1
fi

# Clean up socks5 proxies if present to prevent Docker BuildKit lookup errors
if [[ "${HTTP_PROXY:-}" =~ socks5 ]] || [[ "${HTTPS_PROXY:-}" =~ socks5 ]]; then
    export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY=
fi

# Allow root inside container to draw onto current X11 display
if [ -n "${DISPLAY:-}" ]; then
    xhost +local:root >/dev/null 2>&1 || true
fi

# Determine GPU flags
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_FLAGS=(--gpus all)
fi

# Determine video device
DEV_FLAGS=()
if [ -e "/dev/video0" ]; then
    DEV_FLAGS+=(--device=/dev/video0:/dev/video0)
fi
if [ -e "/dev/snd" ]; then
    DEV_FLAGS+=(--device=/dev/snd:/dev/snd)
fi

# Build image if it doesn't exist
if ! docker image inspect lipflow:latest >/dev/null 2>&1; then
    echo "Building lipflow:latest Docker image..."
    docker build -t lipflow:latest .
fi

echo "Starting Lipflow in Docker..."
exec docker run --rm -it \
    --network=host \
    --ipc=host \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v "${PWD}/models:/app/models" \
    -v "${HOME}/.local/share/lipflow:/root/.local/share/lipflow" \
    "${GPU_FLAGS[@]}" \
    "${DEV_FLAGS[@]}" \
    lipflow:latest "$@"
