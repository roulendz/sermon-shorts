#!/usr/bin/env bash
# setup.sh — works on Mac (M2) and Linux
# On Windows use: setup.ps1

echo "=== Sermon Shorts — Phase 1 Setup ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python $PYTHON_VERSION found"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg not found."
    echo "  Mac:     brew install ffmpeg"
    echo "  Windows: winget install ffmpeg"
    exit 1
fi
echo "FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"

# Create venv
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> Created .env from template — please fill in your API keys and file paths!"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Run pipeline:   python main.py"
