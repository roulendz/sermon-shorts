# setup.ps1 — Windows PowerShell 7+ setup script
# Run with: pwsh -ExecutionPolicy Bypass -File setup.ps1

Write-Host "=== Sermon Shorts — Phase 1 Setup ===" -ForegroundColor Cyan

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "Python found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# Check FFmpeg
try {
    $ffVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "FFmpeg found: $ffVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: FFmpeg not found." -ForegroundColor Red
    Write-Host "  Install with: winget install ffmpeg" -ForegroundColor Yellow
    exit 1
}

# Create venv
Write-Host "`nCreating virtual environment..." -ForegroundColor Cyan
python -m venv .venv

# Activate and install
Write-Host "Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Copy .env if not exists
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "`n>>> Created .env from template — please fill in your API keys and file paths!" -ForegroundColor Yellow
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate venv:  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Run pipeline:   python main.py" -ForegroundColor White
