# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style Rules

- **No abbreviations** in variable names, function names, or parameter names. Use full descriptive words (e.g. `subtitle_file_path` not `sub_fp`, `request_timeout_seconds` not `req_timeout_s`).
- **DRY (Don't Repeat Yourself)** — extract shared logic into reusable functions rather than duplicating code across modules.
- **SRP (Single Responsibility Principle)** — each function and class should do one thing. Keep API clients, business logic, and UI concerns in separate modules.

## What This Is

A Python TUI app (Textual framework) that processes sermon recordings into viral-worthy short clips for TikTok/YouTube Shorts/Instagram Reels. Pipeline: file selection -> audio alignment -> WhisperX transcription -> Manus AI segment selection -> FFmpeg video cutting.

## Commands

```bash
# Run the app
python main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_segment_selector.py

# Run a single test
pytest tests/test_segment_selector.py::test_name
```

## Architecture

**Pipeline flow** — each TUI screen advances one pipeline step, passing `PipelineState` (single dataclass carrier) through:

1. `FileSelectionScreen` — pick video + audio files
2. `AudioAlignmentScreen` — FFT cross-correlation to find audio-to-video offset (or manual entry)
3. `TranscriptionScreen` — TUS chunked upload to WhisperX, polls for SRT, shifts timestamps by offset
4. `SegmentReviewScreen` — sends SRT to Manus AI, parses JSON response with viral scores
5. `CuttingScreen` — FFmpeg stream-copy cuts segments into .mp4 + .srt pairs

**Module boundaries:**
- `api/` — HTTP-only clients (WhisperX TUS protocol, Manus task submission + polling). No business logic.
- `pipeline/` — Pure business logic (prompt building, JSON parsing, subtitle manipulation, FFmpeg commands). No HTTP, no TUI.
- `tui/screens/` — Glue layer. Imports from both `api/` and `pipeline/`, runs work in threads via `@work` decorator.
- `models/` — `PipelineState` dataclass (pipeline carrier) and `VideoSegment` dataclass (single clip).

**Key patterns:**
- TUI screens use `@work(thread=True)` for non-blocking API calls, with `self.app.call_from_thread()` for UI updates
- API clients accept `on_progress` callbacks to feed TUI log widgets
- WhisperX uses TUS resumable upload with 50MB chunks (Cloudflare body size limit)
- Manus JSON extraction handles markdown fences and variable response structures
- Audio alignment uses librosa + NumPy FFT for O(N log N) cross-correlation

## External Dependencies

- **FFmpeg** must be installed and on PATH
- **WhisperX API** at `WHISPERX_API_URL` — speech-to-text via TUS upload
- **Manus AI API** at `MANUS_API_URL` — agent-based segment selection

## Audio Filename Convention

Audio filename suffix determines language: `_A03` = Latvian, `_A04` = Russian, `_A05` = English. Default is Latvian. See `pipeline/language_detect.py`.

## Testing

Tests use `pytest` + `respx` (HTTP mocking). Fixtures in `tests/conftest.py`. Tests never hit real APIs.

## File Layout

Module boundaries above are the layout: `api/`, `pipeline/`, `tui/screens/`,
`models/`, `tests/`. Nothing loose in the repo root beyond `main.py` and the
Python manifests.

One-shot operations (a batch re-cut, a backfill, a data migration) go in
`runs/YYYY-MM-DD-<slug>/` holding `NOTES.md`, `plan.json`, `apply.py`,
`rollback.json`, `log.txt`. Never `-v2` / `-new` / `-final` filename suffixes —
a second attempt is a new run directory. Full rules: `~/.claude/CLAUDE.md`.
