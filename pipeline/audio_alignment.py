"""
pipeline/audio_alignment.py

Detects the time offset between two audio files using FFT cross-correlation.
Designed for the case where a separate audio recording starts some time
after the video recording begins (e.g. video has an intro the audio lacks).

Returns a positive float representing how many seconds into the video
the separate audio file begins.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sample rate used for alignment — low rate is fast and sufficient for sync detection
ALIGNMENT_SAMPLE_RATE_HZ = 8000

# Duration in seconds to load from each file for comparison.
ALIGNMENT_COMPARISON_WINDOW_SECONDS = 300


def detect_audio_offset_seconds(
    video_file_path: str,
    separate_audio_file_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> float:
    """
    Compare the audio track of the video against the separate audio file
    to determine how many seconds into the video the separate audio begins.

    Returns offset in seconds. A return value of 45.3 means:
        subtitle_timestamp + 45.3 = video_timestamp
    """
    import librosa  # imported here to avoid slow startup when not needed

    def _log(msg: str) -> None:
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    _log("Loading video audio track (resampling to 8kHz mono)...")
    video_audio_samples, _ = librosa.load(
        video_file_path,
        sr=ALIGNMENT_SAMPLE_RATE_HZ,
        duration=ALIGNMENT_COMPARISON_WINDOW_SECONDS,
        mono=True,
    )
    video_duration = len(video_audio_samples) / ALIGNMENT_SAMPLE_RATE_HZ
    _log(f"Video audio loaded: {video_duration:.1f}s ({len(video_audio_samples)} samples)")

    _log("Loading separate audio track...")
    separate_audio_samples, _ = librosa.load(
        separate_audio_file_path,
        sr=ALIGNMENT_SAMPLE_RATE_HZ,
        duration=ALIGNMENT_COMPARISON_WINDOW_SECONDS,
        mono=True,
    )
    audio_duration = len(separate_audio_samples) / ALIGNMENT_SAMPLE_RATE_HZ
    _log(f"Audio loaded: {audio_duration:.1f}s ({len(separate_audio_samples)} samples)")

    _log("Running FFT cross-correlation...")
    offset_seconds = cross_correlate_to_find_offset(
        reference_audio=video_audio_samples,
        query_audio=separate_audio_samples,
        sample_rate=ALIGNMENT_SAMPLE_RATE_HZ,
    )

    _log(f"Detected audio offset: {offset_seconds:.3f} seconds")
    return offset_seconds


def cross_correlate_to_find_offset(
    reference_audio: np.ndarray,
    query_audio: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Use FFT-based normalized cross-correlation to find where query_audio
    best matches within reference_audio. Returns the offset in seconds.

    Pure function — no file I/O, fully testable with synthetic data.
    """
    normalized_reference = normalize_audio_amplitude(reference_audio)
    normalized_query = normalize_audio_amplitude(query_audio)

    # FFT-based correlation: O(N log N) instead of O(N^2)
    n = len(normalized_reference) + len(normalized_query) - 1
    fft_size = 1
    while fft_size < n:
        fft_size <<= 1

    fft_ref = np.fft.rfft(normalized_reference, fft_size)
    fft_query = np.fft.rfft(normalized_query, fft_size)
    correlation = np.fft.irfft(fft_ref * np.conj(fft_query), fft_size)[:n]

    # The peak of the correlation tells us the lag (offset) in samples
    peak_sample_index = int(np.argmax(correlation))

    # Convert from correlation index to actual lag in samples
    lag_in_samples = peak_sample_index - (len(normalized_query) - 1)

    offset_seconds = lag_in_samples / sample_rate
    return float(offset_seconds)


def normalize_audio_amplitude(audio_samples: np.ndarray) -> np.ndarray:
    """
    Normalize audio to zero mean and unit variance.
    Ensures cross-correlation is not dominated by volume differences.
    """
    mean = np.mean(audio_samples)
    standard_deviation = np.std(audio_samples)

    if standard_deviation == 0:
        return audio_samples - mean

    return (audio_samples - mean) / standard_deviation
