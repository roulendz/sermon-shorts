"""
tests/test_audio_alignment.py
Tests for pipeline/audio_alignment.py

Uses synthetic audio (a pure tone) to verify the cross-correlation
offset detection is accurate without needing real audio files.
"""

import numpy as np
import pytest
from pipeline.audio_alignment import (
    cross_correlate_to_find_offset,
    normalize_audio_amplitude,
)

SAMPLE_RATE = 8000
ONE_SECOND_OF_SAMPLES = SAMPLE_RATE


def generate_random_audio_with_seed(
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate deterministic white noise. Unlike a sine wave, white noise
    has a unique cross-correlation peak, making it reliable for offset detection tests.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sample_rate * duration_seconds)).astype(np.float32)


def test_cross_correlate_detects_zero_offset_when_audio_files_are_identical():
    audio = generate_random_audio_with_seed(duration_seconds=5.0)
    detected_offset = cross_correlate_to_find_offset(
        reference_audio=audio,
        query_audio=audio,
        sample_rate=SAMPLE_RATE,
    )
    assert abs(detected_offset) < 0.05  # within 50ms tolerance


def test_cross_correlate_detects_positive_offset_when_query_starts_later():
    full_audio = generate_random_audio_with_seed(duration_seconds=10.0)
    known_offset_seconds = 3.0
    known_offset_samples = int(known_offset_seconds * SAMPLE_RATE)

    # Reference is the full audio (like the video's audio track)
    reference_audio = full_audio
    # Query starts 3 seconds into the reference (like the separate audio recording)
    query_audio = full_audio[known_offset_samples:]

    detected_offset = cross_correlate_to_find_offset(
        reference_audio=reference_audio,
        query_audio=query_audio,
        sample_rate=SAMPLE_RATE,
    )

    assert abs(detected_offset - known_offset_seconds) < 0.1  # within 100ms tolerance


def test_cross_correlate_detects_offset_of_one_second():
    full_audio = generate_random_audio_with_seed(duration_seconds=8.0)
    known_offset_samples = ONE_SECOND_OF_SAMPLES

    reference_audio = full_audio
    query_audio = full_audio[known_offset_samples:]

    detected_offset = cross_correlate_to_find_offset(
        reference_audio=reference_audio,
        query_audio=query_audio,
        sample_rate=SAMPLE_RATE,
    )

    assert abs(detected_offset - 1.0) < 0.1


def test_normalize_audio_amplitude_produces_zero_mean():
    audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize_audio_amplitude(audio)
    assert abs(np.mean(normalized)) < 1e-10


def test_normalize_audio_amplitude_produces_unit_standard_deviation():
    audio = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    normalized = normalize_audio_amplitude(audio)
    assert abs(np.std(normalized) - 1.0) < 1e-10


def test_normalize_audio_amplitude_handles_constant_signal_without_division_by_zero():
    constant_audio = np.ones(100)
    normalized = normalize_audio_amplitude(constant_audio)
    # Should not raise — all values should be zero after subtracting the mean
    assert np.all(normalized == 0.0)


def test_normalize_audio_amplitude_does_not_mutate_input():
    audio = np.array([1.0, 2.0, 3.0])
    original_values = audio.copy()
    normalize_audio_amplitude(audio)
    np.testing.assert_array_equal(audio, original_values)
