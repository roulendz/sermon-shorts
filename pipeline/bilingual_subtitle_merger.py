"""
pipeline/bilingual_subtitle_merger.py

Merges two single-language subtitle tracks (e.g. RU pastor + LV translator)
into one bilingual SRT file with precise timing from FFmpeg silence detection.

Flow:
  1. Detect speech regions in each audio track via silence detection
  2. Resolve any overlaps using split-the-middle strategy
  3. Split Transkriptor entries at silence boundaries (they merge across gaps)
  4. Distribute words proportionally across speech chunks
  5. Build merged SRT sorted by timestamp with [RU]/[LV] tags
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

import srt

from pipeline.silence_detection import (
    detect_silence_regions,
    silence_regions_to_speech_regions,
    get_audio_duration_seconds,
    DEFAULT_NOISE_THRESHOLD_DB,
    DEFAULT_MINIMUM_SILENCE_DURATION_SECONDS,
)


def merge_bilingual_subtitles(
    primary_audio_path: Path,
    primary_language_tag: str,
    primary_content: list[dict],
    secondary_audio_path: Path,
    secondary_language_tag: str,
    secondary_content: list[dict],
    output_srt_path: Path,
    audio_to_video_offset_seconds: float = 0.0,
    noise_threshold_db: int = DEFAULT_NOISE_THRESHOLD_DB,
    minimum_silence_duration_seconds: float = DEFAULT_MINIMUM_SILENCE_DURATION_SECONDS,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Merge two language tracks into one bilingual SRT file.

    primary/secondary_content: raw Transkriptor content arrays
    (list of dicts with 'text', 'StartTime', 'EndTime' in milliseconds).
    """
    def _log(message: str) -> None:
        if on_progress:
            on_progress(message)

    # Step 1: Detect speech regions via silence analysis
    _log(f"Analyzing speech regions in {primary_language_tag} track...")
    primary_duration = get_audio_duration_seconds(str(primary_audio_path))
    primary_silences = detect_silence_regions(
        str(primary_audio_path), noise_threshold_db,
        minimum_silence_duration_seconds, on_progress=on_progress,
    )
    primary_speech_raw = silence_regions_to_speech_regions(primary_silences, primary_duration)
    primary_speech = [(s, e) for s, e in primary_speech_raw if e - s >= 0.5]
    _log(f"  {primary_language_tag}: {len(primary_speech_raw)} raw → {len(primary_speech)} speech regions (≥0.5s)")

    _log(f"Analyzing speech regions in {secondary_language_tag} track...")
    secondary_duration = get_audio_duration_seconds(str(secondary_audio_path))
    secondary_silences = detect_silence_regions(
        str(secondary_audio_path), noise_threshold_db,
        minimum_silence_duration_seconds, on_progress=on_progress,
    )
    secondary_speech_raw = silence_regions_to_speech_regions(secondary_silences, secondary_duration)
    secondary_speech = [(s, e) for s, e in secondary_speech_raw if e - s >= 0.5]
    _log(f"  {secondary_language_tag}: {len(secondary_speech_raw)} raw → {len(secondary_speech)} speech regions (≥0.5s)")

    # Step 2a: Remove mic bleed — only on secondary (LV) track.
    # Primary (RU) is the priority speaker, never remove RU speech.
    # LV speech that overlaps with RU speech is mic bleed from pastor's mic.
    _log(f"Removing mic bleed on {secondary_language_tag} track (RU priority — never remove {primary_language_tag})...")
    secondary_speech = _remove_mic_bleed(secondary_speech, primary_speech)
    _log(f"  {secondary_language_tag}: {len(secondary_speech)} regions after bleed removal")

    # Step 2b: RU priority — when RU talks over LV start, push LV start
    # forward to align with RU silence start (RU always finishes first).
    _log("Aligning LV to RU boundaries (RU priority)...")
    secondary_speech_before_align = len(secondary_speech)
    secondary_speech = _align_secondary_to_primary(
        primary_speech, secondary_speech,
    )
    _log(f"  {secondary_language_tag}: {secondary_speech_before_align} → {len(secondary_speech)} regions after alignment")

    # Step 2c: Fill gaps using alternation logic (RU→LV→RU→LV)
    # If gap exists between two RU regions → must be LV (detection missed it)
    # If gap exists between two LV regions → must be RU
    # If gap between RU and LV → split at midpoint
    _log("Filling gaps using alternation logic (no white gaps)...")
    primary_before_fill = len(primary_speech)
    secondary_before_fill = len(secondary_speech)
    primary_speech, secondary_speech = _fill_gaps_by_alternation(
        primary_speech, secondary_speech,
    )
    _log(f"  {primary_language_tag}: {primary_before_fill} → {len(primary_speech)} regions")
    _log(f"  {secondary_language_tag}: {secondary_before_fill} → {len(secondary_speech)} regions")

    # Step 2d: Smart merge — if a silence gap in one track also has silence
    # in the other track, it's just a pause (same speaker continues).
    # Only keep silence boundaries where the OTHER track has speech.
    _log("Smart-merging speech regions (pause vs speaker switch)...")
    primary_speech_before = len(primary_speech)
    secondary_speech_before = len(secondary_speech)

    primary_speech = _merge_pauses_where_other_track_is_silent(
        primary_speech, secondary_speech,
    )
    secondary_speech = _merge_pauses_where_other_track_is_silent(
        secondary_speech, primary_speech,
    )
    _log(f"  {primary_language_tag}: {primary_speech_before} → {len(primary_speech)} regions")
    _log(f"  {secondary_language_tag}: {secondary_speech_before} → {len(secondary_speech)} regions")

    # Step 3: Resolve fine-grained overlaps
    overlaps_before = _find_overlaps(primary_speech, secondary_speech)
    if overlaps_before:
        _log(f"Found {len(overlaps_before)} overlaps ({sum(d for _,_,d in overlaps_before):.2f}s) — resolving...")
        primary_speech, secondary_speech = _split_overlaps_in_middle(primary_speech, secondary_speech)

    # Step 4: Split content at silence boundaries (perfect alignment)
    _log(f"Splitting {primary_language_tag} entries at silence boundaries...")
    primary_entries = _split_content_at_silence_boundaries(
        primary_content, primary_speech, primary_language_tag,
    )
    _log(f"  {primary_language_tag}: {len(primary_entries)} entries")

    _log(f"Splitting {secondary_language_tag} entries at silence boundaries...")
    secondary_entries = _split_content_at_silence_boundaries(
        secondary_content, secondary_speech, secondary_language_tag,
    )
    _log(f"  {secondary_language_tag}: {len(secondary_entries)} entries")

    # Step 4: Merge, sort, enforce alternation, close gaps
    all_entries = primary_entries + secondary_entries
    all_entries.sort(key=lambda e: e["start_seconds"])

    _log("Enforcing strict alternation and closing gaps...")
    all_entries = _merge_consecutive_same_language(all_entries)
    all_entries = _close_gaps_between_entries(all_entries)
    _log(f"  {len(all_entries)} final subtitle entries")

    offset = timedelta(seconds=audio_to_video_offset_seconds)
    subtitles = []
    for index, entry in enumerate(all_entries, start=1):
        start_time = timedelta(seconds=entry["start_seconds"]) + offset
        end_time = timedelta(seconds=entry["end_seconds"]) + offset

        if start_time < timedelta(0):
            continue

        tagged_text = f"[{entry['language_tag']}] {entry['text']}"
        subtitles.append(srt.Subtitle(
            index=index,
            start=start_time,
            end=end_time,
            content=tagged_text,
        ))

    # Reindex after filtering
    for new_index, subtitle in enumerate(subtitles, start=1):
        subtitle.index = new_index

    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    output_srt_path.write_text(srt.compose(subtitles), encoding="utf-8")
    _log(f"Merged bilingual SRT saved: {output_srt_path} ({len(subtitles)} entries)")

    return output_srt_path


def load_transkriptor_content_from_json(json_file_path: Path) -> list[dict]:
    """Load raw Transkriptor content array from a saved JSON file."""
    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if "body" in data and isinstance(data["body"], dict):
        return data["body"].get("content", [])
    return data.get("content", [])


def _split_content_at_silence_boundaries(
    content_entries: list[dict],
    speech_regions: list[tuple[float, float]],
    language_tag: str,
) -> list[dict]:
    """
    Split Transkriptor entries that span multiple speech regions.

    Transkriptor ignores silences and groups text into sentences.
    When an entry spans multiple speech regions, we:
      1. Split text into sentences (by period boundaries)
      2. Assign whole sentences to speech regions by duration proportion
      3. Short tail sentences (1-3 words ending with .) stay with previous chunk
      4. Only split within a sentence as last resort
    """
    subtitle_entries = []

    for content_entry in content_entries:
        entry_start_seconds = content_entry["StartTime"] / 1000.0
        entry_end_seconds = content_entry["EndTime"] / 1000.0
        text = content_entry.get("text", "").strip()

        if not text:
            continue

        # Find all speech regions that overlap with this entry
        covering_regions = _find_covering_speech_regions(
            entry_start_seconds, entry_end_seconds, speech_regions,
        )

        if not covering_regions:
            subtitle_entries.append({
                "start_seconds": entry_start_seconds,
                "end_seconds": entry_end_seconds,
                "text": text,
                "language_tag": language_tag,
            })
            continue

        if len(covering_regions) == 1:
            region_start, region_end = covering_regions[0]
            subtitle_entries.append({
                "start_seconds": max(entry_start_seconds, region_start),
                "end_seconds": min(entry_end_seconds, region_end),
                "text": text,
                "language_tag": language_tag,
            })
            continue

        # Multiple speech regions — split by sentence boundaries
        sentences = _split_into_sentences(text)

        region_durations = [
            min(region_end, entry_end_seconds) - max(region_start, entry_start_seconds)
            for region_start, region_end in covering_regions
        ]
        total_speech_duration = sum(d for d in region_durations if d > 0)

        if total_speech_duration <= 0 or not sentences:
            subtitle_entries.append({
                "start_seconds": entry_start_seconds,
                "end_seconds": entry_end_seconds,
                "text": text,
                "language_tag": language_tag,
            })
            continue

        # Assign sentences to regions proportionally by duration
        chunks = _assign_sentences_to_regions(
            sentences, covering_regions, region_durations, total_speech_duration,
        )

        for chunk_text, region_start, region_end in chunks:
            chunk_start = max(entry_start_seconds, region_start)
            chunk_end = min(entry_end_seconds, region_end)
            if chunk_end <= chunk_start or not chunk_text.strip():
                continue
            subtitle_entries.append({
                "start_seconds": chunk_start,
                "end_seconds": chunk_end,
                "text": chunk_text.strip(),
                "language_tag": language_tag,
            })

    return subtitle_entries


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence fragments at period boundaries.
    Keeps the period with the sentence it ends.
    Merges short tails (1-3 words ending with .) into the previous sentence.

    "First sentence. Second one. OK." → ["First sentence. Second one.", "OK."]
    but if timing allows, "OK." stays with previous.
    """
    import re

    # Split at period followed by space (keep period with left side)
    raw_parts = re.split(r'(?<=\.)\s+', text)

    # Merge short tails (1-3 words ending with period) into previous
    merged = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        word_count = len(part.split())
        if (merged
                and word_count <= 3
                and part.endswith(".")
                and not merged[-1].endswith(".")):
            # Short tail after incomplete previous — merge
            merged[-1] = merged[-1] + " " + part
        elif (merged
                and word_count <= 3
                and part.endswith(".")
                and len(merged[-1].split()) <= 3):
            # Both are short — merge
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return merged if merged else [text]


def _assign_sentences_to_regions(
    sentences: list[str],
    covering_regions: list[tuple[float, float]],
    region_durations: list[float],
    total_speech_duration: float,
) -> list[tuple[str, float, float]]:
    """
    Assign sentences to speech regions proportionally by duration.
    Returns list of (chunk_text, region_start, region_end).

    Strategy:
    - Calculate total word count
    - Each region gets words proportional to its duration
    - Assign whole sentences to regions, preferring sentence boundaries
    - If a single sentence must be split, split by words within it
    """
    total_word_count = sum(len(sentence.split()) for sentence in sentences)
    if total_word_count == 0:
        return []

    result_chunks = []
    sentence_index = 0
    words_assigned = 0

    for region_index, (region_start, region_end) in enumerate(covering_regions):
        region_duration = region_durations[region_index]
        if region_duration <= 0 or sentence_index >= len(sentences):
            continue

        # How many words should this region get?
        proportion = region_duration / total_speech_duration
        target_word_count = round(total_word_count * proportion)

        # At least 1 word per region
        target_word_count = max(1, target_word_count)

        # Last region gets everything remaining
        is_last_region = region_index == len(covering_regions) - 1

        chunk_sentences = []
        chunk_word_count = 0

        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            sentence_word_count = len(sentence.split())

            if is_last_region:
                # Last region takes all remaining sentences
                chunk_sentences.append(sentence)
                chunk_word_count += sentence_word_count
                sentence_index += 1
                continue

            # Would adding this sentence exceed the target?
            if chunk_word_count + sentence_word_count > target_word_count and chunk_sentences:
                # Check if this sentence is short (1-3 words with period)
                # and fits better with current chunk
                if sentence_word_count <= 3 and sentence.rstrip().endswith("."):
                    # Short sentence ending — keep with current chunk
                    chunk_sentences.append(sentence)
                    chunk_word_count += sentence_word_count
                    sentence_index += 1
                break

            chunk_sentences.append(sentence)
            chunk_word_count += sentence_word_count
            sentence_index += 1

            # If we've reached the target and ended at a sentence boundary, stop
            if chunk_word_count >= target_word_count:
                break

        if chunk_sentences:
            chunk_text = " ".join(chunk_sentences)
            result_chunks.append((chunk_text, region_start, region_end))
            words_assigned += chunk_word_count

    return result_chunks


def _identify_speaking_turns(
    primary_speech: list[tuple[float, float]],
    primary_language_tag: str,
    secondary_speech: list[tuple[float, float]],
    secondary_language_tag: str,
) -> list[dict]:
    """
    Identify speaking turns from two tracks' speech regions.

    A "turn" is the entire contiguous time one speaker is active before
    the other takes over. Multiple speech fragments from the same speaker
    (with small internal pauses) are merged into one turn.

    Returns seamless turns: when one ends, the next begins immediately.
    Pattern is always: primary → secondary → primary → secondary...
    """
    # Tag and merge all speech regions
    tagged_regions = []
    for start, end in primary_speech:
        tagged_regions.append((start, end, primary_language_tag))
    for start, end in secondary_speech:
        tagged_regions.append((start, end, secondary_language_tag))
    tagged_regions.sort(key=lambda region: region[0])

    if not tagged_regions:
        return []

    # Group consecutive same-language regions into turns
    turns = []
    current_language = tagged_regions[0][2]
    current_start = tagged_regions[0][0]
    current_end = tagged_regions[0][1]

    for region_start, region_end, language_tag in tagged_regions[1:]:
        if language_tag == current_language:
            # Same speaker continues — extend the turn
            current_end = max(current_end, region_end)
        else:
            # Speaker changed — close current turn, start new one
            turns.append({
                "start_seconds": current_start,
                "end_seconds": current_end,
                "language_tag": current_language,
            })
            current_language = language_tag
            current_start = region_start
            current_end = region_end

    # Close last turn
    turns.append({
        "start_seconds": current_start,
        "end_seconds": current_end,
        "language_tag": current_language,
    })

    # Make turns seamless: close gaps so each turn ends where the next begins
    for i in range(len(turns) - 1):
        gap_start = turns[i]["end_seconds"]
        gap_end = turns[i + 1]["start_seconds"]
        if gap_end > gap_start:
            # Split the gap at midpoint
            midpoint = (gap_start + gap_end) / 2
            turns[i]["end_seconds"] = midpoint
            turns[i + 1]["start_seconds"] = midpoint

    return turns


def _assign_text_to_turns(
    turns: list[dict],
    primary_content: list[dict],
    primary_language_tag: str,
    secondary_content: list[dict],
    secondary_language_tag: str,
) -> list[dict]:
    """
    Assign Transkriptor text to each speaking turn.

    For each turn, find all Transkriptor content entries whose midpoint
    falls within the turn's time range. Concatenate their text.
    """
    entries = []

    for turn in turns:
        turn_start = turn["start_seconds"]
        turn_end = turn["end_seconds"]
        language_tag = turn["language_tag"]

        # Pick the right content source
        if language_tag == primary_language_tag:
            content_source = primary_content
        else:
            content_source = secondary_content

        # Find content entries whose midpoint falls in this turn
        matching_texts = []
        for content_entry in content_source:
            content_start = content_entry["StartTime"] / 1000.0
            content_end = content_entry["EndTime"] / 1000.0
            content_midpoint = (content_start + content_end) / 2.0

            if turn_start <= content_midpoint <= turn_end:
                text = content_entry.get("text", "").strip()
                if text:
                    matching_texts.append(text)

        if not matching_texts:
            continue

        combined_text = " ".join(matching_texts)
        entries.append({
            "start_seconds": turn_start,
            "end_seconds": turn_end,
            "text": combined_text,
            "language_tag": language_tag,
        })

    return entries


def _build_turns_from_content(
    primary_content: list[dict],
    primary_language_tag: str,
    secondary_content: list[dict],
    secondary_language_tag: str,
) -> list[dict]:
    """
    Build speaking turns directly from Transkriptor content entries.
    Sort all entries by timestamp, group consecutive same-language into turns.
    Guarantees strict RU → LV → RU → LV alternation.
    """
    # Tag all entries with their language
    all_entries = []
    for entry in primary_content:
        text = entry.get("text", "").strip()
        if text:
            all_entries.append({
                "start_ms": entry["StartTime"],
                "end_ms": entry["EndTime"],
                "text": text,
                "language_tag": primary_language_tag,
            })
    for entry in secondary_content:
        text = entry.get("text", "").strip()
        if text:
            all_entries.append({
                "start_ms": entry["StartTime"],
                "end_ms": entry["EndTime"],
                "text": text,
                "language_tag": secondary_language_tag,
            })

    # Sort by start time
    all_entries.sort(key=lambda e: e["start_ms"])

    # Group consecutive same-language entries into turns
    turns = []
    current_language = None
    current_texts = []
    current_start_ms = 0
    current_end_ms = 0

    for entry in all_entries:
        if entry["language_tag"] != current_language:
            # Language changed — save previous turn
            if current_texts and current_language:
                turns.append({
                    "start_seconds": current_start_ms / 1000.0,
                    "end_seconds": current_end_ms / 1000.0,
                    "text": " ".join(current_texts),
                    "language_tag": current_language,
                })
            current_language = entry["language_tag"]
            current_texts = [entry["text"]]
            current_start_ms = entry["start_ms"]
            current_end_ms = entry["end_ms"]
        else:
            # Same language — extend turn
            current_texts.append(entry["text"])
            current_end_ms = max(current_end_ms, entry["end_ms"])

    # Save last turn
    if current_texts and current_language:
        turns.append({
            "start_seconds": current_start_ms / 1000.0,
            "end_seconds": current_end_ms / 1000.0,
            "text": " ".join(current_texts),
            "language_tag": current_language,
        })

    return turns


def _refine_turn_boundaries_and_close_gaps(
    turns: list[dict],
    primary_speech: list[tuple[float, float]],
    primary_language_tag: str,
    secondary_speech: list[tuple[float, float]],
    secondary_language_tag: str,
) -> list[dict]:
    """
    Refine turn start/end using silence-detected speech regions,
    then close gaps so each turn ends where the next begins.
    """
    refined = []

    for turn in turns:
        language_tag = turn["language_tag"]
        turn_start = turn["start_seconds"]
        turn_end = turn["end_seconds"]

        # Pick the right speech regions for this language
        if language_tag == primary_language_tag:
            speech_regions = primary_speech
        else:
            speech_regions = secondary_speech

        # Find first and last speech region that overlaps with this turn
        first_speech_start = None
        last_speech_end = None
        for region_start, region_end in speech_regions:
            if region_start < turn_end and region_end > turn_start:
                if first_speech_start is None:
                    first_speech_start = region_start
                last_speech_end = region_end

        # Snap to speech boundaries if found, otherwise keep Transkriptor times
        if first_speech_start is not None:
            refined_start = first_speech_start
            refined_end = last_speech_end
        else:
            refined_start = turn_start
            refined_end = turn_end

        refined.append({
            "start_seconds": refined_start,
            "end_seconds": refined_end,
            "text": turn["text"],
            "language_tag": language_tag,
        })

    # Close gaps: each turn ends where the next begins (seamless)
    for i in range(len(refined) - 1):
        gap_start = refined[i]["end_seconds"]
        gap_end = refined[i + 1]["start_seconds"]
        if gap_end > gap_start:
            midpoint = (gap_start + gap_end) / 2
            refined[i]["end_seconds"] = midpoint
            refined[i + 1]["start_seconds"] = midpoint

    return refined


def _fill_gaps_by_alternation(
    primary_speech: list[tuple[float, float]],
    secondary_speech: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Fill all gaps in the timeline using RU→LV→RU→LV alternation logic.

    Every moment must belong to either primary or secondary:
    - Gap between two primary regions → assign to secondary
    - Gap between two secondary regions → assign to primary
    - Gap between primary and secondary → split at midpoint
    - Gap at same boundary → split at midpoint

    Also resolves overlaps by splitting at midpoint.
    """
    # Tag all regions
    tagged = []
    for s, e in primary_speech:
        tagged.append((s, e, "P"))
    for s, e in secondary_speech:
        tagged.append((s, e, "S"))
    tagged.sort(key=lambda x: x[0])

    if not tagged:
        return list(primary_speech), list(secondary_speech)

    # First resolve overlaps between adjacent regions
    resolved = []
    for region in tagged:
        if not resolved:
            resolved.append(region)
            continue
        prev_s, prev_e, prev_label = resolved[-1]
        curr_s, curr_e, curr_label = region

        if curr_s < prev_e:
            # Overlap — split at midpoint
            midpoint = (curr_s + min(prev_e, curr_e)) / 2
            resolved[-1] = (prev_s, midpoint, prev_label)
            resolved.append((midpoint, curr_e, curr_label))
        else:
            resolved.append(region)

    # Now fill gaps between consecutive regions
    filled = [resolved[0]]
    for i in range(1, len(resolved)):
        prev_s, prev_e, prev_label = filled[-1]
        curr_s, curr_e, curr_label = resolved[i]

        gap = curr_s - prev_e
        if gap > 0.05:  # gap exists
            if prev_label == curr_label:
                # Same language on both sides → gap belongs to OTHER language
                gap_label = "S" if prev_label == "P" else "P"
                filled.append((prev_e, curr_s, gap_label))
            else:
                # Different languages → split gap at midpoint
                midpoint = (prev_e + curr_s) / 2
                filled[-1] = (prev_s, midpoint, prev_label)
                filled.append((midpoint, curr_e, curr_label))
                continue

        filled.append((curr_s, curr_e, curr_label))

    # Merge consecutive same-label regions
    merged = [filled[0]]
    for s, e, label in filled[1:]:
        if label == merged[-1][2] and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), label)
        else:
            merged.append((s, e, label))

    # Split back into primary and secondary
    new_primary = [(s, e) for s, e, l in merged if l == "P"]
    new_secondary = [(s, e) for s, e, l in merged if l == "S"]

    return new_primary, new_secondary


def _split_overlaps_at_midpoint(
    regions_a: list[tuple[float, float]],
    regions_b: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Resolve both overlaps AND gaps between two tracks' speech regions
    by splitting at the midpoint.

    - Overlaps: each track gives up half (split at midpoint)
    - Gaps: each track extends half (meet at midpoint)

    Result: the entire timeline is covered by either A or B with no
    overlaps and no gaps.
    """
    # Combine all regions with labels, sort by start time
    tagged = []
    for s, e in regions_a:
        tagged.append((s, e, "A"))
    for s, e in regions_b:
        tagged.append((s, e, "B"))
    tagged.sort(key=lambda x: x[0])

    if not tagged:
        return list(regions_a), list(regions_b)

    # Build resolved lists
    resolved_a = list(regions_a)
    resolved_b = list(regions_b)

    # Step 1: Resolve overlaps — split at midpoint
    changed = True
    while changed:
        changed = False
        for i in range(len(resolved_a)):
            for j in range(len(resolved_b)):
                a_start, a_end = resolved_a[i]
                b_start, b_end = resolved_b[j]

                overlap_start = max(a_start, b_start)
                overlap_end = min(a_end, b_end)

                if overlap_start < overlap_end - 0.05:
                    midpoint = (overlap_start + overlap_end) / 2
                    if a_start < b_start:
                        resolved_a[i] = (a_start, midpoint)
                        resolved_b[j] = (midpoint, b_end)
                    else:
                        resolved_b[j] = (b_start, midpoint)
                        resolved_a[i] = (midpoint, a_end)
                    changed = True
                    break
            if changed:
                break

    # Step 2: Close gaps — extend each side to midpoint of the gap
    all_regions = []
    for i, (s, e) in enumerate(resolved_a):
        all_regions.append((s, e, "A", i))
    for i, (s, e) in enumerate(resolved_b):
        all_regions.append((s, e, "B", i))
    all_regions.sort(key=lambda x: x[0])

    for idx in range(len(all_regions) - 1):
        curr_start, curr_end, curr_label, curr_i = all_regions[idx]
        next_start, next_end, next_label, next_i = all_regions[idx + 1]

        gap = next_start - curr_end
        if gap > 0.05:  # >50ms gap
            midpoint = (curr_end + next_start) / 2
            # Extend current region's end to midpoint
            if curr_label == "A":
                resolved_a[curr_i] = (resolved_a[curr_i][0], midpoint)
            else:
                resolved_b[curr_i] = (resolved_b[curr_i][0], midpoint)
            # Extend next region's start to midpoint
            if next_label == "A":
                resolved_a[next_i] = (midpoint, resolved_a[next_i][1])
            else:
                resolved_b[next_i] = (midpoint, resolved_b[next_i][1])

    # Remove regions that became too small
    resolved_a = [(s, e) for s, e in resolved_a if e - s >= 0.1]
    resolved_b = [(s, e) for s, e in resolved_b if e - s >= 0.1]

    return resolved_a, resolved_b


def _align_secondary_to_primary(
    primary_speech: list[tuple[float, float]],
    secondary_speech: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    RU (primary) has priority. Trim LV (secondary) speech on BOTH sides:

    1. If RU speech overlaps LV START → push LV start forward to where RU ends
       (pastor still talking when translator starts → wait for pastor)

    2. If RU speech overlaps LV END → pull LV end backward to where RU starts
       (pastor starts speaking before translator finishes → translator stops)

    If overlap consumes the entire secondary region, remove it.
    """
    aligned = []
    for sec_start, sec_end in secondary_speech:
        new_start = sec_start
        new_end = sec_end

        for pri_start, pri_end in primary_speech:
            if pri_start < new_end and pri_end > new_start:
                # Primary overlaps with start of secondary → push start forward
                if pri_end > new_start and pri_end < new_end:
                    new_start = max(new_start, pri_end)

                # Primary overlaps with end of secondary → pull end backward
                if pri_start > new_start and pri_start < new_end:
                    new_end = min(new_end, pri_start)

        if new_start < new_end and new_end - new_start >= 0.2:
            aligned.append((new_start, new_end))

    return aligned


def _remove_mic_bleed(
    speech_regions: list[tuple[float, float]],
    other_track_speech: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Remove speech regions that are mostly overlapping with the other track's
    speech. If the other speaker is talking and this track picks up audio,
    it's mic bleed — not real speech.

    A region is considered bleed if more than 50% of its duration overlaps
    with the other track's speech.
    """
    cleaned = []
    for region_start, region_end in speech_regions:
        region_duration = region_end - region_start

        # Calculate how much of this region overlaps with other track's speech
        overlap_with_other = 0.0
        for other_start, other_end in other_track_speech:
            overlap_start = max(region_start, other_start)
            overlap_end = min(region_end, other_end)
            if overlap_start < overlap_end:
                overlap_with_other += overlap_end - overlap_start

        overlap_ratio = overlap_with_other / region_duration if region_duration > 0 else 0

        if overlap_ratio > 0.5:
            # More than half overlaps with other speaker → mic bleed, remove
            continue
        else:
            cleaned.append((region_start, region_end))

    return cleaned


def _merge_pauses_where_other_track_is_silent(
    speech_regions: list[tuple[float, float]],
    other_track_speech: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Merge consecutive speech regions when the gap between them has NO speech
    in the other track. This means the speaker just paused (same train of
    thought), not a real speaker switch.

    If the gap DOES have other-track speech, keep the boundary (real switch).
    """
    if len(speech_regions) <= 1:
        return list(speech_regions)

    sorted_regions = sorted(speech_regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]

    for region_start, region_end in sorted_regions[1:]:
        previous_start, previous_end = merged[-1]
        gap_start = previous_end
        gap_end = region_start

        if gap_end <= gap_start:
            # Overlapping or adjacent — merge
            merged[-1] = (previous_start, max(previous_end, region_end))
            continue

        # Check if the other track has speech during this gap
        other_has_speech_in_gap = False
        for other_start, other_end in other_track_speech:
            overlap_start = max(gap_start, other_start)
            overlap_end = min(gap_end, other_end)
            if overlap_end - overlap_start > 0.3:  # at least 300ms of other speech
                other_has_speech_in_gap = True
                break

        if other_has_speech_in_gap:
            # Real speaker switch — keep the boundary
            merged.append((region_start, region_end))
        else:
            # Just a pause — merge across the gap
            merged[-1] = (previous_start, region_end)

    return merged


def _merge_consecutive_same_language(entries: list[dict]) -> list[dict]:
    """
    Merge consecutive entries with the same language into one.
    Combines text and extends time range.
    Guarantees strict RU → LV → RU → LV alternation.
    """
    if not entries:
        return []

    merged = [entries[0].copy()]

    for entry in entries[1:]:
        if entry["language_tag"] == merged[-1]["language_tag"]:
            merged[-1]["end_seconds"] = entry["end_seconds"]
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged


def _close_gaps_between_entries(entries: list[dict]) -> list[dict]:
    """
    Close gaps between entries so each ends where the next begins.
    Splits gap at midpoint between the two entries.
    """
    if len(entries) <= 1:
        return entries

    result = [e.copy() for e in entries]
    for i in range(len(result) - 1):
        gap_start = result[i]["end_seconds"]
        gap_end = result[i + 1]["start_seconds"]
        if gap_end > gap_start:
            midpoint = (gap_start + gap_end) / 2
            result[i]["end_seconds"] = midpoint
            result[i + 1]["start_seconds"] = midpoint

    return result


def _enforce_strict_alternation(turns: list[dict]) -> list[dict]:
    """
    Merge consecutive same-language turns to enforce strict RU→LV→RU→LV.
    If two adjacent turns have the same language, combine them into one
    (extending start to earliest, end to latest, keeping same language).
    """
    if not turns:
        return []

    merged = [turns[0].copy()]

    for turn in turns[1:]:
        if turn["language_tag"] == merged[-1]["language_tag"]:
            # Same language — extend the previous turn to cover both
            merged[-1]["end_seconds"] = turn["end_seconds"]
        else:
            # Different language — new turn, close the gap
            merged[-1]["end_seconds"] = turn["start_seconds"]
            merged.append(turn.copy())

    return merged


def _merge_nearby_regions(
    regions: list[tuple[float, float]],
    max_gap_seconds: float,
) -> list[tuple[float, float]]:
    """
    Merge speech regions that are close together (gap < max_gap_seconds).
    Small internal pauses (breathing, emphasis) get absorbed into one block.
    Only gaps larger than max_gap_seconds create separate blocks.
    """
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]

    for region_start, region_end in sorted_regions[1:]:
        previous_start, previous_end = merged[-1]
        gap = region_start - previous_end

        if gap <= max_gap_seconds:
            # Close enough — merge into one block
            merged[-1] = (previous_start, max(previous_end, region_end))
        else:
            merged.append((region_start, region_end))

    return merged


def _find_covering_speech_regions(
    entry_start: float,
    entry_end: float,
    speech_regions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Find all speech regions that overlap with a time range."""
    covering = []
    for region_start, region_end in speech_regions:
        if region_start < entry_end and region_end > entry_start:
            covering.append((region_start, region_end))
    return covering


def _find_overlaps(
    regions_a: list[tuple[float, float]],
    regions_b: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """Find time ranges where both A and B have speech."""
    overlaps = []
    for start_a, end_a in regions_a:
        for start_b, end_b in regions_b:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                overlaps.append((overlap_start, overlap_end, overlap_end - overlap_start))
    return overlaps


def _split_overlaps_in_middle(
    regions_a: list[tuple[float, float]],
    regions_b: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Resolve overlaps by splitting at the midpoint.
    The region that started first keeps up to the midpoint,
    the other starts from the midpoint.
    """
    resolved_a = list(regions_a)
    resolved_b = list(regions_b)

    for i, (start_a, end_a) in enumerate(resolved_a):
        for j, (start_b, end_b) in enumerate(resolved_b):
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                midpoint = (overlap_start + overlap_end) / 2
                if start_a < start_b:
                    resolved_a[i] = (start_a, midpoint)
                    resolved_b[j] = (midpoint, end_b)
                else:
                    resolved_b[j] = (start_b, midpoint)
                    resolved_a[i] = (midpoint, end_a)

    return resolved_a, resolved_b
