"""
pipeline/segment_selector.py

Builds the Manus AI prompt from SRT subtitle content,
and parses the JSON response into a list of VideoSegment objects.

This is the bridge between api/manus_client.py (HTTP only)
and models/video_segment.py (data only).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from typing import Sequence

import srt

from models.video_segment import VideoSegment

logger = logging.getLogger(__name__)

MINIMUM_CLIPS = 5
MAXIMUM_CLIPS = 14

SEGMENT_SELECTION_PROMPT_TEMPLATE = """\
You are a world-class video content strategist and editor specializing in \
TikTok, YouTube Shorts, and Instagram Reels for church and ministry content. \
You identify transcript moments that are emotionally powerful, self-contained, \
and highly shareable. You are precise and your output is always valid JSON.

Read the ENTIRE SRT transcript below and identify the {minimum_clips} to \
{maximum_clips} most viral-worthy segments.

STRICT RULES:
1. Each clip must be between 50 and 90 seconds long
2. Return timestamps in ABSOLUTE SECONDS as a float (e.g. 95.350), \
derived from the SRT timestamps in the file
3. Start 0.5-1.0 seconds before a strong hook word
4. End 0.5-1.0 seconds after a strong payoff word for natural pacing
5. Each clip must be a complete, self-contained thought — never cut mid-sentence
6. Segments must NOT overlap
7. Do not select generic intros, outros, or announcements
8. The transcript is bilingual — [RU] lines are Russian (pastor), [LV] lines \
are Latvian (translator). Both languages are present in the video.

GRAMMAR CORRECTION — for each selected clip:
- Review both [RU] and [LV] text within the clip's time range
- Fix any transcription grammar errors, misspellings, or awkward phrasing
- Provide corrected text for both languages in the output
- The transcript was auto-generated, so expect minor errors

SCORING — rate each segment 0-10:
- hook_power: Does the opening grab attention and stop scrolling immediately?
- emotional_impact: Does it evoke awe, humor, inspiration, or shock?
- discussion_potential: Will it spark debate or strong comments?
- shareability: Is it quotable or deeply relatable to a wide audience?

overall_viral_score = (hook_power * 0.4) + (emotional_impact * 0.3) \
+ (discussion_potential * 0.2) + (shareability * 0.1)

Order results from highest to lowest overall_viral_score.

Respond with ONLY valid JSON — no markdown fences, no explanation, nothing else:

{{
  "potential_clips": [
    {{
      "index": 1,
      "start_time": <float>,
      "end_time": <float>,
      "suggested_title": "<punchy viral-worthy title in Latvian, max 70 characters use /humanizer skill>",
      "suggested_title_ru": "<punchy viral-worthy title in Russian, max 70 characters use /humanizer skill>",
      "suggested_title_en": "<punchy viral-worthy title in English, max 70 characters use /humanizer skill>",
      "viral_hook_text_overlay": "<max 10 words for opening text in Latvian overlay use /humanizer skill>",
      "viral_hook_text_overlay_ru": "<max 10 words for opening text in Russian overlay use /humanizer skill>",
      "viral_hook_text_overlay_en": "<max 10 words for opening text in English overlay use /humanizer skill>",
      "scoring_analysis": {{
        "hook_power_score": <int 0-10>,
        "emotional_impact_score": <int 0-10>,
        "discussion_potential_score": <int 0-10>,
        "shareability_score": <int 0-10>,
        "overall_viral_score": <float>
      }},
      "reason_for_selection": "<one sentence expert analysis>",
      "corrected_text_ru": "<grammar-corrected Russian text for this clip>",
      "corrected_text_lv": "<grammar-corrected Latvian text for this clip>"
    }}
  ]
}}

SRT TRANSCRIPT:
{srt_content}
"""


def build_segment_selection_prompt(srt_content: str) -> str:
    return SEGMENT_SELECTION_PROMPT_TEMPLATE.format(
        minimum_clips=MINIMUM_CLIPS,
        maximum_clips=MAXIMUM_CLIPS,
        srt_content=srt_content,
    )


def parse_segments_from_manus_response(
    manus_response_text: str,
    all_subtitles: list[srt.Subtitle],
) -> list[VideoSegment]:
    """
    Parse the JSON response from Manus into VideoSegment objects.
    Timestamps in the response are absolute seconds (float).
    Transcript text is extracted from the full subtitle list by time window.
    """
    logger.info(
        "Parsing Manus response (%d characters) with %d subtitles",
        len(manus_response_text),
        len(all_subtitles),
    )
    response_json = extract_json_from_response_text(manus_response_text)
    raw_clips = response_json.get("potential_clips", [])

    if not raw_clips:
        raise ValueError(f"Manus response contained no segments: {response_json}")

    video_segments = []
    for raw_clip in raw_clips:
        start_time = convert_seconds_float_to_timedelta(raw_clip["start_time"])
        end_time = convert_seconds_float_to_timedelta(raw_clip["end_time"])

        # Use Manus corrected text if available, otherwise extract from subtitles
        corrected_ru = raw_clip.get("corrected_text_ru", "")
        corrected_lv = raw_clip.get("corrected_text_lv", "")
        if corrected_ru or corrected_lv:
            transcript_text = f"{corrected_ru}\n{corrected_lv}".strip()
        else:
            transcript_text = extract_transcript_text_for_window(
                all_subtitles, start_time, end_time
            )
        if not transcript_text:
            transcript_text = raw_clip.get("reason_for_selection", f"Segment {raw_clip['index']}")

        scoring = raw_clip.get("scoring_analysis", {})

        video_segments.append(
            VideoSegment(
                index=raw_clip["index"],
                start_time=start_time,
                end_time=end_time,
                transcript_text=transcript_text,
                selection_reason=raw_clip.get("reason_for_selection", ""),
                hook_power_score=scoring.get("hook_power_score", 0),
                emotional_impact_score=scoring.get("emotional_impact_score", 0),
                discussion_potential_score=scoring.get("discussion_potential_score", 0),
                shareability_score=scoring.get("shareability_score", 0),
                overall_viral_score=scoring.get("overall_viral_score", 0.0),
                suggested_title=raw_clip.get("suggested_title", f"Segment {raw_clip['index']}"),
                viral_hook_text_overlay=raw_clip.get("viral_hook_text_overlay", ""),
            )
        )

    logger.info(f"Parsed {len(video_segments)} segments from Manus response")
    return video_segments


def extract_json_from_response_text(response_text: str) -> dict:
    """
    Extract a JSON object from response text that may contain
    surrounding markdown fences, extra text, or conversation history.
    Finds the first valid JSON object containing 'potential_clips'.
    """
    cleaned_text = response_text.strip()

    # Remove markdown fences
    cleaned_text = re.sub(r"```(?:json)?\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```", "", cleaned_text)

    # Try direct parse first
    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict) and "potential_clips" in parsed:
            logger.info("Parsed JSON directly from response text")
            return parsed
        logger.warning(
            "Direct JSON parse succeeded but missing 'potential_clips' key. "
            "Top-level keys: %s",
            list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
    except json.JSONDecodeError:
        pass

    # Find JSON object by scanning for opening brace containing potential_clips
    candidates_checked = 0
    for match in re.finditer(r'\{', cleaned_text):
        start = match.start()
        # Try progressively larger substrings from this brace
        depth = 0
        for i in range(start, len(cleaned_text)):
            if cleaned_text[i] == '{':
                depth += 1
            elif cleaned_text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned_text[start:i + 1]
                    candidates_checked += 1
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "potential_clips" in parsed:
                            logger.info(
                                "Found JSON with 'potential_clips' via brace scan "
                                "(checked %d candidates)",
                                candidates_checked,
                            )
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    logger.error(
        "Could not find JSON with 'potential_clips' in Manus response. "
        "Checked %d brace-delimited candidates. Response length: %d chars",
        candidates_checked,
        len(response_text),
    )
    raise ValueError(
        f"Could not find JSON with 'potential_clips' in Manus response.\n"
        f"Response length: {len(response_text)} chars. "
        f"Checked {candidates_checked} JSON candidates.\n"
        f"Response (first 500 chars): {response_text[:500]}\n"
        f"Response (last 500 chars): {response_text[-500:]}"
    )


def convert_seconds_float_to_timedelta(seconds: float) -> timedelta:
    """
    Convert an absolute seconds float (e.g. 95.350) into a timedelta.
    This is the timestamp format returned by the Manus AI prompt.
    """
    return timedelta(seconds=seconds)


def extract_transcript_text_for_window(
    subtitles: Sequence[srt.Subtitle],
    window_start: timedelta,
    window_end: timedelta,
) -> str:
    subtitles_in_window = [
        subtitle for subtitle in subtitles
        if subtitle.start >= window_start and subtitle.end <= window_end
    ]
    return " ".join(subtitle.content.strip() for subtitle in subtitles_in_window)
