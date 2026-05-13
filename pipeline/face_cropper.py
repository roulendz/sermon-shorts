"""
pipeline/face_cropper.py

Crops a landscape (16:9) video segment to portrait (9:16) with body pose tracking.
Uses dead-zone + velocity-clamped tracking for smooth, professional camera movement.

Tracking rules:
- Body torso (shoulder midpoint) inside middle third of crop: crop stays still
- Torso drifts outside: crop glides toward body at max N pixels/frame
- Result: zero jitter when speaker is still, butter-smooth panning when they move

Debug mode renders tracking overlay on full-frame video for tuning.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
POSE_MODEL_FILENAME = "pose_landmarker_heavy.task"

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

NOSE_INDEX = 0
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

LANDMARK_VISIBILITY_THRESHOLD = 0.5
FACING_SMOOTHING_FACTOR = 0.15
MOVEMENT_SMOOTHING_FACTOR = 0.2
VELOCITY_SCALE_RATIO = 0.008
MAX_DEAD_ZONE_SHIFT_RATIO = 1.0 / 3.0
MOVEMENT_WEIGHT = 0.8
FACING_WEIGHT = 0.2


class PortraitCropError(Exception):
    pass


class DeadZoneCropTracker:
    """
    Dead-zone + proportional-easing horizontal crop tracker.

    The crop window only moves when the target drifts outside the middle third
    (rule of thirds). Movement covers a fraction of the remaining distance each
    frame — fast when far, smooth when close. When target is lost, crop holds.
    """

    def __init__(
        self,
        crop_width: int,
        source_width: int,
        easing_factor: float = 0.3,
    ):
        self.crop_width = crop_width
        self.source_width = source_width
        self.easing_factor = easing_factor
        self.current_crop_center: float = source_width / 2.0
        self._initialized: bool = False
        self._smoothed_facing: float = 0.0
        self._smoothed_movement: float = 0.0
        self._previous_target_x: Optional[float] = None
        self._velocity_scale: float = crop_width * VELOCITY_SCALE_RATIO
        self.combined_lead_direction: float = 0.0

    def update(self, target_center_x: Optional[int], facing_direction: float = 0.0) -> int:
        """
        Feed detected body center X position and facing direction.
        Computes movement velocity from consecutive frames and combines with
        facing direction (70/30 weight) for lead room offset.
        Moving right → space on right. Facing left while still → space on left.
        """
        self._smoothed_facing += (facing_direction - self._smoothed_facing) * FACING_SMOOTHING_FACTOR

        if target_center_x is not None and self._previous_target_x is not None:
            velocity = target_center_x - self._previous_target_x
            movement_direction = max(-1.0, min(1.0, velocity / self._velocity_scale))
        else:
            movement_direction = 0.0

        if target_center_x is not None:
            self._previous_target_x = float(target_center_x)

        self._smoothed_movement += (movement_direction - self._smoothed_movement) * MOVEMENT_SMOOTHING_FACTOR

        self.combined_lead_direction = max(-1.0, min(1.0,
            MOVEMENT_WEIGHT * self._smoothed_movement + FACING_WEIGHT * self._smoothed_facing,
        ))

        if target_center_x is None:
            return self._clamp_crop_x_start()

        lead_room_offset = self.combined_lead_direction * self.crop_width * MAX_DEAD_ZONE_SHIFT_RATIO

        if not self._initialized:
            self.current_crop_center = float(target_center_x) + lead_room_offset
            self._initialized = True
            return self._clamp_crop_x_start()

        adjusted_target_x = target_center_x + lead_room_offset

        crop_left = self.current_crop_center - self.crop_width / 2
        position_in_crop = adjusted_target_x - crop_left

        dead_zone_left = self.crop_width / 3
        dead_zone_right = self.crop_width * 2 / 3

        if dead_zone_left <= position_in_crop <= dead_zone_right:
            return self._clamp_crop_x_start()

        if position_in_crop < dead_zone_left:
            target_center = adjusted_target_x - dead_zone_left + self.crop_width / 2
        else:
            target_center = adjusted_target_x - dead_zone_right + self.crop_width / 2

        delta = target_center - self.current_crop_center
        self.current_crop_center += delta * self.easing_factor

        return self._clamp_crop_x_start()

    def _clamp_crop_x_start(self) -> int:
        crop_x_start = int(self.current_crop_center - self.crop_width / 2)
        return max(0, min(crop_x_start, self.source_width - self.crop_width))


def ensure_pose_detection_model(model_directory: Path) -> Path:
    """Download the pose landmarker model if not already cached."""
    model_path = model_directory / POSE_MODEL_FILENAME
    if not model_path.exists():
        model_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading pose detection model to {model_path}")
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path


def create_pose_landmarker(model_path: Path):
    """Create a MediaPipe pose landmarker in VIDEO running mode."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _detect_pose_landmarks(
    landmarker,
    frame_rgb: np.ndarray,
    timestamp_milliseconds: int,
):
    """Detect pose landmarks for the first person. Returns landmarks list or None."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_milliseconds)

    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def _shoulder_center_x(landmarks, frame_width: int) -> Optional[int]:
    """Extract shoulder midpoint X in pixels. Falls back to single visible shoulder."""
    left_shoulder = landmarks[LEFT_SHOULDER_INDEX]
    right_shoulder = landmarks[RIGHT_SHOULDER_INDEX]

    left_visible = left_shoulder.visibility > LANDMARK_VISIBILITY_THRESHOLD
    right_visible = right_shoulder.visibility > LANDMARK_VISIBILITY_THRESHOLD

    if left_visible and right_visible:
        return int((left_shoulder.x + right_shoulder.x) / 2 * frame_width)
    if left_visible:
        return int(left_shoulder.x * frame_width)
    if right_visible:
        return int(right_shoulder.x * frame_width)
    return None


def _compute_facing_direction(landmarks) -> float:
    """
    Compute facing direction from nose position relative to shoulder midpoint.
    Returns -1.0 (facing left) to +1.0 (facing right), normalized by shoulder width.
    Returns 0.0 if landmarks not visible enough.
    """
    nose = landmarks[NOSE_INDEX]
    left_shoulder = landmarks[LEFT_SHOULDER_INDEX]
    right_shoulder = landmarks[RIGHT_SHOULDER_INDEX]

    if nose.visibility < LANDMARK_VISIBILITY_THRESHOLD:
        return 0.0
    if left_shoulder.visibility < LANDMARK_VISIBILITY_THRESHOLD:
        return 0.0
    if right_shoulder.visibility < LANDMARK_VISIBILITY_THRESHOLD:
        return 0.0

    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    if shoulder_width < 0.01:
        return 0.0

    facing_ratio = (nose.x - shoulder_midpoint_x) / shoulder_width
    return max(-1.0, min(1.0, facing_ratio))


def crop_segment_to_portrait(
    source_video_path: Path,
    output_file_path: Path,
    start_seconds: float,
    duration_seconds: float,
    model_directory: Path,
    easing_factor: float = 0.3,
    speed_multiplier: float = 1.0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Crop a landscape video segment to 9:16 portrait with pose tracking.
    Uses dead-zone tracking: crop only moves when body leaves the middle third,
    and movement is velocity-clamped for smooth panning.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = ensure_pose_detection_model(model_directory)
    landmarker = create_pose_landmarker(model_path)

    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise PortraitCropError(f"Cannot open video: {source_video_path}")

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame_index = int(start_seconds * frame_rate)
    total_frames = int(duration_seconds * frame_rate)

    crop_width = int(source_height * OUTPUT_WIDTH / OUTPUT_HEIGHT)
    crop_height = source_height

    tracker = DeadZoneCropTracker(
        crop_width=crop_width,
        source_width=source_width,
        easing_factor=easing_factor,
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

    ffmpeg_arguments = _build_portrait_encoding_command(
        source_video_path=source_video_path,
        output_file_path=output_file_path,
        crop_width=crop_width,
        crop_height=crop_height,
        frame_rate=frame_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        speed_multiplier=speed_multiplier,
    )

    logger.info(
        "Portrait crop: %s → %s (start=%.1fs, dur=%.1fs, speed=%.1fx)",
        source_video_path.name, output_file_path.name,
        start_seconds, duration_seconds, speed_multiplier,
    )
    logger.debug("FFmpeg command: %s", " ".join(ffmpeg_arguments))

    ffmpeg_process = subprocess.Popen(
        ffmpeg_arguments,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_chunks: list[bytes] = []

    def _drain_stderr():
        while True:
            chunk = ffmpeg_process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        for frame_index in range(total_frames):
            success, frame = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((start_frame_index + frame_index) / frame_rate * 1000)

            landmarks = _detect_pose_landmarks(landmarker, frame_rgb, timestamp_ms)
            body_center_x = _shoulder_center_x(landmarks, source_width) if landmarks else None
            facing_direction = _compute_facing_direction(landmarks) if landmarks else 0.0

            crop_x_start = tracker.update(body_center_x, facing_direction)

            cropped_frame = frame[:, crop_x_start : crop_x_start + crop_width]

            if not cropped_frame.flags["C_CONTIGUOUS"]:
                cropped_frame = np.ascontiguousarray(cropped_frame)

            ffmpeg_process.stdin.write(cropped_frame.tobytes())

            if on_progress and frame_index % 30 == 0:
                on_progress((frame_index / total_frames) * 100)

    finally:
        capture.release()
        landmarker.close()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        stderr_thread.join(timeout=5)

    if ffmpeg_process.returncode != 0:
        stderr_output = b"".join(stderr_chunks).decode(errors="replace")
        raise PortraitCropError(
            f"FFmpeg portrait encoding failed (exit code {ffmpeg_process.returncode}):\n"
            f"{stderr_output[-2000:]}"
        )

    if on_progress:
        on_progress(100.0)

    logger.info(f"Portrait crop saved: {output_file_path}")
    return output_file_path


def generate_debug_video(
    source_video_path: Path,
    output_file_path: Path,
    start_seconds: float,
    duration_seconds: float,
    model_directory: Path,
    easing_factor: float = 0.3,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Render debug overlay on full-frame video showing tracking data.
    Draws: shoulder/hip landmarks (green), torso connections, crop rectangle (red),
    rule-of-thirds lines (yellow), shoulder center dot (cyan).
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = ensure_pose_detection_model(model_directory)
    landmarker = create_pose_landmarker(model_path)

    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise PortraitCropError(f"Cannot open video: {source_video_path}")

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame_index = int(start_seconds * frame_rate)
    total_frames = int(duration_seconds * frame_rate)

    crop_width = int(source_height * OUTPUT_WIDTH / OUTPUT_HEIGHT)

    tracker = DeadZoneCropTracker(
        crop_width=crop_width,
        source_width=source_width,
        easing_factor=easing_factor,
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

    ffmpeg_arguments = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{source_width}x{source_height}",
        "-framerate", str(frame_rate),
        "-i", "pipe:0",
        "-ss", f"{start_seconds:.3f}",
        "-t", f"{duration_seconds:.3f}",
        "-i", str(source_video_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-shortest",
        str(output_file_path),
    ]

    logger.info(
        "Debug video: %s → %s (start=%.1fs, dur=%.1fs)",
        source_video_path.name, output_file_path.name,
        start_seconds, duration_seconds,
    )
    logger.debug("FFmpeg command: %s", " ".join(ffmpeg_arguments))

    ffmpeg_process = subprocess.Popen(
        ffmpeg_arguments,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_chunks: list[bytes] = []

    def _drain_stderr():
        while True:
            chunk = ffmpeg_process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)
    color_cyan = (255, 255, 0)
    color_white = (255, 255, 255)

    try:
        for frame_index in range(total_frames):
            success, frame = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((start_frame_index + frame_index) / frame_rate * 1000)

            landmarks = _detect_pose_landmarks(landmarker, frame_rgb, timestamp_ms)
            body_center_x = None
            facing_direction = 0.0

            if landmarks is not None:
                body_center_x = _shoulder_center_x(landmarks, source_width)
                facing_direction = _compute_facing_direction(landmarks)
                _draw_pose_overlay(
                    frame, landmarks, source_width, source_height,
                    color_green, color_cyan,
                )

            crop_x_start = tracker.update(body_center_x, facing_direction)
            crop_x_end = crop_x_start + crop_width

            cv2.rectangle(
                frame,
                (crop_x_start, 0),
                (crop_x_end, source_height),
                color_red, 3,
            )

            dead_zone_left = crop_x_start + crop_width // 3
            dead_zone_right = crop_x_start + crop_width * 2 // 3
            cv2.line(frame, (dead_zone_left, 0), (dead_zone_left, source_height), color_yellow, 1)
            cv2.line(frame, (dead_zone_right, 0), (dead_zone_right, source_height), color_yellow, 1)

            dim_left = frame[:, :crop_x_start]
            dim_right = frame[:, crop_x_end:]
            if dim_left.size > 0:
                frame[:, :crop_x_start] = (dim_left * 0.4).astype(np.uint8)
            if dim_right.size > 0:
                frame[:, crop_x_end:] = (dim_right * 0.4).astype(np.uint8)

            body_label = f"body_x={body_center_x}" if body_center_x else "NO POSE"
            crop_label = f"crop=[{crop_x_start}..{crop_x_end}] center={int(tracker.current_crop_center)}"
            lead_direction = tracker.combined_lead_direction
            lead_label = (
                f"lead={lead_direction:+.2f} "
                f"(move={tracker._smoothed_movement:+.2f} face={tracker._smoothed_facing:+.2f}) "
                f"{'<< LEFT' if lead_direction < -0.1 else 'RIGHT >>' if lead_direction > 0.1 else 'CENTER'}"
            )
            cv2.putText(frame, body_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_white, 2)
            cv2.putText(frame, crop_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_white, 2)
            cv2.putText(frame, lead_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_yellow, 2)

            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            ffmpeg_process.stdin.write(frame.tobytes())

            if on_progress and frame_index % 30 == 0:
                on_progress((frame_index / total_frames) * 100)

    finally:
        capture.release()
        landmarker.close()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        stderr_thread.join(timeout=5)

    if ffmpeg_process.returncode != 0:
        stderr_output = b"".join(stderr_chunks).decode(errors="replace")
        raise PortraitCropError(
            f"FFmpeg debug video failed (exit code {ffmpeg_process.returncode}):\n"
            f"{stderr_output[-2000:]}"
        )

    if on_progress:
        on_progress(100.0)

    return output_file_path


def _draw_pose_overlay(
    frame: np.ndarray,
    landmarks,
    frame_width: int,
    frame_height: int,
    landmark_color: tuple,
    center_color: tuple,
) -> None:
    """Draw shoulder/hip landmarks, torso connections, and shoulder center on frame."""
    key_indices = [LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, LEFT_HIP_INDEX, RIGHT_HIP_INDEX]
    points: dict[int, tuple[int, int]] = {}

    for index in key_indices:
        landmark = landmarks[index]
        if landmark.visibility > LANDMARK_VISIBILITY_THRESHOLD:
            pixel_x = int(landmark.x * frame_width)
            pixel_y = int(landmark.y * frame_height)
            points[index] = (pixel_x, pixel_y)
            cv2.circle(frame, (pixel_x, pixel_y), 6, landmark_color, -1)

    if LEFT_SHOULDER_INDEX in points and RIGHT_SHOULDER_INDEX in points:
        cv2.line(frame, points[LEFT_SHOULDER_INDEX], points[RIGHT_SHOULDER_INDEX], landmark_color, 2)
        center_x = (points[LEFT_SHOULDER_INDEX][0] + points[RIGHT_SHOULDER_INDEX][0]) // 2
        center_y = (points[LEFT_SHOULDER_INDEX][1] + points[RIGHT_SHOULDER_INDEX][1]) // 2
        cv2.circle(frame, (center_x, center_y), 8, center_color, -1)

    if LEFT_HIP_INDEX in points and RIGHT_HIP_INDEX in points:
        cv2.line(frame, points[LEFT_HIP_INDEX], points[RIGHT_HIP_INDEX], landmark_color, 2)

    if LEFT_SHOULDER_INDEX in points and LEFT_HIP_INDEX in points:
        cv2.line(frame, points[LEFT_SHOULDER_INDEX], points[LEFT_HIP_INDEX], landmark_color, 2)

    if RIGHT_SHOULDER_INDEX in points and RIGHT_HIP_INDEX in points:
        cv2.line(frame, points[RIGHT_SHOULDER_INDEX], points[RIGHT_HIP_INDEX], landmark_color, 2)


def _build_portrait_encoding_command(
    source_video_path: Path,
    output_file_path: Path,
    crop_width: int,
    crop_height: int,
    frame_rate: float,
    start_seconds: float,
    duration_seconds: float,
    speed_multiplier: float = 1.0,
) -> list[str]:
    """Build FFmpeg command for portrait encoding with audio from original."""
    input_framerate = frame_rate * speed_multiplier

    audio_filter_arguments: list[str] = []
    if speed_multiplier != 1.0:
        audio_filter_arguments = ["-af", f"atempo={speed_multiplier}"]

    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{crop_width}x{crop_height}",
        "-framerate", str(input_framerate),
        "-i", "pipe:0",
        "-ss", f"{start_seconds:.3f}",
        "-t", f"{duration_seconds:.3f}",
        "-i", str(source_video_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",
        "-r", str(frame_rate),
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        *audio_filter_arguments,
        "-movflags", "+faststart",
        "-shortest",
        str(output_file_path),
    ]


def build_portrait_output_path(landscape_video_path: Path) -> Path:
    """Generate the portrait output path from a landscape segment path."""
    stem = landscape_video_path.stem
    return landscape_video_path.parent / f"{stem} [portrait].mp4"


def build_debug_output_path(landscape_video_path: Path) -> Path:
    """Generate the debug overlay output path from a landscape segment path."""
    stem = landscape_video_path.stem
    return landscape_video_path.parent / f"{stem} [debug].mp4"
