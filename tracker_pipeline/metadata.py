from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import ffprobe_json


@dataclass(slots=True)
class VideoMetadata:
    file_name: str
    video_stem: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    file_size_bytes: int
    source_bitrate_bps: int


def _safe_int(value: str | int | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _safe_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def probe_video(video_path: Path) -> VideoMetadata:
    probe = ffprobe_json(video_path)
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    width = _safe_int(video_stream.get("width"))
    height = _safe_int(video_stream.get("height"))
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    fps = _fraction_to_float(r_frame_rate)
    frame_count = _safe_int(video_stream.get("nb_frames"))
    duration_sec = _safe_float(video_stream.get("duration")) or _safe_float(format_info.get("duration"))
    file_size_bytes = video_path.stat().st_size

    bitrate_bps = _safe_int(format_info.get("bit_rate"))
    if bitrate_bps <= 0 and duration_sec > 0:
        bitrate_bps = int((file_size_bytes * 8) / duration_sec)

    return VideoMetadata(
        file_name=video_path.name,
        video_stem=video_path.stem,
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        file_size_bytes=file_size_bytes,
        source_bitrate_bps=bitrate_bps,
    )


def _fraction_to_float(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        den_value = float(den)
        return 0.0 if den_value == 0 else float(num) / den_value
    return float(value)
