from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(slots=True)
class PipelineConfig:
    root_dir: Path
    originals_dir: Path
    annotations_dir: Path
    compressed_dir: Path
    metadata_dir: Path
    tracks_dir: Path
    track_frames_dir: Path
    annotated_videos_dir: Path
    metrics_dir: Path
    plots_dir: Path
    logs_dir: Path
    cache_dir: Path
    ffmpeg_passlog_dir: Path
    bitrate_ratios: list[float] = field(
        default_factory=lambda: [0.30, 0.15, 0.08, 0.04, 0.02]
    )
    trackers: list[str] = field(default_factory=lambda: ["csrt", "kcf"])
    iou_thresholds: list[float] = field(
        default_factory=lambda: [round(x, 2) for x in [i / 20 for i in range(1, 20)]]
    )
    low_iou_failure_threshold: float = 0.10
    low_iou_failure_streak: int = 5
    mp4_fourcc: str = "mp4v"
    ffmpeg_preset: str = "medium"
    audio_bitrate: str = "128k"
    compression_workers: int = max(1, (os.cpu_count() or 4) // 2)
    tracking_workers: int = max(1, (os.cpu_count() or 4) // 2)


def load_config(root_dir: Path | None = None) -> PipelineConfig:
    root = root_dir or Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    results_dir = root / "results"
    cache_dir = root / ".cache"
    return PipelineConfig(
        root_dir=root,
        originals_dir=data_dir / "videos" / "originals",
        annotations_dir=data_dir / "annotations",
        compressed_dir=data_dir / "videos" / "compressed",
        metadata_dir=results_dir / "metadata",
        tracks_dir=results_dir / "tracks",
        track_frames_dir=results_dir / "tracks" / "per_frame",
        annotated_videos_dir=results_dir / "tracks" / "annotated_videos",
        metrics_dir=results_dir / "metrics",
        plots_dir=results_dir / "plots",
        logs_dir=results_dir / "logs",
        cache_dir=cache_dir,
        ffmpeg_passlog_dir=cache_dir / "ffmpeg_passlogs",
    )
