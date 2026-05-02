from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig


def annotation_csv_path(config: PipelineConfig) -> Path:
    return config.annotations_dir / "first_frame_boxes.csv"


def videos_metadata_csv_path(config: PipelineConfig) -> Path:
    return config.metadata_dir / "videos.csv"


def compressions_metadata_csv_path(config: PipelineConfig) -> Path:
    return config.metadata_dir / "compressions.csv"


def per_frame_track_csv_path(
    config: PipelineConfig,
    video_stem: str,
    variant_name: str,
    tracker_name: str,
) -> Path:
    return config.track_frames_dir / f"{video_stem}__{variant_name}__{tracker_name}.csv"


def annotated_video_path(
    config: PipelineConfig,
    video_stem: str,
    variant_name: str,
    tracker_name: str,
) -> Path:
    return config.annotated_videos_dir / f"{video_stem}__{variant_name}__{tracker_name}.mp4"


def tracking_runs_csv_path(config: PipelineConfig) -> Path:
    return config.tracks_dir / "tracking_runs.csv"


def per_run_summary_csv_path(config: PipelineConfig) -> Path:
    return config.metrics_dir / "per_run_summary.csv"


def per_video_tracker_summary_csv_path(config: PipelineConfig) -> Path:
    return config.metrics_dir / "per_video_tracker_summary.csv"


def per_tracker_bitrate_summary_csv_path(config: PipelineConfig) -> Path:
    return config.metrics_dir / "per_tracker_bitrate_summary.csv"


def per_frame_metrics_csv_path(config: PipelineConfig) -> Path:
    return config.metrics_dir / "per_frame_metrics.csv"


def compression_output_path(config: PipelineConfig, video_stem: str, ratio_label: str) -> Path:
    return config.compressed_dir / video_stem / f"{video_stem}__{ratio_label}.mp4"


def ffmpeg_passlog_prefix(
    config: PipelineConfig,
    video_stem: str,
    ratio_label: str,
) -> Path:
    return config.ffmpeg_passlog_dir / f"{video_stem}__{ratio_label}"
