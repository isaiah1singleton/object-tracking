from __future__ import annotations

from pathlib import Path

import pandas as pd

from .annotation import annotate_videos, load_annotations
from .compression import compress_all
from .config import PipelineConfig
from .io_utils import ensure_dirs
from .metadata import VideoMetadata, probe_video
from .metrics import evaluate_runs
from .paths import videos_metadata_csv_path
from .plots import generate_plots
from .tracking import run_tracking_jobs


def initialize_directories(config: PipelineConfig) -> None:
    ensure_dirs(
        [
            config.originals_dir,
            config.annotations_dir,
            config.compressed_dir,
            config.metadata_dir,
            config.tracks_dir,
            config.track_frames_dir,
            config.annotated_videos_dir,
            config.metrics_dir,
            config.plots_dir,
            config.logs_dir,
            config.cache_dir,
            config.ffmpeg_passlog_dir,
        ]
    )


def discover_videos(config: PipelineConfig, *, file_name: str | None = None) -> list[Path]:
    video_paths = sorted(config.originals_dir.glob("*.mp4"))
    if file_name:
        video_paths = [path for path in video_paths if path.name == file_name]
    if not video_paths:
        raise FileNotFoundError(
            f"No .mp4 videos found in {config.originals_dir}. Place originals there before running."
        )
    return video_paths


def collect_video_metadata(config: PipelineConfig, video_paths: list[Path]) -> list[VideoMetadata]:
    metadata = [probe_video(path) for path in video_paths]
    df = pd.DataFrame(
        [
            {
                "file_name": meta.file_name,
                "video_stem": meta.video_stem,
                "path": str(meta.path),
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_sec": meta.duration_sec,
                "file_size_bytes": meta.file_size_bytes,
                "source_bitrate_bps": meta.source_bitrate_bps,
            }
            for meta in metadata
        ]
    ).sort_values("file_name")
    df.to_csv(videos_metadata_csv_path(config), index=False)
    return metadata


def ensure_annotations(config: PipelineConfig, video_paths: list[Path], *, force: bool = False) -> pd.DataFrame:
    current = load_annotations(config)
    existing_names = set(current["file_name"].tolist())
    needed = [path for path in video_paths if force or path.name not in existing_names]
    if needed:
        return annotate_videos(config, video_paths, force=force)
    return current.sort_values("file_name")


def build_tracking_jobs(
    config: PipelineConfig,
    video_metadata: list[VideoMetadata],
    annotations_df: pd.DataFrame,
    compression_df: pd.DataFrame,
    *,
    trackers: list[str] | None = None,
    single_video: str | None = None,
) -> list[dict]:
    trackers = trackers or config.trackers
    annotations_lookup = {
        row["file_name"]: (
            float(row["x"]),
            float(row["y"]),
            float(row["w"]),
            float(row["h"]),
        )
        for _, row in annotations_df.iterrows()
    }
    compression_rows = compression_df.to_dict(orient="records")
    compression_lookup_by_video: dict[str, list[dict]] = {}
    for row in compression_rows:
        compression_lookup_by_video.setdefault(row["video_stem"], []).append(row)

    jobs: list[dict] = []
    for meta in video_metadata:
        if single_video and meta.file_name != single_video:
            continue
        bbox = annotations_lookup.get(meta.file_name)
        if bbox is None:
            raise ValueError(f"Missing annotation for {meta.file_name}")

        for tracker_name in trackers:
            jobs.append(
                {
                    "video_stem": meta.video_stem,
                    "variant_name": "original",
                    "tracker_name": tracker_name,
                    "video_path": str(meta.path),
                    "bbox": bbox,
                }
            )
            for compression_row in compression_lookup_by_video.get(meta.video_stem, []):
                jobs.append(
                    {
                        "video_stem": meta.video_stem,
                        "variant_name": compression_row["variant_name"],
                        "tracker_name": tracker_name,
                        "video_path": compression_row["output_path"],
                        "bbox": bbox,
                    }
                )
    return jobs


def run_full_pipeline(
    config: PipelineConfig,
    *,
    file_name: str | None = None,
    force_annotations: bool = False,
    force_compression: bool = False,
    force_tracking: bool = False,
    render_tracking_videos: bool = True,
) -> dict:
    initialize_directories(config)
    print("[pipeline] Discovering videos.")
    video_paths = discover_videos(config, file_name=file_name)
    print(f"[pipeline] Found {len(video_paths)} video(s).")
    print("[pipeline] Collecting source metadata.")
    video_metadata = collect_video_metadata(config, video_paths)
    print("[pipeline] Ensuring annotations.")
    annotations_df = ensure_annotations(config, video_paths, force=force_annotations)
    print("[pipeline] Compressing videos.")
    compression_df = compress_all(config, video_metadata, force=force_compression)
    print("[pipeline] Building tracking jobs.")
    tracking_jobs = build_tracking_jobs(config, video_metadata, annotations_df, compression_df, single_video=file_name)
    print(f"[pipeline] Built {len(tracking_jobs)} tracking job(s).")
    print("[pipeline] Running trackers.")
    tracking_index_df = run_tracking_jobs(
        config,
        tracking_jobs,
        render=render_tracking_videos,
        force=force_tracking,
    )
    print("[pipeline] Computing metrics.")
    per_frame_df, per_run_df, per_video_tracker_df, per_tracker_bitrate_df = evaluate_runs(
        config,
        tracking_index_df,
        compression_df,
    )
    print("[pipeline] Generating plots.")
    generate_plots(config, per_run_df, per_tracker_bitrate_df)
    print("[pipeline] Complete.")
    return {
        "video_metadata": video_metadata,
        "annotations_df": annotations_df,
        "compression_df": compression_df,
        "tracking_index_df": tracking_index_df,
        "per_frame_df": per_frame_df,
        "per_run_df": per_run_df,
        "per_video_tracker_df": per_video_tracker_df,
        "per_tracker_bitrate_df": per_tracker_bitrate_df,
    }
