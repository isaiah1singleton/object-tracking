from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import (
    build_tracking_jobs,
    collect_video_metadata,
    discover_videos,
    ensure_annotations,
    initialize_directories,
    run_full_pipeline,
)
from .compression import compress_all
from .metrics import evaluate_runs
from .paths import compressions_metadata_csv_path, tracking_runs_csv_path
from .plots import generate_plots
from .tracking import run_tracking_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compression robustness pipeline for OpenCV single-object trackers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="Draw first-frame bounding boxes.")
    annotate.add_argument("--video", help="Only annotate one video file name.")
    annotate.add_argument("--force", action="store_true", help="Overwrite existing annotations.")

    compress = subparsers.add_parser("compress", help="Compress videos using the relative bitrate ladder.")
    compress.add_argument("--video", help="Only process one video file name.")
    compress.add_argument("--force", action="store_true", help="Re-encode even if outputs already exist.")

    track = subparsers.add_parser("track", help="Run trackers on original and compressed videos.")
    track.add_argument("--video", help="Only process one video file name.")
    track.add_argument("--force", action="store_true", help="Re-run tracking even if CSV outputs already exist.")
    track.add_argument("--no-render", action="store_true", help="Skip annotated tracking video rendering.")

    evaluate = subparsers.add_parser("evaluate", help="Compute metrics and generate plots from tracking outputs.")
    evaluate.add_argument("--video", help="Only process one video file name.")

    run_all = subparsers.add_parser("run-all", help="Run the full pipeline.")
    run_all.add_argument("--video", help="Only process one video file name.")
    run_all.add_argument("--force-annotations", action="store_true")
    run_all.add_argument("--force-compression", action="store_true")
    run_all.add_argument("--force-tracking", action="store_true")
    run_all.add_argument("--no-render", action="store_true", help="Skip annotated tracking video rendering.")

    subparsers.add_parser("show-config", help="Print key folders and settings.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    initialize_directories(config)

    if args.command == "show-config":
        print_config(config)
        return

    if args.command == "run-all":
        run_full_pipeline(
            config,
            file_name=args.video,
            force_annotations=args.force_annotations,
            force_compression=args.force_compression,
            force_tracking=args.force_tracking,
            render_tracking_videos=not args.no_render,
        )
        return

    video_paths = discover_videos(config, file_name=getattr(args, "video", None))
    video_metadata = collect_video_metadata(config, video_paths)

    if args.command == "annotate":
        ensure_annotations(config, video_paths, force=args.force)
        return

    if args.command == "compress":
        compress_all(config, video_metadata, force=args.force)
        return

    annotations_df = ensure_annotations(config, video_paths, force=False)

    tracking_jobs = build_tracking_jobs(
        config,
        video_metadata,
        annotations_df,
        _load_required_compression_df(config),
        single_video=getattr(args, "video", None),
    )

    if args.command == "track":
        run_tracking_jobs(
            config,
            tracking_jobs,
            render=not getattr(args, "no_render", False),
            force=getattr(args, "force", False),
        )
        return

    compression_df = _load_required_compression_df(config)
    tracking_index_df = _load_required_tracking_index_df(config)
    _, per_run_df, _, per_tracker_bitrate_df = evaluate_runs(config, tracking_index_df, compression_df)
    generate_plots(config, per_run_df, per_tracker_bitrate_df)


def print_config(config) -> None:
    print(f"Root: {config.root_dir}")
    print(f"Original videos: {config.originals_dir}")
    print(f"Annotations: {config.annotations_dir}")
    print(f"Compressed videos: {config.compressed_dir}")
    print(f"Tracks: {config.track_frames_dir}")
    print(f"Metrics: {config.metrics_dir}")
    print(f"Plots: {config.plots_dir}")
    print(f"Relative bitrate ladder: {config.bitrate_ratios}")
    print(f"Trackers: {config.trackers}")


def _load_required_compression_df(config) -> pd.DataFrame:
    path = compressions_metadata_csv_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing compression metadata at {path}. Run `python3 -m tracker_pipeline.cli compress` first."
        )
    return pd.read_csv(path)


def _load_required_tracking_index_df(config) -> pd.DataFrame:
    path = tracking_runs_csv_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tracking index at {path}. Run `python3 -m tracker_pipeline.cli track` first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
