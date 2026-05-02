from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .io_utils import ensure_dir, run_command
from .metadata import VideoMetadata, probe_video
from .paths import compression_output_path, compressions_metadata_csv_path, ffmpeg_passlog_prefix


def ratio_to_label(ratio: float) -> str:
    return f"r{str(ratio).replace('.', 'p')}"


def compress_all(
    config: PipelineConfig,
    video_metadata: list[VideoMetadata],
    *,
    force: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    ensure_dir(config.compressed_dir)
    ensure_dir(config.metadata_dir)
    ensure_dir(config.ffmpeg_passlog_dir)

    jobs: list[dict] = []
    for meta in video_metadata:
        for ratio in config.bitrate_ratios:
            ratio_label = ratio_to_label(ratio)
            target_bitrate_bps = max(100_000, int(meta.source_bitrate_bps * ratio))
            output_path = compression_output_path(config, meta.video_stem, ratio_label)
            jobs.append(
                {
                    "video_path": str(meta.path),
                    "video_stem": meta.video_stem,
                    "ratio": ratio,
                    "ratio_label": ratio_label,
                    "target_bitrate_bps": target_bitrate_bps,
                    "output_path": str(output_path),
                    "preset": config.ffmpeg_preset,
                    "audio_bitrate": config.audio_bitrate,
                    "passlog_prefix": str(ffmpeg_passlog_prefix(config, meta.video_stem, ratio_label)),
                    "force": force,
                }
            )

    rows: list[dict] = []
    total = len(jobs)
    print(f"[compress] Starting {total} compression jobs across {len(video_metadata)} videos.")
    with ProcessPoolExecutor(max_workers=workers or config.compression_workers) as executor:
        futures = [executor.submit(_compress_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            print(
                "[compress] "
                f"{idx}/{total} finished: "
                f"{result['video_stem']} {result['variant_name']} "
                f"target={result['target_bitrate_bps']} actual={result['actual_bitrate_bps']} "
                f"bpppf={result['bits_per_pixel_per_frame']:.6f} status={result['status']}"
            )

    df = pd.DataFrame(rows).sort_values(["video_stem", "ratio"], ascending=[True, False])
    df.to_csv(compressions_metadata_csv_path(config), index=False)
    print(f"[compress] Wrote {compressions_metadata_csv_path(config)}")
    return df


def _compress_one(job: dict) -> dict:
    video_path = Path(job["video_path"])
    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not job["force"] and _is_valid_video_file(output_path):
        compressed_meta = probe_video(output_path)
        return {
            "video_stem": job["video_stem"],
            "source_file_name": video_path.name,
            "variant_name": job["ratio_label"],
            "ratio": job["ratio"],
            "target_bitrate_bps": job["target_bitrate_bps"],
            "output_file_name": output_path.name,
            "output_path": str(output_path),
            "actual_bitrate_bps": compressed_meta.source_bitrate_bps,
            "width": compressed_meta.width,
            "height": compressed_meta.height,
            "fps": compressed_meta.fps,
            "duration_sec": compressed_meta.duration_sec,
            "bits_per_pixel_per_frame": _bits_per_pixel_per_frame(
                compressed_meta.source_bitrate_bps,
                compressed_meta.width,
                compressed_meta.height,
                compressed_meta.fps,
            ),
            "status": "reused",
        }

    bitrate = str(job["target_bitrate_bps"])
    passlog_prefix = job["passlog_prefix"]

    cmd_pass1 = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        job["preset"],
        "-b:v",
        bitrate,
        "-pass",
        "1",
        "-passlogfile",
        passlog_prefix,
        "-an",
        "-f",
        "null",
        "/dev/null",
    ]
    run_command(cmd_pass1)

    cmd_pass2 = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        job["preset"],
        "-b:v",
        bitrate,
        "-pass",
        "2",
        "-passlogfile",
        passlog_prefix,
        "-c:a",
        "aac",
        "-b:a",
        job["audio_bitrate"],
        str(output_path),
    ]
    run_command(cmd_pass2)

    compressed_meta = probe_video(output_path)
    return {
        "video_stem": job["video_stem"],
        "source_file_name": video_path.name,
        "variant_name": job["ratio_label"],
        "ratio": job["ratio"],
        "target_bitrate_bps": job["target_bitrate_bps"],
        "output_file_name": output_path.name,
        "output_path": str(output_path),
        "actual_bitrate_bps": compressed_meta.source_bitrate_bps,
        "width": compressed_meta.width,
        "height": compressed_meta.height,
        "fps": compressed_meta.fps,
        "duration_sec": compressed_meta.duration_sec,
        "bits_per_pixel_per_frame": _bits_per_pixel_per_frame(
            compressed_meta.source_bitrate_bps,
            compressed_meta.width,
            compressed_meta.height,
            compressed_meta.fps,
        ),
        "status": "encoded",
    }


def _bits_per_pixel_per_frame(
    bitrate_bps: int,
    width: int,
    height: int,
    fps: float,
) -> float:
    denom = max(float(width) * float(height) * max(fps, 1e-9), 1e-9)
    return float(bitrate_bps / denom)


def _is_valid_video_file(path: Path) -> bool:
    try:
        meta = probe_video(path)
    except Exception:
        return False
    return meta.width > 0 and meta.height > 0 and meta.duration_sec > 0
