from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io_utils import ensure_dir
from .paths import annotated_video_path, per_frame_track_csv_path, tracking_runs_csv_path


TRACKER_LABELS = {
    "csrt": "TrackerCSRT",
    "kcf": "TrackerKCF",
}


def _build_tracker(tracker_name: str):
    if tracker_name == "csrt":
        return cv2.TrackerCSRT.create()
    if tracker_name == "kcf":
        return cv2.TrackerKCF.create()
    raise ValueError(f"Unsupported tracker: {tracker_name}")


def run_tracking_jobs(
    config: PipelineConfig,
    jobs: list[dict[str, Any]],
    *,
    render: bool = True,
    force: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    ensure_dir(config.track_frames_dir)
    ensure_dir(config.annotated_videos_dir)

    prepared_jobs = []
    for job in jobs:
        prepared = dict(job)
        prepared["render"] = render
        prepared["force"] = force
        prepared_jobs.append(prepared)

    rows: list[dict] = []
    total = len(prepared_jobs)
    print(f"[track] Starting {total} tracking jobs.")
    with ProcessPoolExecutor(max_workers=workers or config.tracking_workers) as executor:
        futures = [executor.submit(_track_one, config.root_dir, job) for job in prepared_jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            print(
                "[track] "
                f"{idx}/{total} finished: "
                f"{result['video_stem']} {result['variant_name']} {result['tracker_name']} "
                f"frames={result['frame_count']} status={result['status']}"
            )

    df = pd.DataFrame(rows).sort_values(["video_stem", "variant_name", "tracker_name"])
    df.to_csv(tracking_runs_csv_path(config), index=False)
    print(f"[track] Wrote {tracking_runs_csv_path(config)}")
    return df


def _track_one(root_dir: Path, job: dict[str, Any]) -> dict:
    from .config import load_config

    config = load_config(root_dir)
    tracker_name = job["tracker_name"]
    video_stem = job["video_stem"]
    variant_name = job["variant_name"]
    video_path = Path(job["video_path"])
    bbox = _normalize_bbox(job["bbox"])

    csv_path = per_frame_track_csv_path(config, video_stem, variant_name, tracker_name)
    video_out_path = annotated_video_path(config, video_stem, variant_name, tracker_name)

    if csv_path.exists() and (not job["render"] or video_out_path.exists()):
        if not job["force"]:
            df_existing = pd.read_csv(csv_path)
            frame_count = len(df_existing)
            return {
                "video_stem": video_stem,
                "variant_name": variant_name,
                "tracker_name": tracker_name,
                "video_path": str(video_path),
                "track_csv_path": str(csv_path),
                "annotated_video_path": str(video_out_path) if job["render"] else "",
                "frame_count": frame_count,
                "status": "reused",
            }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video {video_path}. "
            "The file is likely corrupted or incomplete. Re-run compression with "
            f"`python3 -m tracker_pipeline.cli compress --force --video {video_path.stem.split('__')[0]}.mp4` "
            "or force the full compression stage again."
        )

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            f"Failed to read first frame from {video_path}. "
            "The file is likely corrupted or incomplete. Re-run compression for this source video."
        )

    tracker = _build_tracker(tracker_name)
    tracker.init(first_frame, bbox)

    writer = None
    if job["render"]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*config.mp4_fourcc)
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    rows = []
    frame_index = 0
    current_bbox = bbox

    while True:
        if frame_index == 0:
            frame = first_frame
            update_ok = True
        else:
            ok, frame = cap.read()
            if not ok:
                break
            update_ok, current_bbox = tracker.update(frame)

        x, y, w, h = current_bbox if update_ok else (np.nan, np.nan, np.nan, np.nan)
        rows.append(
            {
                "video_stem": video_stem,
                "variant_name": variant_name,
                "tracker_name": tracker_name,
                "frame_index": frame_index,
                "update_ok": bool(update_ok),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            }
        )

        if writer is not None:
            drawn = frame.copy()
            _draw_box(drawn, bbox, (0, 255, 0), "init")
            if update_ok:
                _draw_box(drawn, current_bbox, (0, 0, 255), TRACKER_LABELS[tracker_name])
            cv2.putText(
                drawn,
                f"{video_stem} | {variant_name} | {tracker_name} | frame {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(drawn)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return {
        "video_stem": video_stem,
        "variant_name": variant_name,
        "tracker_name": tracker_name,
        "video_path": str(video_path),
        "track_csv_path": str(csv_path),
        "annotated_video_path": str(video_out_path) if job["render"] else "",
        "frame_count": len(df),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "status": "tracked",
    }


def _draw_box(frame: np.ndarray, bbox: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _normalize_bbox(raw_bbox: Any) -> tuple[float, float, float, float]:
    if len(raw_bbox) != 4:
        raise ValueError(f"Bounding box must have 4 values, got {raw_bbox}")

    values: list[int] = []
    for value in raw_bbox:
        if isinstance(value, np.generic):
            value = value.item()
        values.append(int(round(float(value))))

    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise ValueError(f"Bounding box must have positive width and height, got {raw_bbox}")

    return (x, y, w, h)
