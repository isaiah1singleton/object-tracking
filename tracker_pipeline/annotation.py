from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .config import PipelineConfig
from .io_utils import ensure_dir
from .paths import annotation_csv_path


ANNOTATION_COLUMNS = ["file_name", "video_stem", "x", "y", "w", "h"]


def load_annotations(config: PipelineConfig) -> pd.DataFrame:
    path = annotation_csv_path(config)
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    return pd.read_csv(path)


def annotate_videos(
    config: PipelineConfig,
    video_paths: list[Path],
    *,
    force: bool = False,
) -> pd.DataFrame:
    ensure_dir(config.annotations_dir)
    existing = load_annotations(config)
    existing_by_name = {row["file_name"]: row for _, row in existing.iterrows()}
    output_by_name: dict[str, dict] = {
        row["file_name"]: row.to_dict() for _, row in existing.iterrows()
    }
    total = len(video_paths)
    print(f"[annotate] Preparing annotations for {total} videos.")

    for idx, video_path in enumerate(video_paths, start=1):
        if not force and video_path.name in existing_by_name:
            print(f"[annotate] {idx}/{total} reused: {video_path.name}")
            continue

        print(f"[annotate] {idx}/{total} drawing ROI: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {video_path}")

        title = f"Select ROI - {video_path.name}"
        roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        x, y, w, h = [int(v) for v in roi]
        if w <= 0 or h <= 0:
            raise RuntimeError(f"No valid ROI selected for {video_path.name}")

        output_by_name[video_path.name] = {
            "file_name": video_path.name,
            "video_stem": video_path.stem,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

    df = pd.DataFrame(output_by_name.values(), columns=ANNOTATION_COLUMNS).sort_values("file_name")
    df.to_csv(annotation_csv_path(config), index=False)
    print(f"[annotate] Wrote {annotation_csv_path(config)}")
    return df
