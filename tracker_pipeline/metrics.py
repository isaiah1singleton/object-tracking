from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import (
    per_frame_metrics_csv_path,
    per_run_summary_csv_path,
    per_tracker_bitrate_summary_csv_path,
    per_video_tracker_summary_csv_path,
)


def evaluate_runs(
    config: PipelineConfig,
    tracking_index_df: pd.DataFrame,
    compression_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_lookup = {
        (row["video_stem"], row["tracker_name"]): row
        for _, row in tracking_index_df[tracking_index_df["variant_name"] == "original"].iterrows()
    }
    compression_lookup = {
        (row["video_stem"], row["variant_name"]): row for _, row in compression_df.iterrows()
    }

    per_frame_rows: list[dict] = []
    per_run_rows: list[dict] = []

    comparison_df = tracking_index_df[tracking_index_df["variant_name"] != "original"].copy()
    total = len(comparison_df)
    print(f"[evaluate] Evaluating {total} compressed tracking runs.")
    for idx, (_, row) in enumerate(comparison_df.iterrows(), start=1):
        baseline_row = baseline_lookup[(row["video_stem"], row["tracker_name"])]
        compressed_track = pd.read_csv(row["track_csv_path"])
        baseline_track = pd.read_csv(baseline_row["track_csv_path"])

        merged = compressed_track.merge(
            baseline_track,
            on=["video_stem", "tracker_name", "frame_index"],
            suffixes=("_comp", "_base"),
            how="inner",
        )

        metrics_df = _compute_frame_metrics(
            merged,
            low_iou_failure_threshold=config.low_iou_failure_threshold,
        )
        metrics_df["variant_name"] = row["variant_name"]
        metrics_df["ratio"] = compression_lookup[(row["video_stem"], row["variant_name"])]["ratio"]
        per_frame_rows.extend(metrics_df.to_dict(orient="records"))

        run_summary = _summarize_run(
            metrics_df,
            iou_thresholds=config.iou_thresholds,
            low_iou_failure_threshold=config.low_iou_failure_threshold,
            low_iou_failure_streak=config.low_iou_failure_streak,
            fps=float(compression_lookup[(row["video_stem"], row["variant_name"])]["fps"]),
        )
        run_summary.update(
            {
                "video_stem": row["video_stem"],
                "tracker_name": row["tracker_name"],
                "variant_name": row["variant_name"],
                "ratio": compression_lookup[(row["video_stem"], row["variant_name"])]["ratio"],
                "target_bitrate_bps": compression_lookup[(row["video_stem"], row["variant_name"])]["target_bitrate_bps"],
                "actual_bitrate_bps": compression_lookup[(row["video_stem"], row["variant_name"])]["actual_bitrate_bps"],
            }
        )
        per_run_rows.append(run_summary)
        print(
            "[evaluate] "
            f"{idx}/{total} finished: "
            f"{row['video_stem']} {row['variant_name']} {row['tracker_name']} "
            f"mean_iou={run_summary['mean_iou']:.4f} "
            f"success050={run_summary['success_rate_iou_050']:.4f}"
        )

    per_frame_df = pd.DataFrame(per_frame_rows).sort_values(
        ["video_stem", "tracker_name", "ratio", "frame_index"],
        ascending=[True, True, False, True],
    )
    per_run_df = pd.DataFrame(per_run_rows).sort_values(
        ["video_stem", "tracker_name", "ratio"],
        ascending=[True, True, False],
    )

    per_video_tracker_df = (
        per_run_df.groupby(["video_stem", "tracker_name"], as_index=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            success_auc=("success_auc", "mean"),
            success_rate_iou_050=("success_rate_iou_050", "mean"),
            mean_normalized_center_error=("mean_normalized_center_error", "mean"),
            update_success_rate=("update_success_rate", "mean"),
            mean_time_to_first_failure_sec=("time_to_first_failure_sec", "mean"),
        )
        .sort_values(["video_stem", "tracker_name"])
    )

    per_tracker_bitrate_df = (
        per_run_df.groupby(["tracker_name", "ratio"], as_index=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            success_auc=("success_auc", "mean"),
            success_rate_iou_050=("success_rate_iou_050", "mean"),
            mean_normalized_center_error=("mean_normalized_center_error", "mean"),
            update_success_rate=("update_success_rate", "mean"),
        )
        .sort_values(["tracker_name", "ratio"], ascending=[True, False])
    )

    per_frame_df.to_csv(per_frame_metrics_csv_path(config), index=False)
    per_run_df.to_csv(per_run_summary_csv_path(config), index=False)
    per_video_tracker_df.to_csv(per_video_tracker_summary_csv_path(config), index=False)
    per_tracker_bitrate_df.to_csv(per_tracker_bitrate_summary_csv_path(config), index=False)
    print(f"[evaluate] Wrote {per_frame_metrics_csv_path(config)}")
    print(f"[evaluate] Wrote {per_run_summary_csv_path(config)}")
    print(f"[evaluate] Wrote {per_video_tracker_summary_csv_path(config)}")
    print(f"[evaluate] Wrote {per_tracker_bitrate_summary_csv_path(config)}")

    return per_frame_df, per_run_df, per_video_tracker_df, per_tracker_bitrate_df


def _compute_frame_metrics(
    merged: pd.DataFrame,
    *,
    low_iou_failure_threshold: float,
) -> pd.DataFrame:
    base_boxes = merged[["x_base", "y_base", "w_base", "h_base"]].to_numpy(dtype=float)
    comp_boxes = merged[["x_comp", "y_comp", "w_comp", "h_comp"]].to_numpy(dtype=float)
    ious = np.array([_iou(a, b) for a, b in zip(base_boxes, comp_boxes)])
    normalized_center_errors = np.array(
        [_normalized_center_error(a, b) for a, b in zip(base_boxes, comp_boxes)]
    )

    out = merged[
        ["video_stem", "tracker_name", "frame_index", "update_ok_comp", "update_ok_base"]
    ].copy()
    out["iou"] = ious
    out["normalized_center_error"] = normalized_center_errors
    out["low_iou"] = out["iou"] < low_iou_failure_threshold
    return out


def _summarize_run(
    metrics_df: pd.DataFrame,
    *,
    iou_thresholds: list[float],
    low_iou_failure_threshold: float,
    low_iou_failure_streak: int,
    fps: float,
) -> dict:
    iou = metrics_df["iou"].to_numpy(dtype=float)
    nce = metrics_df["normalized_center_error"].to_numpy(dtype=float)
    update_ok = metrics_df["update_ok_comp"].astype(bool).to_numpy()
    success_rates = {threshold: float(np.mean(iou >= threshold)) for threshold in iou_thresholds}

    low_iou_mask = iou < low_iou_failure_threshold
    longest_low_iou_streak = _longest_streak(low_iou_mask)
    first_failure_frame = _first_streak_start(low_iou_mask, low_iou_failure_streak)

    summary = {
        "frame_count": int(len(metrics_df)),
        "mean_iou": float(np.nanmean(iou)),
        "median_iou": float(np.nanmedian(iou)),
        "success_auc": float(np.mean(list(success_rates.values()))),
        "mean_normalized_center_error": float(np.nanmean(nce)),
        "median_normalized_center_error": float(np.nanmedian(nce)),
        "update_success_rate": float(np.mean(update_ok)),
        "longest_low_iou_streak_frames": int(longest_low_iou_streak),
        "first_failure_frame": int(first_failure_frame) if first_failure_frame is not None else -1,
        "time_to_first_failure_sec": float(first_failure_frame / max(fps, 1e-6))
        if first_failure_frame is not None
        else np.nan,
    }

    for threshold in [0.25, 0.50, 0.75]:
        summary[f"success_rate_iou_{int(threshold * 100):03d}"] = success_rates[threshold]

    return summary


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    if np.any(np.isnan(box_a)) or np.any(np.isnan(box_b)):
        return 0.0

    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def _normalized_center_error(box_a: np.ndarray, box_b: np.ndarray) -> float:
    if np.any(np.isnan(box_a)) or np.any(np.isnan(box_b)):
        return 1.0

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_center = np.array([ax + aw / 2.0, ay + ah / 2.0], dtype=float)
    b_center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=float)
    distance = float(np.linalg.norm(a_center - b_center))
    norm = max(1.0, np.hypot(aw, ah))
    return distance / norm


def _longest_streak(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _first_streak_start(mask: np.ndarray, required_length: int) -> int | None:
    current = 0
    start = 0
    for idx, value in enumerate(mask):
        if value:
            if current == 0:
                start = idx
            current += 1
            if current >= required_length:
                return start
        else:
            current = 0
    return None
