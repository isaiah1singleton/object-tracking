from __future__ import annotations

import os
from pathlib import Path

_mpl_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import PipelineConfig
from .io_utils import ensure_dir


def generate_plots(
    config: PipelineConfig,
    per_run_df: pd.DataFrame,
    per_tracker_bitrate_df: pd.DataFrame,
) -> None:
    ensure_dir(config.plots_dir)
    print("[plots] Generating plots.")
    _plot_metric_by_ratio(config, per_tracker_bitrate_df, "mean_iou", "Mean IoU to Baseline", "mean_iou_vs_ratio.png")
    _plot_metric_by_ratio(
        config,
        per_tracker_bitrate_df,
        "success_rate_iou_050",
        "Success Rate at IoU >= 0.50",
        "success_050_vs_ratio.png",
    )
    _plot_metric_by_ratio(
        config,
        per_tracker_bitrate_df,
        "mean_normalized_center_error",
        "Mean Normalized Center Error",
        "mean_nce_vs_ratio.png",
    )
    _plot_metric_by_ratio(
        config,
        per_tracker_bitrate_df.assign(failure_rate=1.0 - per_tracker_bitrate_df["update_success_rate"]),
        "failure_rate",
        "Failure Rate (1 - Update Success Rate)",
        "failure_rate_vs_ratio.png",
    )
    _plot_video_grid(config, per_run_df, "mean_iou", "Per-Video Mean IoU", "per_video_mean_iou.png")
    _plot_video_grid(
        config,
        per_run_df,
        "success_rate_iou_050",
        "Per-Video Success Rate at IoU >= 0.50",
        "per_video_success_050.png",
    )
    _plot_boxplot_by_tracker(
        config,
        per_run_df,
        "mean_iou",
        "Distribution of Mean IoU by Tracker",
        "mean_iou_boxplot_by_tracker.png",
    )
    _plot_boxplot_by_tracker(
        config,
        per_run_df,
        "mean_normalized_center_error",
        "Distribution of Mean Normalized Center Error by Tracker",
        "mean_nce_boxplot_by_tracker.png",
    )
    _plot_tracker_ranking_bar(config, per_run_df)
    _plot_degradation_slope_bar(config, per_run_df)
    _plot_accuracy_robustness_scatter(config, per_run_df)
    _plot_video_bitrate_heatmap(config, per_run_df, "mean_iou", "Mean IoU Heatmap", "mean_iou_heatmap.png")
    print(f"[plots] Wrote plots to {config.plots_dir}")


def _plot_metric_by_ratio(
    config: PipelineConfig,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_name: str,
) -> None:
    plt.figure(figsize=(9, 6))
    for tracker_name, tracker_df in df.groupby("tracker_name"):
        tracker_df = tracker_df.sort_values("ratio", ascending=False)
        plt.plot(tracker_df["ratio"], tracker_df[metric], marker="o", label=tracker_name.upper())
    plt.xlabel("Relative Bitrate Ratio")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.plots_dir / output_name, dpi=180)
    plt.close()


def _plot_video_grid(
    config: PipelineConfig,
    per_run_df: pd.DataFrame,
    metric: str,
    title: str,
    output_name: str,
) -> None:
    videos = sorted(per_run_df["video_stem"].unique().tolist())
    cols = 2
    rows = max(1, (len(videos) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)

    for ax, video_stem in zip(axes.flatten(), videos):
        video_df = per_run_df[per_run_df["video_stem"] == video_stem]
        for tracker_name, tracker_df in video_df.groupby("tracker_name"):
            tracker_df = tracker_df.sort_values("ratio", ascending=False)
            ax.plot(tracker_df["ratio"], tracker_df[metric], marker="o", label=tracker_name.upper())
        ax.set_title(video_stem)
        ax.set_xlabel("Relative Bitrate Ratio")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flatten()[len(videos):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(config.plots_dir / output_name, dpi=180)
    plt.close(fig)


def _plot_boxplot_by_tracker(
    config: PipelineConfig,
    per_run_df: pd.DataFrame,
    metric: str,
    title: str,
    output_name: str,
) -> None:
    plt.figure(figsize=(8, 6))
    order = sorted(per_run_df["tracker_name"].unique().tolist())
    data = [per_run_df.loc[per_run_df["tracker_name"] == tracker, metric].dropna().tolist() for tracker in order]
    plt.boxplot(data, tick_labels=[name.upper() for name in order])
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.plots_dir / output_name, dpi=180)
    plt.close()


def _plot_tracker_ranking_bar(config: PipelineConfig, per_run_df: pd.DataFrame) -> None:
    ranking_df = (
        per_run_df.groupby("tracker_name", as_index=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            success_auc=("success_auc", "mean"),
            success_rate_iou_050=("success_rate_iou_050", "mean"),
            mean_normalized_center_error=("mean_normalized_center_error", "mean"),
            update_success_rate=("update_success_rate", "mean"),
        )
        .sort_values("mean_iou", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].bar(ranking_df["tracker_name"].str.upper(), ranking_df["mean_iou"])
    axes[0].set_title("Mean IoU")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(ranking_df["tracker_name"].str.upper(), ranking_df["success_rate_iou_050"])
    axes[1].set_title("Success @ IoU >= 0.50")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(ranking_df["tracker_name"].str.upper(), ranking_df["update_success_rate"])
    axes[2].set_title("Update Success Rate")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Tracker Ranking Summary")
    fig.tight_layout()
    fig.savefig(config.plots_dir / "tracker_ranking_summary.png", dpi=180)
    plt.close(fig)


def _plot_degradation_slope_bar(config: PipelineConfig, per_run_df: pd.DataFrame) -> None:
    rows = []
    for (video_stem, tracker_name), group in per_run_df.groupby(["video_stem", "tracker_name"]):
        group = group.sort_values("ratio")
        if len(group) < 2:
            continue
        slope = _safe_linear_slope(group["ratio"].to_numpy(), group["mean_iou"].to_numpy())
        rows.append({"video_stem": video_stem, "tracker_name": tracker_name, "mean_iou_slope": slope})

    slope_df = pd.DataFrame(rows)
    if slope_df.empty:
        return

    summary = slope_df.groupby("tracker_name", as_index=False)["mean_iou_slope"].mean()
    plt.figure(figsize=(8, 6))
    plt.bar(summary["tracker_name"].str.upper(), summary["mean_iou_slope"])
    plt.title("Average Degradation Slope Across Videos")
    plt.ylabel("Slope of Mean IoU vs Relative Bitrate Ratio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.plots_dir / "degradation_slope_bar.png", dpi=180)
    plt.close()


def _plot_accuracy_robustness_scatter(config: PipelineConfig, per_run_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    for tracker_name, tracker_df in per_run_df.groupby("tracker_name"):
        plt.scatter(
            tracker_df["mean_iou"],
            1.0 - tracker_df["update_success_rate"],
            label=tracker_name.upper(),
            alpha=0.8,
        )
    plt.xlabel("Mean IoU to Baseline")
    plt.ylabel("Failure Rate (1 - Update Success Rate)")
    plt.title("Accuracy vs Robustness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.plots_dir / "accuracy_vs_robustness_scatter.png", dpi=180)
    plt.close()


def _plot_video_bitrate_heatmap(
    config: PipelineConfig,
    per_run_df: pd.DataFrame,
    metric: str,
    title: str,
    output_name: str,
) -> None:
    trackers = sorted(per_run_df["tracker_name"].unique().tolist())
    fig, axes = plt.subplots(1, len(trackers), figsize=(6 * max(1, len(trackers)), 7), squeeze=False)

    for ax, tracker_name in zip(axes.flatten(), trackers):
        tracker_df = per_run_df[per_run_df["tracker_name"] == tracker_name].copy()
        pivot = tracker_df.pivot(index="video_stem", columns="ratio", values=metric)
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_title(tracker_name.upper())
        ax.set_xlabel("Relative Bitrate Ratio")
        ax.set_ylabel("Video")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(config.plots_dir / output_name, dpi=180)
    plt.close(fig)


def _safe_linear_slope(x, y) -> float:
    if len(x) < 2:
        return float("nan")
    coeffs = __import__("numpy").polyfit(x, y, 1)
    return float(coeffs[0])
