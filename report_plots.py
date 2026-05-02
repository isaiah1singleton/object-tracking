from __future__ import annotations

import os
from pathlib import Path

_mpl_cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 17,
    }
)


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "results" / "metrics"
OUTPUT_DIR = ROOT / "results" / "report_plots"
GUIDE_PATH = OUTPUT_DIR / "plot_guide.md"
VIDEO_DISPLAY_NAMES = {
    "baseline": "open_field",
    "runner_dog": "runner",
    "zipline": "cable_car",
}
MEAN_IOU_CMAP = LinearSegmentedColormap.from_list(
    "mean_iou_red_tan_green",
    ["#b2182b", "#e8d7b7", "#1b7837"],
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(METRICS_DIR / "per_run_summary.csv")
    per_video = pd.read_csv(METRICS_DIR / "per_video_tracker_summary.csv")
    per_tracker_ratio = pd.read_csv(METRICS_DIR / "per_tracker_bitrate_summary.csv")
    per_frame = pd.read_csv(METRICS_DIR / "per_frame_metrics.csv")

    print("[report-plots] Generating report-ready figures.")
    plot_mean_iou_vs_ratio(per_run, per_tracker_ratio)
    plot_mean_iou_grouped_bars(per_tracker_ratio)
    plot_mean_nce_grouped_bars(per_tracker_ratio)
    plot_update_success_grouped_bars(per_tracker_ratio)
    plot_success050_vs_ratio(per_tracker_ratio)
    plot_success_auc_curve(per_frame)
    plot_mean_iou_boxplot(per_run)
    plot_robustness_vs_accuracy(per_run)
    plot_per_video_mean_iou(per_run)
    plot_per_video_success050(per_run)
    plot_mean_iou_heatmap(per_run)
    plot_success050_heatmap(per_run)
    plot_tracker_summary_bars(per_video)
    plot_stability_comparison(per_run)
    write_guide()
    print(f"[report-plots] Wrote figures and guide to {OUTPUT_DIR}")


def plot_mean_iou_vs_ratio(per_run: pd.DataFrame, df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5.5))
    colors = {"csrt": "tab:blue", "kcf": "tab:orange"}
    all_mean_iou_values = []
    for tracker_name, tracker_df in df.groupby("tracker_name"):
        tracker_df = tracker_df.sort_values("ratio", ascending=False)
        all_mean_iou_values.extend(tracker_df["mean_iou"].tolist())
        spread_df = (
            per_run[per_run["tracker_name"] == tracker_name]
            .groupby("ratio")
            .agg(
                mean_iou=("mean_iou", "mean"),
                std_iou=("mean_iou", "std"),
                n=("mean_iou", "count"),
            )
            .reset_index()
            .sort_values("ratio", ascending=False)
        )
        spread_df["sem"] = spread_df["std_iou"].fillna(0.0) / np.sqrt(spread_df["n"].clip(lower=1))
        spread_df["ci95"] = 1.96 * spread_df["sem"]
        lower = np.clip(spread_df["mean_iou"] - spread_df["ci95"], 1e-4, 1.0)
        upper = np.clip(spread_df["mean_iou"] + spread_df["ci95"], 1e-4, 1.0)

        plt.fill_between(
            spread_df["ratio"],
            lower,
            upper,
            alpha=0.18,
            color=colors.get(tracker_name),
        )
        plt.plot(
            tracker_df["ratio"],
            tracker_df["mean_iou"],
            marker="o",
            linewidth=2,
            color=colors.get(tracker_name),
            label=tracker_name.upper(),
        )
    plt.xlabel("Relative Bitrate Ratio")
    plt.ylabel("Mean IoU to Baseline")
    plt.title("Mean IoU vs Compression Level")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_mean_iou_vs_ratio.png", dpi=220)
    plt.close()


def plot_mean_iou_grouped_bars(df: pd.DataFrame) -> None:
    pivot = (
        df.pivot(index="ratio", columns="tracker_name", values="mean_iou")
        .sort_index(ascending=False)
    )
    ratios = [f"{ratio:.2f}" for ratio in pivot.index.tolist()]
    x = np.arange(len(ratios), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    csrt_vals = pivot["csrt"].to_numpy() if "csrt" in pivot.columns else np.zeros(len(pivot))
    kcf_vals = pivot["kcf"].to_numpy() if "kcf" in pivot.columns else np.zeros(len(pivot))

    bars1 = ax.bar(x - width / 2, csrt_vals, width, label="CSRT")
    bars2 = ax.bar(x + width / 2, kcf_vals, width, label="KCF")

    ax.set_xlabel("Compression Level (Relative Bitrate Ratio)")
    ax.set_ylabel("Mean IoU to Baseline")
    ax.set_title("Mean IoU by Compression Level")
    ax.set_xticks(x, ratios)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_mean_iou_grouped_bars.png", dpi=220)
    plt.close(fig)


def plot_mean_nce_grouped_bars(df: pd.DataFrame) -> None:
    pivot = (
        df.pivot(index="ratio", columns="tracker_name", values="mean_normalized_center_error")
        .sort_index(ascending=False)
    )
    ratios = [f"{ratio:.2f}" for ratio in pivot.index.tolist()]
    x = np.arange(len(ratios), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    csrt_vals = pivot["csrt"].to_numpy() if "csrt" in pivot.columns else np.zeros(len(pivot))
    kcf_vals = pivot["kcf"].to_numpy() if "kcf" in pivot.columns else np.zeros(len(pivot))

    bars1 = ax.bar(x - width / 2, csrt_vals, width, label="CSRT")
    bars2 = ax.bar(x + width / 2, kcf_vals, width, label="KCF")

    ax.set_xlabel("Compression Level (Relative Bitrate Ratio)")
    ax.set_ylabel("Mean Normalized Center Error")
    ax.set_title("Mean Normalized Center Error by Compression Level")
    ax.set_xticks(x, ratios)
    upper = max(float(np.max(csrt_vals)) if len(csrt_vals) else 0.0, float(np.max(kcf_vals)) if len(kcf_vals) else 0.0)
    ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    offset = upper * 0.03 if upper > 0 else 0.02
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_mean_nce_grouped_bars.png", dpi=220)
    plt.close(fig)


def plot_update_success_grouped_bars(df: pd.DataFrame) -> None:
    pivot = (
        df.pivot(index="ratio", columns="tracker_name", values="update_success_rate")
        .sort_index(ascending=False)
    )
    ratios = [f"{ratio:.2f}" for ratio in pivot.index.tolist()]
    x = np.arange(len(ratios), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    csrt_vals = pivot["csrt"].to_numpy() if "csrt" in pivot.columns else np.zeros(len(pivot))
    kcf_vals = pivot["kcf"].to_numpy() if "kcf" in pivot.columns else np.zeros(len(pivot))

    bars1 = ax.bar(x - width / 2, csrt_vals, width, label="CSRT")
    bars2 = ax.bar(x + width / 2, kcf_vals, width, label="KCF")

    ax.set_xlabel("Compression Level (Relative Bitrate Ratio)")
    ax.set_ylabel("Average Update Success Rate")
    ax.set_title("Average Update Success Rate by Compression Level")
    ax.set_xticks(x, ratios)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_update_success_grouped_bars.png", dpi=220)
    plt.close(fig)


def plot_success050_vs_ratio(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    for tracker_name, tracker_df in df.groupby("tracker_name"):
        tracker_df = tracker_df.sort_values("ratio", ascending=False)
        plt.plot(
            tracker_df["ratio"],
            tracker_df["success_rate_iou_050"],
            marker="o",
            linewidth=2,
            label=tracker_name.upper(),
        )
    plt.xlabel("Relative Bitrate Ratio", fontsize=16)
    plt.ylabel("Success Rate at IoU >= 0.50", fontsize=16)
    plt.title("Success@0.50 vs Compression Level", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_success050_vs_ratio.png", dpi=220)
    plt.close()


def plot_success_auc_curve(per_frame: pd.DataFrame) -> None:
    thresholds = np.arange(0.05, 1.00, 0.05)
    plt.figure(figsize=(8, 5.5))
    colors = {"csrt": "tab:blue", "kcf": "tab:orange"}

    for tracker_name, tracker_df in per_frame.groupby("tracker_name"):
        per_video_rates = []

        for _, video_df in tracker_df.groupby("video_stem"):
            video_rates = [float((video_df["iou"] >= threshold).mean()) for threshold in thresholds]
            per_video_rates.append(video_rates)

        per_video_rates_arr = np.array(per_video_rates, dtype=float)
        mean_rates = per_video_rates_arr.mean(axis=0)
        sem = per_video_rates_arr.std(axis=0, ddof=1) / np.sqrt(max(per_video_rates_arr.shape[0], 1))
        sem = np.nan_to_num(sem, nan=0.0)
        ci95 = 1.96 * sem
        lower = np.clip(mean_rates - ci95, 0.0, 1.0)
        upper = np.clip(mean_rates + ci95, 0.0, 1.0)

        plt.fill_between(thresholds, lower, upper, alpha=0.18, color=colors.get(tracker_name))
        plt.plot(
            thresholds,
            mean_rates,
            linewidth=2,
            color=colors.get(tracker_name),
            label=f"{tracker_name.upper()}",
        )

    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.title("Success Curve (AUC Basis)")
    plt.xlim(float(thresholds.min()) - 0.01, float(thresholds.max()) + 0.01)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_success_auc_curve.png", dpi=220)
    plt.close()


def plot_mean_iou_boxplot(per_run: pd.DataFrame) -> None:
    order = ["csrt", "kcf"]
    data = [
        per_run.loc[per_run["tracker_name"] == tracker_name, "mean_iou"].dropna().tolist()
        for tracker_name in order
    ]

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot(data, tick_labels=[name.upper() for name in order])
    plt.ylabel("Mean IoU to Baseline")
    plt.title("Distribution of Mean IoU by Tracker")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_mean_iou_boxplot.png", dpi=220)
    plt.close()


def plot_robustness_vs_accuracy(per_run: pd.DataFrame) -> None:
    df = per_run.copy()
    df["robustness_loss"] = 1.0 - df["success_rate_iou_050"]

    plt.figure(figsize=(8, 5.5))
    for tracker_name, tracker_df in df.groupby("tracker_name"):
        plt.scatter(
            tracker_df["mean_iou"],
            tracker_df["robustness_loss"],
            alpha=0.75,
            s=40,
            label=tracker_name.upper(),
        )
    plt.xlabel("Mean IoU to Baseline")
    plt.ylabel("Robustness Loss (1 - Success@0.50)")
    plt.title("Accuracy vs Robustness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_accuracy_vs_robustness.png", dpi=220)
    plt.close()


def plot_per_video_mean_iou(per_run: pd.DataFrame) -> None:
    _plot_video_grid(
        per_run,
        metric="mean_iou",
        ylabel="Mean IoU",
        title="Per-Video Mean IoU vs Compression",
        output_name="08_per_video_mean_iou.png",
    )


def plot_per_video_success050(per_run: pd.DataFrame) -> None:
    _plot_video_grid(
        per_run,
        metric="success_rate_iou_050",
        ylabel="Success@0.50",
        title="Per-Video Success@0.50 vs Compression",
        output_name="09_per_video_success050.png",
    )


def plot_mean_iou_heatmap(per_run: pd.DataFrame) -> None:
    df = per_run.copy()
    df["video_label"] = df["video_stem"].map(VIDEO_DISPLAY_NAMES).fillna(df["video_stem"])
    tracker_order = ["csrt", "kcf"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), squeeze=False)

    for ax, tracker_name in zip(axes.flatten(), tracker_order):
        tracker_df = df[df["tracker_name"] == tracker_name].copy()
        pivot = tracker_df.pivot(index="video_label", columns="ratio", values="mean_iou")
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap=MEAN_IOU_CMAP)
        ax.set_title(tracker_name.upper())
        ax.set_xlabel("Relative Bitrate Ratio")
        ax.set_ylabel("Video")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{ratio:.2f}" for ratio in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())

        for row_idx in range(len(pivot.index)):
            for col_idx in range(len(pivot.columns)):
                value = float(pivot.iloc[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value >= 0.55 else "white",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean IoU")

    fig.suptitle("Per-Video Mean IoU Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_mean_iou_heatmap.png", dpi=220)
    plt.close(fig)


def plot_success050_heatmap(per_run: pd.DataFrame) -> None:
    df = per_run.copy()
    df["video_label"] = df["video_stem"].map(VIDEO_DISPLAY_NAMES).fillna(df["video_stem"])
    tracker_order = ["csrt", "kcf"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), squeeze=False)

    for ax, tracker_name in zip(axes.flatten(), tracker_order):
        tracker_df = df[df["tracker_name"] == tracker_name].copy()
        pivot = tracker_df.pivot(index="video_label", columns="ratio", values="success_rate_iou_050")
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap=MEAN_IOU_CMAP)
        ax.set_title(tracker_name.upper())
        ax.set_xlabel("Relative Bitrate Ratio")
        ax.set_ylabel("Video")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{ratio:.2f}" for ratio in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())

        for row_idx in range(len(pivot.index)):
            for col_idx in range(len(pivot.columns)):
                value = float(pivot.iloc[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value >= 0.55 else "white",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Success@0.50")

    fig.suptitle("Per-Video Success@0.50 Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_success050_heatmap.png", dpi=220)
    plt.close(fig)


def _plot_video_grid(
    per_run: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    videos = sorted(per_run["video_stem"].unique().tolist())
    cols = 3
    rows = (len(videos) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows), squeeze=False)

    for ax, video_stem in zip(axes.flatten(), videos):
        subset = per_run[per_run["video_stem"] == video_stem]
        for tracker_name, tracker_df in subset.groupby("tracker_name"):
            tracker_df = tracker_df.sort_values("ratio", ascending=False)
            ax.plot(
                tracker_df["ratio"],
                tracker_df[metric],
                marker="o",
                linewidth=2,
                label=tracker_name.upper(),
            )
        ax.set_title(video_stem)
        ax.set_xlabel("Ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    for ax in axes.flatten()[len(videos):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=220)
    plt.close(fig)


def plot_tracker_summary_bars(per_video: pd.DataFrame) -> None:
    summary = (
        per_video.groupby("tracker_name", as_index=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            success_rate_iou_050=("success_rate_iou_050", "mean"),
            mean_normalized_center_error=("mean_normalized_center_error", "mean"),
        )
        .sort_values("tracker_name")
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    labels = summary["tracker_name"].str.upper()

    axes[0].bar(labels, summary["mean_iou"])
    axes[0].set_title("Average Mean IoU")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, summary["success_rate_iou_050"])
    axes[1].set_title("Average Success@0.50")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(labels, summary["mean_normalized_center_error"])
    axes[2].set_title("Average Normalized Center Error")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Overall Tracker Summary")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_tracker_summary_bars.png", dpi=220)
    plt.close(fig)


def plot_stability_comparison(per_run: pd.DataFrame) -> None:
    summary = (
        per_run.groupby("tracker_name", as_index=False)
        .agg(
            update_success_rate=("update_success_rate", "mean"),
            mean_normalized_center_error=("mean_normalized_center_error", "mean"),
        )
        .sort_values("tracker_name")
    )

    labels = summary["tracker_name"].str.upper()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    axes[0].bar(labels, summary["update_success_rate"])
    axes[0].set_title("Average Update Success Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)
    for idx, value in enumerate(summary["update_success_rate"]):
        axes[0].text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, summary["mean_normalized_center_error"])
    axes[1].set_title("Average Normalized Center Error")
    axes[1].grid(True, axis="y", alpha=0.3)
    upper = float(summary["mean_normalized_center_error"].max()) if not summary.empty else 1.0
    axes[1].set_ylim(0, upper * 1.18 if upper > 0 else 1.0)
    for idx, value in enumerate(summary["mean_normalized_center_error"]):
        axes[1].text(idx, value + (upper * 0.03 if upper > 0 else 0.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Tracker Stability Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_stability_comparison.png", dpi=220)
    plt.close(fig)


def write_guide() -> None:
    guide = """# Report Plot Guide

## 01_mean_iou_vs_ratio.png
What it shows:
- Average overlap agreement with the original baseline trajectory as compression increases.

Why it matters:
- This is the main summary of how much each tracker's output changes under compression.
- Higher values mean the compressed-video run stays closer to the original-video baseline.

## 02_success050_vs_ratio.png
## 02_mean_iou_grouped_bars.png
What it shows:
- A grouped bar chart of average mean IoU at each compression level, with separate bars for CSRT and KCF.

Why it matters:
- This is a direct ratio-by-ratio comparison between the two trackers.
- It is useful when the goal is to compare algorithms at each compression level rather than emphasizing trend lines.

## 03_mean_nce_grouped_bars.png
What it shows:
- A grouped bar chart of average mean normalized center error at each compression level, with separate bars for CSRT and KCF.

Why it matters:
- Lower values indicate that the tracked box center stayed closer to the baseline trajectory.
- This complements mean IoU by showing localization drift directly.

## 04_update_success_grouped_bars.png
What it shows:
- A grouped bar chart of average update success rate at each compression level, with separate bars for CSRT and KCF.

Why it matters:
- This directly shows how often each tracker continued reporting successful updates as compression increased.
- It is useful for identifying abrupt tracker breakdown under severe compression.

## 05_success050_vs_ratio.png
What it shows:
- Fraction of runs whose framewise overlap stays at or above IoU 0.50.

Why it matters:
- Mean IoU can hide failures. Success@0.50 is easier to interpret as a practical "usable tracking" threshold.

## 03_accuracy_vs_robustness.png
## 06_success_auc_curve.png
What it shows:
- Success rate as the IoU threshold becomes stricter.
- The shaded band shows variability across videos.

Why it matters:
- This is the underlying success curve whose area corresponds to success AUC.
- A curve that stays higher for longer indicates stronger overlap consistency across strict and lenient thresholds.

## 07_accuracy_vs_robustness.png
What it shows:
- Each point is one run.
- X-axis is mean IoU.
- Y-axis is robustness loss, defined as 1 - Success@0.50.

Why it matters:
- This separates trackers that have high average overlap from trackers that also remain stable across many frames.
- Lower and farther right is better.

## 08_per_video_mean_iou.png
What it shows:
- Mean IoU vs compression for each video separately.

Why it matters:
- Compression does not affect every scene the same way.
- This figure reveals whether a tracker is consistently strong or only strong on easier clips.

## 09_per_video_success050.png
What it shows:
- Success@0.50 vs compression for each video separately.

Why it matters:
- This is one of the clearest ways to identify catastrophic failures on specific sequences.
- It is especially useful for discussing difficult scenes such as clutter, glare, or similar-object confusion.

## 12_mean_iou_heatmap.png
What it shows:
- Side-by-side heatmaps of mean IoU by video and compression level, with one panel for CSRT and one panel for KCF.

Why it matters:
- This gives a compact summary of scene-specific behavior across the full dataset.
- It is especially useful for comparing which videos favor one tracker over the other at each compression level.

## 13_success050_heatmap.png
What it shows:
- Side-by-side heatmaps of success@0.50 by video and compression level, with one panel for CSRT and one panel for KCF.

Why it matters:
- This is a compact replacement for the multi-panel per-video success plot.
- It makes scene-specific tracker failures and strong-performing regions immediately visible.

## 14_mean_iou_boxplot.png
What it shows:
- A boxplot of the distribution of per-run mean IoU values for CSRT and KCF.

Why it matters:
- This summarizes spread, median performance, and outliers.
- It is useful for showing whether one tracker is more variable or more consistently strong across runs.

## 10_tracker_summary_bars.png
What it shows:
- Overall tracker averages for mean IoU, Success@0.50, and normalized center error.

Why it matters:
- This is a compact summary figure for the paper.
- It gives a quick comparison between trackers using one overlap metric, one thresholded success metric, and one localization-error metric.

## 11_stability_comparison.png
What it shows:
- A direct tracker-to-tracker comparison of average update success rate and average normalized center error.

Why it matters:
- This figure is useful when discussing stability.
- Higher update success and lower center error indicate that a tracker is less likely to break down and less likely to drift away from the baseline trajectory.
"""
    GUIDE_PATH.write_text(guide)


if __name__ == "__main__":
    main()
