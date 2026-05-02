# Object Tracking Compression Robustness Pipeline

This project evaluates how video compression affects OpenCV single-object trackers using a relative bitrate ladder.

## What it does

1. Loads original source videos from `data/videos/originals/`.
2. Lets you draw a first-frame bounding box for each video.
3. Compresses each original video with FFmpeg using a relative bitrate ladder.
4. Runs OpenCV `TrackerCSRT` and `TrackerKCF` on the original and compressed videos.
5. Compares compressed tracking outputs against the original-video tracker outputs.
6. Saves detailed CSV logs, summary CSVs, plots, and optional annotated tracking videos.

## Video Input Folder

Place your original videos here:

`data/videos/originals/`

For your dataset, that would be files like:

- `data/videos/originals/baseline.mp4`
- `data/videos/originals/butterfly.mp4`
- `data/videos/originals/car.mp4`
- `data/videos/originals/ducks.mp4`
- `data/videos/originals/runner_dog.mp4`
- `data/videos/originals/sailboats.mp4`
- `data/videos/originals/traffic.mp4`
- `data/videos/originals/walk_into_sun.mp4`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the full workflow:

```bash
python3 -m tracker_pipeline.cli run-all
```

This will:

1. prompt for first-frame boxes if they do not already exist,
2. compress the videos,
3. run baseline tracking on originals,
4. run tracking on compressed versions,
5. compute metrics,
6. generate CSV summaries and plots.

## Recommended Workflow

### 1. Annotate first-frame boxes

```bash
python3 -m tracker_pipeline.cli annotate
```

Controls in the OpenCV ROI window:

- drag a box around the target,
- press `Enter` or `Space` to confirm,
- press `c` to cancel selection.

Annotations are saved to:

- `data/annotations/first_frame_boxes.csv`

### 2. Compress videos

```bash
python3 -m tracker_pipeline.cli compress
```

### 3. Run tracking

```bash
python3 -m tracker_pipeline.cli track
```

### 4. Evaluate and plot

```bash
python3 -m tracker_pipeline.cli evaluate
```

## Relative Bitrate Ladder

The default ladder is:

- `0.30x`
- `0.15x`
- `0.08x`
- `0.04x`
- `0.02x`

The original video is used as the baseline reference and is not re-encoded for the baseline condition.

The bitrate anchor is estimated from the original file bitrate if available. If not available, it falls back to:

`file_size_bits / duration_seconds`

## Main Outputs

### Metadata and annotations

- `data/annotations/first_frame_boxes.csv`
- `results/metadata/videos.csv`
- `results/metadata/compressions.csv`

### Tracking outputs

- `results/tracks/per_frame/*.csv`
- `results/tracks/annotated_videos/*.mp4`

### Evaluation outputs

- `results/metrics/per_run_summary.csv`
- `results/metrics/per_video_tracker_summary.csv`
- `results/metrics/per_tracker_bitrate_summary.csv`
- `results/metrics/per_frame_metrics.csv`

### Plots

- `results/plots/*.png`

## Metrics

Compressed runs are compared against the same tracker on the original video.

Main metrics:

- mean IoU to baseline
- success rate at IoU thresholds `0.25`, `0.50`, `0.75`
- success AUC across IoU thresholds
- mean normalized center error
- median normalized center error
- update success rate
- time to first failure
- longest consecutive low-IoU streak

## Notes

- Compression uses H.264 `libx264` two-pass encoding for controlled bitrate targets.
- Resolution and frame rate are preserved.
- Tracking jobs and compression jobs can run in parallel across videos.
- Existing outputs are reused unless `--force` is passed.

## Useful Commands

Show current configuration:

```bash
python3 -m tracker_pipeline.cli show-config
```

Force re-run compression:

```bash
python3 -m tracker_pipeline.cli compress --force
```

Track only one video:

```bash
python3 -m tracker_pipeline.cli track --video baseline.mp4
```

Skip annotated video rendering for faster runtime:

```bash
python3 -m tracker_pipeline.cli track --no-render
```
