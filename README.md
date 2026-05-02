# Object Tracking Compression Robustness Experiment

This project evaluates how video compression affects the robustness of two OpenCV single-object trackers:

- `TrackerCSRT`
- `TrackerKCF`

The experiment compares tracking on compressed videos against tracking on the corresponding original videos, treating the original-video tracking output as the baseline reference.

## What This Repository Contains

This repository includes:

- the Python pipeline for annotation, compression, tracking, evaluation, and plotting
- annotation metadata
- compression metadata
- metric CSV outputs

This repository does **not** include:

- original videos
- compressed videos
- rendered tracking videos
- generated plot/image outputs

Those files are excluded through `.gitignore` because they are too large for GitHub.

## Project Structure

- `tracker_pipeline/` — main experiment pipeline
- `report_plots.py` — script for generating paper-ready figures from metric CSVs
- `data/annotations/first_frame_boxes.csv` — saved first-frame target boxes
- `results/metadata/` — source/compression metadata CSVs
- `results/metrics/` — evaluation metric CSVs

## Requirements

You need:

- Python 3
- FFmpeg installed and available on the command line
- Python packages listed in `requirements.txt`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Videos

Place the original `.mp4` videos in:

```bash
data/videos/originals/
```

These source videos are not included in the repository.

## Reproducing the Experiment

From the project root:

```bash
source .venv/bin/activate
python3 -m tracker_pipeline.cli annotate
python3 -m tracker_pipeline.cli compress
python3 -m tracker_pipeline.cli track
python3 -m tracker_pipeline.cli evaluate
```

Or run the full pipeline:

```bash
python3 -m tracker_pipeline.cli run-all
```

## Reproducing the Paper Figures

After the metric CSVs have been generated, create the report figures with:

```bash
source .venv/bin/activate
python3 report_plots.py
```

## Main Outputs

Important generated CSV files:

- `data/annotations/first_frame_boxes.csv`
- `results/metadata/videos.csv`
- `results/metadata/compressions.csv`
- `results/metrics/per_run_summary.csv`
- `results/metrics/per_video_tracker_summary.csv`
- `results/metrics/per_tracker_bitrate_summary.csv`
- `results/metrics/per_frame_metrics.csv`

## Notes

- Compression uses H.264 with `libx264` two-pass encoding.
- The relative bitrate ladder used in the experiment is:
  - `0.30`
  - `0.15`
  - `0.08`
  - `0.04`
  - `0.02`
- The original-video tracker output is used as the baseline reference, so the reported metrics measure consistency under compression rather than absolute ground-truth tracking accuracy.
