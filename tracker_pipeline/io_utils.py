from __future__ import annotations

from pathlib import Path
import json
import subprocess


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def run_command(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def ffprobe_json(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_command(cmd)
    return json.loads(result.stdout)
