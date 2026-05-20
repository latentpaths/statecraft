from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def raw_dir_for_config(config_path: Path) -> Path:
    return HERE / "results" / config_path.resolve().stem / "raw"


def result_dir_for_raw(raw_dir: Path) -> Path:
    raw_dir = raw_dir.resolve()
    if raw_dir.name == "raw":
        return raw_dir.parent
    return raw_dir


def figures_dir_for_raw(raw_dir: Path) -> Path:
    return result_dir_for_raw(raw_dir) / "figures"
