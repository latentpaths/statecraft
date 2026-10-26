from __future__ import annotations

from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIGS_DIR = HERE / "configs"
RESULTS_DIR = HERE / "results"
LOGS_DIR = HERE / "logs"
DEFAULT_CONFIG = CONFIGS_DIR / "medium.json"


def model_slug(model: str) -> str:
    return model.replace("/", "--")


def config_result_dir(config_path: Path) -> Path:
    return RESULTS_DIR / config_path.resolve().stem


def model_result_dir(config_path: Path, model: str) -> Path:
    return config_result_dir(config_path) / model_slug(model)


def raw_dir_for_config(config_path: Path, model: str, run_id: str | None = None) -> Path:
    base = model_result_dir(config_path, model)
    if run_id is not None:
        base = base / run_id
    return base / "raw"


def model_logs_dir(config_path: Path, model: str, run_id: str | None = None) -> Path:
    base = LOGS_DIR / config_path.resolve().stem / model_slug(model)
    if run_id is not None:
        base = base / run_id
    return base


def config_csvs_dir(config_path: Path) -> Path:
    return config_result_dir(config_path) / "csvs"


def model_csvs_dir(config_path: Path, model: str) -> Path:
    return model_result_dir(config_path, model) / "csvs"


def config_figures_dir(config_path: Path) -> Path:
    return config_result_dir(config_path) / "figures"


def model_figures_dir(config_path: Path, model: str) -> Path:
    return model_result_dir(config_path, model) / "figures"


def raw_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path]
    if path.name == "raw":
        return sorted(path.glob("*.jsonl"))
    return sorted(path.rglob("raw/*.jsonl"))


def infer_result_dir(input_path: Path, config_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = input_path.resolve()
    if not input_path.exists():
        return config_result_dir(config_path)
    if path.is_file():
        path = path.parent
    if path.name == "raw":
        parent = path.parent
        models = {model_slug(str(row["model"])) for row in rows}
        if parent.name in models:
            return parent.parent
        if parent.parent.name in models:
            return parent.parent.parent
        return parent
    return path
