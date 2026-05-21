from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS_DIR = HERE / "configs"
RESULTS_DIR = HERE / "results"
DEFAULT_CONFIG = CONFIGS_DIR / "medium.json"


def model_slug(model: str) -> str:
    return model.replace("/", "--")


def config_result_dir(config_path: Path) -> Path:
    return RESULTS_DIR / config_path.resolve().stem


def model_result_dir(config_path: Path, model: str) -> Path:
    return config_result_dir(config_path) / model_slug(model)


def raw_dir_for_config(config_path: Path, model: str) -> Path:
    return model_result_dir(config_path, model) / "raw"


def config_csvs_dir(config_path: Path) -> Path:
    return config_result_dir(config_path) / "csvs"


def model_csvs_dir(config_path: Path, model: str) -> Path:
    return model_result_dir(config_path, model) / "csvs"


def config_figures_dir(config_path: Path) -> Path:
    return config_result_dir(config_path) / "figures"


def model_figures_dir(config_path: Path, model: str) -> Path:
    return model_result_dir(config_path, model) / "figures"
