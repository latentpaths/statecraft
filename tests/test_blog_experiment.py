import csv
import importlib.util
import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "blog"


def load_aggregate_module():
    spec = importlib.util.spec_from_file_location("blog_aggregate", EXPERIMENT_DIR / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_module():
    spec = importlib.util.spec_from_file_location("blog_run", EXPERIMENT_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_normalizes_latest_raw_schema():
    run = load_run_module()
    row = {
        "reward": 0.5,
        "token_usage": {"input_tokens": 100, "output_tokens": 25},
        "metrics": {
            "num_turns": 4,
            "undertow_solved": 0,
            "undertow_calls": 6,
            "undertow_budget": 10,
            "undertow_reference_calls": 5,
            "undertow_progress": 0.5,
            "undertow_failed_calls": 2,
            "undertow_failure_invalid_action": 0,
            "undertow_failure_schema_error": 0,
            "undertow_failure_unknown_state_reference": 0,
            "undertow_failure_incorrect_value": 2,
            "undertow_failure_stale_state_reference": 0,
            "undertow_failure_violated_constraint": 0,
            "undertow_failure_distractor_action": 0,
            "undertow_failure_premature_submit": 0,
            "undertow_failure_budget_exceeded": 0,
            "undertow_failure_repeated_call": 0,
        },
    }
    job = {
        "config": "medium",
        "mode": "standard",
        "family": "hidden_dag",
        "model": "m",
        "seed": 3,
        "task_params": {"n_core": 5},
    }

    raw = run.normalize_prime_row(row, job)

    assert {"config", "seed", "num_turns", "input_tokens", "output_tokens", "total_tokens"} <= set(raw)
    assert raw["config"] == "medium"
    assert raw["seed"] == 3
    assert "efficiency" not in raw
    assert raw["failed_calls"] == 2.0
    assert raw["failure_counter"] == {"incorrect_value": 2}
    assert raw["total_tokens"] == 125.0


def test_run_timeout_row_uses_lean_schema():
    run = load_run_module()
    job = {
        "config": "hard",
        "mode": "ptc",
        "family": "hidden_dag",
        "model": "m",
        "seed": 0,
        "task_params": {"n_core": 1, "depth": 1},
    }

    row = run.timeout_row(job)

    assert row["solved"] is False
    assert row["reward"] == 0.0
    assert row["progress"] == 0.0
    assert row["failed_calls"] == 1.0
    assert row["failure_counter"] == {"timeout": 1}
    assert "failure_calls" not in row


def test_aggregate_writes_config_and_per_model_research_csvs(tmp_path: Path):
    aggregate = load_aggregate_module()
    result_dir = tmp_path / "medium"
    rows = [
        make_row(
            model="deepseek/deepseek-v4-flash",
            mode="standard",
            family="hidden_dag",
            params={"n_core": 5, "depth": 2},
            solved=True,
            calls=5.0,
            budget=10.0,
            failure_counter={},
        ),
        make_row(
            model="deepseek/deepseek-v4-flash",
            mode="ptc",
            family="hidden_dag",
            params={"n_core": 5, "depth": 2},
            solved=False,
            calls=10.0,
            budget=10.0,
            failure_counter={"incorrect_value": 2},
        ),
        make_row(
            model="qwen/qwen3.6-35b-a3b",
            mode="standard",
            family="repair_chain",
            params={"n_objects": 3, "n_repair_steps": 2},
            solved=True,
            calls=4.0,
            budget=10.0,
            failure_counter={"repeated_call": 1},
        ),
    ]
    write_raw(
        result_dir
        / "deepseek--deepseek-v4-flash"
        / "raw"
        / "mode-standard__family-hidden_dag__seed-0__params-a.jsonl",
        [rows[0]],
    )
    write_raw(
        result_dir
        / "deepseek--deepseek-v4-flash"
        / "raw"
        / "mode-ptc__family-hidden_dag__seed-1__params-a.jsonl",
        [rows[1]],
    )
    write_raw(
        result_dir
        / "qwen--qwen3.6-35b-a3b"
        / "raw"
        / "mode-standard__family-repair_chain__seed-0__params-b.jsonl",
        [rows[2]],
    )

    assert aggregate.main(["--input", str(result_dir)]) == 0

    config_csvs = result_dir / "csvs"
    assert csv_models(config_csvs / "episodes.csv") == {
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3.6-35b-a3b",
    }
    episodes_header = (config_csvs / "episodes.csv").read_text().splitlines()[0]
    assert "config" in episodes_header
    assert "total_tokens" in episodes_header
    assert "failed_calls" in episodes_header
    assert "call_ratio" not in episodes_header
    assert "n_core" in episodes_header
    assert "n_objects" in (config_csvs / "by_params.csv").read_text()
    assert "avg_failed_calls" in (config_csvs / "summary.csv").read_text().splitlines()[0]
    assert "incorrect_value" in (config_csvs / "failures.csv").read_text()
    assert "repeated_call" in (config_csvs / "failure_by_config.csv").read_text()
    assert (config_csvs / "by_config_name.csv").exists()

    model_csvs = result_dir / "deepseek--deepseek-v4-flash" / "csvs"
    assert csv_models(model_csvs / "episodes.csv") == {"deepseek/deepseek-v4-flash"}
    assert (model_csvs / "summary.csv").read_text().count("\n") == 2
    assert "seed" in (model_csvs / "by_seed.csv").read_text().splitlines()[0]

    qwen_csvs = result_dir / "qwen--qwen3.6-35b-a3b" / "csvs"
    assert csv_models(qwen_csvs / "episodes.csv") == {"qwen/qwen3.6-35b-a3b"}


def test_aggregate_rejects_inconsistent_or_old_rows():
    aggregate = load_aggregate_module()
    old_row = make_row(
        model="m",
        mode="standard",
        family="hidden_dag",
        params={"n_core": 1},
        solved=False,
        calls=1.0,
        budget=3.0,
        failure_counter={},
    )
    old_row["failure_calls"] = old_row.pop("failed_calls")

    try:
        aggregate.normalize_row(old_row)
    except ValueError as exc:
        assert "old metric fields" in str(exc)
    else:
        raise AssertionError("old metric row was accepted")

    bad_row = make_row(
        model="m",
        mode="standard",
        family="hidden_dag",
        params={"n_core": 1},
        solved=False,
        calls=1.0,
        budget=3.0,
        failure_counter={},
    )
    bad_row["reward"] = 0.1

    try:
        aggregate.normalize_row(bad_row)
    except ValueError as exc:
        assert "unsolved rows must have reward=0" in str(exc)
    else:
        raise AssertionError("inconsistent reward row was accepted")


def test_raw_dir_for_config_matches_hybrid_layout():
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from paths import (
        config_csvs_dir,
        config_figures_dir,
        config_result_dir,
        model_csvs_dir,
        model_figures_dir,
        model_logs_dir,
        model_result_dir,
        model_slug,
        raw_dir_for_config,
    )

    config = EXPERIMENT_DIR / "configs" / "medium.json"
    model = "deepseek/deepseek-v4-flash"

    assert model_slug(model) == "deepseek--deepseek-v4-flash"
    assert config_result_dir(config) == EXPERIMENT_DIR / "results" / "medium"
    assert model_result_dir(config, model) == EXPERIMENT_DIR / "results" / "medium" / "deepseek--deepseek-v4-flash"
    assert raw_dir_for_config(config, model) == EXPERIMENT_DIR / "results" / "medium" / "deepseek--deepseek-v4-flash" / "raw"
    assert raw_dir_for_config(config, model, "20260522-120000-123456") == (
        EXPERIMENT_DIR
        / "results"
        / "medium"
        / "deepseek--deepseek-v4-flash"
        / "20260522-120000-123456"
        / "raw"
    )
    assert config_csvs_dir(config) == EXPERIMENT_DIR / "results" / "medium" / "csvs"
    assert model_csvs_dir(config, model) == EXPERIMENT_DIR / "results" / "medium" / "deepseek--deepseek-v4-flash" / "csvs"
    assert config_figures_dir(config) == EXPERIMENT_DIR / "results" / "medium" / "figures"
    assert model_figures_dir(config, model) == EXPERIMENT_DIR / "results" / "medium" / "deepseek--deepseek-v4-flash" / "figures"
    assert model_logs_dir(config, model) == EXPERIMENT_DIR / "logs" / "medium" / "deepseek--deepseek-v4-flash"
    assert model_logs_dir(config, model, "20260522-120000-123456") == (
        EXPERIMENT_DIR
        / "logs"
        / "medium"
        / "deepseek--deepseek-v4-flash"
        / "20260522-120000-123456"
    )


def test_aggregate_finds_timestamped_raw_dirs(tmp_path: Path):
    aggregate = load_aggregate_module()
    row = make_row(
        model="deepseek/deepseek-v4-flash",
        mode="standard",
        family="hidden_dag",
        params={"n_core": 5},
        solved=True,
        calls=5.0,
        budget=10.0,
        failure_counter={},
    )
    write_raw(
        tmp_path
        / "medium"
        / "deepseek--deepseek-v4-flash"
        / "20260522-120000-123456"
        / "raw"
        / "mode-standard__family-hidden_dag__seed-0__params-a.jsonl",
        [row],
    )

    files = aggregate.raw_files(tmp_path / "medium")

    assert len(files) == 1
    assert "20260522-120000-123456" in str(files[0])


def test_run_log_path_uses_job_identity(tmp_path: Path):
    run = load_run_module()
    job = {
        "mode": "standard",
        "family": "hidden_dag",
        "seed": 3,
        "task_params": {"depth": 2, "n_core": 5},
    }

    path = run.log_path_for_job(job, tmp_path)

    assert path.parent == tmp_path
    assert path.name.startswith("mode-standard__family-hidden_dag__seed-3__params-")
    assert path.suffix == ".log"


def test_blog_experiment_default_config_is_medium_grid():
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from paths import DEFAULT_CONFIG

    config = json.loads(DEFAULT_CONFIG.read_text())

    assert DEFAULT_CONFIG.name == "medium.json"
    assert config["modes"] == ["standard", "ptc"]
    assert config["seeds"] == [0, 1]
    assert config["families"]["hidden_dag"]["n_core"] == [40]
    assert config["families"]["repair_chain"]["n_objects"] == [16]


def make_row(
    *,
    model: str,
    mode: str,
    family: str,
    params: dict[str, object],
    solved: bool,
    calls: float,
    budget: float,
    failure_counter: dict[str, int],
) -> dict[str, object]:
    return {
        "config": "medium",
        "seed": 0,
        "mode": mode,
        "family": family,
        "model": model,
        "params": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "solved": solved,
        "reward": 1.0 if solved else 0.0,
        "calls": calls,
        "budget": budget,
        "reference_calls": 5.0,
        "progress": 1.0 if solved else 0.4,
        "failed_calls": float(sum(failure_counter.values())),
        "failure_counter": failure_counter,
        "num_turns": 4.0,
        "input_tokens": 100.0,
        "output_tokens": 25.0,
        "total_tokens": 125.0,
    }


def write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def csv_models(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["model"] for row in csv.DictReader(handle)}
