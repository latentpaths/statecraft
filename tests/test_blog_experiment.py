import importlib.util
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "blog"


def load_aggregate_module():
    spec = importlib.util.spec_from_file_location("blog_aggregate", EXPERIMENT_DIR / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_writes_summary_by_config_and_failures(tmp_path: Path):
    aggregate = load_aggregate_module()
    result_dir = tmp_path / "configs_hard"
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(parents=True)
    rows = [
        {
            "mode": "standard",
            "family": "hidden_dag",
            "model": "m",
            "params": '{"n_core":5}',
            "solved": True,
            "reward": 1.0,
            "calls": 5.0,
            "budget": 10.0,
            "reference_calls": 5.0,
            "efficiency": 1.0,
            "progress": 1.0,
            "state_changes": 5.0,
            "failure_calls": 0.0,
            "failure_counter": {},
            "first_failure": None,
            "final_failure": None,
        },
        {
            "mode": "standard",
            "family": "hidden_dag",
            "model": "m",
            "params": '{"n_core":5}',
            "solved": False,
            "reward": 0.1,
            "calls": 10.0,
            "budget": 10.0,
            "reference_calls": 5.0,
            "efficiency": 0.5,
            "progress": 0.4,
            "state_changes": 3.0,
            "failure_calls": 2.0,
            "failure_counter": {"incorrect_value": 2},
            "first_failure": "incorrect_value",
            "final_failure": "incorrect_value",
        },
    ]
    with (raw_dir / "sample.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert aggregate.main(["--input", str(raw_dir)]) == 0

    assert (result_dir / "summary.csv").read_text().count("\n") == 2
    assert "incorrect_value" in (result_dir / "failures.csv").read_text()
    assert "n_core" in (result_dir / "by_config.csv").read_text()


def test_raw_dir_for_config_matches_config_stem():
    import sys

    sys.path.insert(0, str(EXPERIMENT_DIR))
    from paths import figures_dir_for_raw, raw_dir_for_config, result_dir_for_raw

    config = EXPERIMENT_DIR / "configs" / "configs_hard.json"
    raw_dir = raw_dir_for_config(config)

    assert raw_dir == EXPERIMENT_DIR / "results" / "configs_hard" / "raw"
    assert result_dir_for_raw(raw_dir) == EXPERIMENT_DIR / "results" / "configs_hard"
    assert figures_dir_for_raw(raw_dir) == EXPERIMENT_DIR / "results" / "configs_hard" / "figures"


def test_blog_experiment_config_is_the_fixed_grid():
    config = json.loads((EXPERIMENT_DIR / "configs" / "configs.json").read_text())

    assert config["modes"] == ["standard", "ptc"]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["families"]["hidden_dag"]["n_core"] == [5, 10]
    assert config["families"]["repair_chain"]["n_objects"] == [1, 3]
