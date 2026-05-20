from pathlib import Path

import pytest

from undertow.cli import main


def test_cli_smoke_generate_run_and_summarize(tmp_path: Path, capsys):
    task_path = tmp_path / "task.json"
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.csv"

    assert main(["generate", "--family", "hidden_dag", "--seed", "31", "--output", str(task_path)]) == 0
    assert main(["run", "--task", str(task_path), "--mode", "oracle", "--output", str(results_path)]) == 0
    assert main(["summarize", "--input", str(results_path), "--csv", str(summary_path)]) == 0

    captured = capsys.readouterr()
    assert "solve_rate" in captured.out
    assert summary_path.exists()


def test_cli_saves_hf_dataset(tmp_path: Path):
    pytest.importorskip("datasets")
    output_path = tmp_path / "dataset"

    assert (
        main(
            [
                "dataset",
                "--mode",
                "standard",
                "--family",
                "hidden_dag",
                "--max-examples",
                "1",
                "--params",
                '{"n_core": 1, "depth": 1}',
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    assert (output_path / "dataset_info.json").exists()


def test_cli_rejects_model_eval_modes(tmp_path: Path, capsys):
    task_path = tmp_path / "task.json"
    assert main(["generate", "--family", "hidden_dag", "--seed", "32", "--output", str(task_path)]) == 0

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--task",
                str(task_path),
                "--mode",
                "standard",
            ]
        )
    assert exc.value.code == 2
    assert "invalid choice: 'standard'" in capsys.readouterr().err


def test_cli_rejects_removed_model_flags(tmp_path: Path, capsys):
    task_path = tmp_path / "task.json"
    assert main(["generate", "--family", "hidden_dag", "--seed", "33", "--output", str(task_path)]) == 0

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--task",
                str(task_path),
                "--mode",
                "oracle",
                "--model",
                "gpt-test",
            ]
        )

    assert exc.value.code == 2
    assert "unrecognized arguments: --model gpt-test" in capsys.readouterr().err
