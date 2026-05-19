import importlib.util

import pytest

from undertow.core import UndertowEnv
from undertow.core import generate_dataset_records, save_hf_dataset, to_hf_dataset
from undertow.core.runtime import env_for_task, generate_task, generate_tasks, run_oracle


def test_core_has_no_verifiers_adapter_module():
    assert importlib.util.find_spec("undertow.core.verifiers") is None


def test_runtime_generates_and_reconstructs_supported_families():
    tasks = generate_tasks(seed=100, max_examples=2)

    assert [task.family for task in tasks] == ["hidden_dag", "repair_chain"]
    assert all(isinstance(env_for_task(task), UndertowEnv) for task in tasks)


@pytest.mark.parametrize("family", ["hidden_dag", "repair_chain"])
def test_runtime_oracle_solves_generated_tasks(family):
    task = generate_task(family, seed=101)
    env = env_for_task(task)

    result = run_oracle(env)

    assert result.success is True
    assert result.mode == "oracle"


def test_generated_env_exposes_native_tool_schema():
    task = generate_task("hidden_dag", seed=102, params={"n_core": 1, "depth": 1})
    env = env_for_task(task)

    assert env.tools()[0]["name"] == "core_0"


def test_snapshot_restore_round_trips_trace_and_result():
    task = generate_task("hidden_dag", seed=103, params={"n_core": 1, "depth": 1})
    env = env_for_task(task)
    func = task.hidden_state["functions"][0]
    value = task.public_spec["initial_values"][func["inputs"][0]]

    env.call(func["name"], **{func["inputs"][0]: value})
    env.submit({"value": task.target["value"]})
    restored = env_for_task(task, env.snapshot())

    assert restored.trace() == env.trace()
    assert restored.result().to_dict() == env.result().to_dict()


@pytest.mark.parametrize("mode", ["standard", "ptc"])
def test_dataset_generation_works_for_modes(mode):
    records = generate_dataset_records(
        mode=mode,
        family="hidden_dag",
        seed=104,
        max_examples=1,
        task_params={"n_core": 1, "depth": 1},
    )

    assert records[0]["info"]["family"] == "hidden_dag"
    assert records[0]["prompt"][0]["role"] == "system"


def test_core_records_save_as_hf_dataset(tmp_path):
    pytest.importorskip("datasets")

    records = generate_dataset_records(
        mode="standard",
        family="hidden_dag",
        seed=105,
        max_examples=1,
        task_params={"n_core": 1, "depth": 1},
    )
    dataset = to_hf_dataset(records)
    output_path = tmp_path / "undertow-dataset"

    save_hf_dataset(
        output_path,
        mode="standard",
        family="hidden_dag",
        seed=105,
        max_examples=1,
        task_params={"n_core": 1, "depth": 1},
    )

    assert dataset[0]["info"]["task_id"] == records[0]["info"]["task_id"]
    assert (output_path / "dataset_info.json").exists()
