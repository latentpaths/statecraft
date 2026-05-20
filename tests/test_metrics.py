from undertow.core.families.hidden_dag import HiddenDagEnv, generate_hidden_dag_task
from undertow.core.families.repair_chain import RepairChainEnv, generate_repair_chain_task
from undertow.core.metrics import summarize
from undertow.core.types import FailureCode


def test_episode_result_includes_budget_progress_and_failure_metrics():
    task = generate_hidden_dag_task(seed=41, n_core=2, depth=2)
    env = HiddenDagEnv(task)
    first = task.hidden_state["functions"][0]
    second = task.hidden_state["functions"][1]
    initial_value = task.public_spec["initial_values"][first["inputs"][0]]

    first_obs = env.call(first["name"], **{first["inputs"][0]: initial_value})
    env.call(second["name"], **{second["inputs"][0]: 1})
    env.submit({"value": first_obs["value"]})
    result = env.result()

    assert result.budget == task.budget
    assert result.calls == 3
    assert result.budget == task.budget
    assert result.reference_calls == len(task.public_spec["functions"]) + 1
    assert result.efficiency == result.reference_calls / max(result.calls, result.reference_calls)
    assert result.progress == 0.5
    assert result.reward == result.progress * result.efficiency
    assert result.state_changes == 1
    assert result.failure_calls == 2
    assert result.first_failure == FailureCode.INCORRECT_VALUE
    assert result.final_failure == FailureCode.PREMATURE_SUBMIT


def test_repair_chain_progress_tracks_applied_repair_stages():
    task = generate_repair_chain_task(seed=42, n_objects=1, n_repair_steps=2)
    env = RepairChainEnv(task)
    object_id, obj = next(iter(task.hidden_state["objects"].items()))
    scan = env.call("scan_object", object_id=object_id)
    env.call("move_to", location=scan["location"])
    env.call(
        "apply_tool",
        object_id=object_id,
        tool_id=obj["required_tools"][0],
        observation_version=scan["observation_version"],
    )

    assert env.result().progress == 0.5


def test_summarize_reports_efficiency_and_failure_rates():
    rows = [
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "solved": True,
            "reward": 1.0,
            "calls": 2,
            "budget": 5,
            "efficiency": 1.0,
            "progress": 1.0,
            "state_changes": 2,
            "failure_calls": 0,
            "failure_counter": {},
        },
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "solved": False,
            "reward": 0.0625,
            "calls": 4,
            "budget": 4,
            "efficiency": 0.5,
            "progress": 0.25,
            "state_changes": 1,
            "failure_calls": 1,
            "failure_counter": {"budget_exceeded": 1},
        },
    ]

    summary = summarize(rows)[0]

    assert summary["solve_rate"] == 0.5
    assert summary["avg_reward"] == 0.53125
    assert summary["avg_progress"] == 0.625
    assert summary["avg_efficiency"] == 0.75
    assert summary["avg_calls"] == 3.0
    assert summary["avg_state_changes"] == 1.5
    assert summary["avg_failure_calls"] == 0.5
    assert summary["budget_exhaustion_rate"] == 0.5
