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
    assert result.remaining_budget == task.budget - result.num_calls
    assert result.budget_used_ratio == result.num_calls / task.budget
    assert result.progress_score == 0.5
    assert result.useful_calls == 1
    assert result.wasted_calls == 2
    assert result.first_failure == FailureCode.INCORRECT_VALUE
    assert result.final_failure == FailureCode.PREMATURE_SUBMIT


def test_repair_chain_progress_score_tracks_applied_repair_stages():
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

    assert env.result().progress_score == 0.5


def test_summarize_reports_efficiency_and_failure_rates():
    rows = [
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "success": True,
            "num_calls": 2,
            "useful_calls": 2,
            "remaining_budget": 3,
            "failure_counter": {},
        },
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "success": False,
            "num_calls": 4,
            "useful_calls": 1,
            "remaining_budget": 0,
            "progress_score": 0.25,
            "failure_counter": {"budget_exceeded": 1},
        },
    ]

    summary = summarize(rows)[0]

    assert summary["success_rate"] == 0.5
    assert summary["avg_calls"] == 3.0
    assert summary["avg_calls_success"] == 2.0
    assert summary["avg_calls_failure"] == 4.0
    assert summary["avg_failed_progress"] == 0.25
    assert summary["budget_exhaustion_rate"] == 0.5
    assert summary["useful_call_rate"] == 0.5
