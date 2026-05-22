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
    assert result.progress == 1 / 3
    assert result.reward == 0.0
    assert result.failed_calls == 2
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

    assert env.result().progress == 1 / 3


def test_repair_chain_full_state_progress_stays_below_one_until_submit():
    task = generate_repair_chain_task(seed=43, n_objects=1, n_repair_steps=1)
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

    before_submit = env.result()
    assert before_submit.solved is False
    assert before_submit.progress == 0.5
    assert before_submit.reward == 0.0

    env.submit({"repaired": [object_id]})

    after_submit = env.result()
    assert after_submit.solved is True
    assert after_submit.progress == 1.0
    assert after_submit.reward == after_submit.reference_calls / max(
        after_submit.calls, after_submit.reference_calls
    )


def test_noisy_hidden_dag_wrong_values_do_not_count_as_progress():
    task = generate_hidden_dag_task(
        seed=44,
        n_core=2,
        depth=2,
        invalid_return_mode="noisy",
    )
    env = HiddenDagEnv(task)
    second = task.hidden_state["functions"][1]

    env.call(second["name"], **{second["inputs"][0]: 1})

    result = env.result()
    assert result.solved is False
    assert result.progress == 0.0
    assert result.reward == 0.0
    assert result.failed_calls == 1


def test_summarize_reports_lean_reward_and_failure_rates():
    rows = [
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "solved": True,
            "reward": 1.0,
            "calls": 2,
            "budget": 5,
            "reference_calls": 2,
            "progress": 1.0,
            "failed_calls": 0,
            "failure_counter": {},
        },
        {
            "family": "hidden_dag",
            "mode": "ptc",
            "model": "m",
            "solved": False,
            "reward": 0.0,
            "calls": 4,
            "budget": 4,
            "reference_calls": 2,
            "progress": 0.25,
            "failed_calls": 1,
            "failure_counter": {"timeout": 1},
        },
    ]

    summary = summarize(rows)[0]

    assert summary["solve_rate"] == 0.5
    assert summary["avg_reward"] == 0.5
    assert summary["avg_progress"] == 0.625
    assert summary["avg_calls"] == 3.0
    assert summary["avg_failed_calls"] == 0.5
    assert summary["failed_call_rate"] == 1 / 6
    assert summary["timeout_rate"] == 0.5
