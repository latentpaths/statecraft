from statecraft.core import FailureCode
from statecraft.core.families.hidden_dag import HiddenDagEnv, generate_hidden_dag_task


def test_hidden_dag_generation_is_deterministic():
    first = generate_hidden_dag_task(seed=7, n_core=6, depth=3, n_connected_distractors=2)
    second = generate_hidden_dag_task(seed=7, n_core=6, depth=3, n_connected_distractors=2)

    assert first.public_spec == second.public_spec
    assert first.hidden_state == second.hidden_state
    assert first.target == second.target


def test_hidden_dag_default_budget_allows_discovery_overhead():
    task = generate_hidden_dag_task(
        seed=8,
        n_core=5,
        depth=2,
        n_connected_distractors=5,
        n_disconnected_distractors=0,
    )

    assert task.budget == 33


def test_hidden_dag_golden_path_succeeds():
    task = generate_hidden_dag_task(seed=3, n_core=5, depth=3)
    env = HiddenDagEnv(task)
    known = dict(task.public_spec["initial_values"])

    for func_name in task.hidden_state["solution"]:
        func = next(f for f in task.hidden_state["functions"] if f["name"] == func_name)
        obs = env.call(func_name, **{name: known[name] for name in func["inputs"]})
        assert obs["ok"] is True
        known[obs["variable"]] = obs["value"]

    assert env.submit({"value": task.target["value"]})["ok"] is True
    assert env.result().solved is True


def test_hidden_dag_unknown_input_fails_cleanly():
    task = generate_hidden_dag_task(seed=4, n_core=4, depth=4)
    env = HiddenDagEnv(task)
    second = task.hidden_state["functions"][1]

    obs = env.call(second["name"], **{second["inputs"][0]: 123})

    assert obs["ok"] is False
    assert env.trace()[-1]["failure"] == FailureCode.UNKNOWN_STATE_REFERENCE


def test_hidden_dag_connected_distractor_is_callable_but_classified():
    task = generate_hidden_dag_task(seed=5, n_core=3, depth=2, n_connected_distractors=1)
    env = HiddenDagEnv(task)
    known = dict(task.public_spec["initial_values"])
    distractor = next(f for f in task.hidden_state["functions"] if f["node_type"] == "connected_distractor")
    for func_name in task.hidden_state["solution"]:
        if distractor["inputs"][0] in known:
            break
        func = next(f for f in task.hidden_state["functions"] if f["name"] == func_name)
        if all(name in known for name in func["inputs"]):
            obs = env.call(func_name, **{name: known[name] for name in func["inputs"]})
            known[obs["variable"]] = obs["value"]

    obs = env.call(distractor["name"], **{distractor["inputs"][0]: known[distractor["inputs"][0]]})

    assert obs["ok"] is True
    assert env.trace()[-1]["failure"] == FailureCode.DISTRACTOR_ACTION


def test_hidden_dag_reminder_mode_includes_known_values():
    task = generate_hidden_dag_task(seed=6, n_core=2, depth=2, observation_mode="reminder")
    env = HiddenDagEnv(task)
    first = task.hidden_state["functions"][0]
    initial = task.public_spec["initial_values"][first["inputs"][0]]

    obs = env.call(first["name"], **{first["inputs"][0]: initial})

    assert "known_values" in obs
    assert first["output"] in obs["known_values"]


def test_hidden_dag_minimal_mode_omits_known_values():
    task = generate_hidden_dag_task(seed=6, n_core=2, depth=2, observation_mode="minimal")
    env = HiddenDagEnv(task)
    first = task.hidden_state["functions"][0]
    initial = task.public_spec["initial_values"][first["inputs"][0]]

    obs = env.call(first["name"], **{first["inputs"][0]: initial})

    assert "known_values" not in obs
