from statecraft.core import FailureCode
from statecraft.core.families.repair_chain import RepairChainEnv, generate_repair_chain_task


def test_repair_chain_generation_is_deterministic():
    first = generate_repair_chain_task(seed=11, n_objects=2, n_repair_steps=2)
    second = generate_repair_chain_task(seed=11, n_objects=2, n_repair_steps=2)

    assert first.public_spec == second.public_spec
    assert first.hidden_state == second.hidden_state
    assert first.target == second.target


def test_repair_chain_default_budget_allows_discovery_overhead():
    task = generate_repair_chain_task(
        seed=10,
        n_objects=3,
        n_repair_steps=3,
        n_tools=6,
        n_distractor_tools=2,
    )
    public_solver_calls = 8 + 3 + 3 + 9 + 3 + 1

    assert task.budget == public_solver_calls * 3


def test_repair_chain_golden_path_succeeds():
    task = generate_repair_chain_task(seed=12, n_objects=2, n_repair_steps=2)
    env = RepairChainEnv(task)

    for object_id, obj in task.hidden_state["objects"].items():
        scan = env.call("scan_object", object_id=object_id)
        env.call("move_to", location=obj["location"])
        version = scan["observation_version"]
        for tool_id in obj["required_tools"]:
            obs = env.call(
                "apply_tool",
                object_id=object_id,
                tool_id=tool_id,
                observation_version=version,
            )
            assert obs["ok"] is True
            version = obs["observation_version"]
        assert env.call("test_object", object_id=object_id, observation_version=version)["ok"] is True

    assert env.submit({"repaired": list(task.hidden_state["objects"])})["ok"] is True
    assert env.result().solved is True


def test_repair_chain_can_be_solved_from_public_observations_only():
    task = generate_repair_chain_task(seed=16, n_objects=3, n_repair_steps=3)
    env = RepairChainEnv(task)
    spec = task.public_spec

    stage_tools_by_part = {}
    for tool_id in spec["tools_available"]:
        inspected = env.call("inspect_tool", tool_id=tool_id)
        for part, stages in inspected["compatible_stages_by_part"].items():
            for stage in stages:
                stage_tools_by_part[(part, stage)] = tool_id

    for object_id in spec["objects"]:
        scan = env.call("scan_object", object_id=object_id)
        env.call("move_to", location=scan["location"])
        version = scan["observation_version"]
        part = scan["part"]
        for stage in range(scan["repair_stage"], task.params["n_repair_steps"]):
            obs = env.call(
                "apply_tool",
                object_id=object_id,
                tool_id=stage_tools_by_part[(part, stage)],
                observation_version=version,
            )
            assert obs["ok"] is True
            version = obs["observation_version"]
        assert env.call("test_object", object_id=object_id, observation_version=version)["ok"] is True

    assert env.submit({"repaired": spec["objects"]})["ok"] is True
    assert env.result().solved is True


def test_inspect_tool_exposes_stage_planning_information():
    task = generate_repair_chain_task(seed=17, n_objects=2, n_repair_steps=2)
    env = RepairChainEnv(task)
    tool_id = task.hidden_state["objects"]["machine_0"]["required_tools"][0]

    inspected = env.call("inspect_tool", tool_id=tool_id)

    assert inspected["compatible_stages_by_part"]
    assert "compatible_stages_by_part" in inspected


def test_repair_chain_wrong_tool_is_constraint_violation():
    task = generate_repair_chain_task(seed=13, n_objects=1, n_repair_steps=1, n_tools=3)
    env = RepairChainEnv(task)
    object_id, obj = next(iter(task.hidden_state["objects"].items()))
    wrong_tool = next(tool for tool in task.public_spec["tools_available"] if tool != obj["required_tools"][0])

    env.call("move_to", location=obj["location"])
    obs = env.call("apply_tool", object_id=object_id, tool_id=wrong_tool)

    assert obs["ok"] is False
    assert env.trace()[-1]["failure"] in {
        FailureCode.VIOLATED_CONSTRAINT,
        FailureCode.DISTRACTOR_ACTION,
    }


def test_repair_chain_stale_observation_is_classified():
    task = generate_repair_chain_task(seed=14, n_objects=1, n_repair_steps=2, staleness_rate=1.0)
    env = RepairChainEnv(task)
    object_id, obj = next(iter(task.hidden_state["objects"].items()))
    scan = env.call("scan_object", object_id=object_id)
    env.call("move_to", location=obj["location"])
    env.call(
        "apply_tool",
        object_id=object_id,
        tool_id=obj["required_tools"][0],
        observation_version=scan["observation_version"],
    )

    obs = env.call(
        "apply_tool",
        object_id=object_id,
        tool_id=obj["required_tools"][1],
        observation_version=scan["observation_version"],
    )

    assert obs["ok"] is False
    assert env.trace()[-1]["failure"] == FailureCode.STALE_STATE_REFERENCE


def test_repair_chain_premature_submit_fails():
    task = generate_repair_chain_task(seed=15, n_objects=1, n_repair_steps=1)
    env = RepairChainEnv(task)

    obs = env.submit({"repaired": list(task.hidden_state["objects"])})

    assert obs["ok"] is False
    assert env.trace()[-1]["failure"] == FailureCode.PREMATURE_SUBMIT
