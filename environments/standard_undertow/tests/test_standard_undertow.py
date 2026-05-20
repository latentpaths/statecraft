import asyncio
import json

import verifiers as vf

import standard_undertow
from undertow.core.runtime import task_from_dict


def load_small_env(config=None):
    return standard_undertow.load_environment(
        config
        or {
            "taskset": {
                "family": "hidden_dag",
                "max_examples": 1,
                "task_params": {"n_core": 1, "depth": 1},
            }
        }
    )


async def setup_state(env):
    task = env.taskset.to_task(env.taskset.rows()[0])
    state = vf.State.for_task(task)
    return task, await env.harness.setup_state(task, state)


def test_load_environment_uses_verifiers_v1_env():
    env = load_small_env()

    assert isinstance(env, vf.Env)
    assert isinstance(env.harness, vf.Harness)


def test_native_tools_plus_submit_are_model_visible():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))

    tool_names = [tool.name for tool in env.harness.runtime.tool_defs(state) or []]
    submit_schema = next(
        tool.parameters
        for tool in env.harness.runtime.tool_defs(state) or []
        if tool.name == "submit"
    )

    assert "submit" in tool_names
    assert "core_0" in tool_names
    assert task["max_turns"] == task_from_dict(task["info"]["undertow_task"]).budget + 4
    assert submit_schema["properties"]["value"]["type"] == ["integer", "null"]
    assert submit_schema["properties"]["repaired"]["type"] == ["array", "null"]
    assert set(submit_schema["required"]) == {"value", "repaired"}


def test_default_and_configured_max_turns():
    env = load_small_env()
    task = task_from_dict(env.taskset.rows()[0]["info"]["undertow_task"])

    assert env.taskset.rows()[0]["max_turns"] == task.budget + 4

    overridden = load_small_env(
        {
            "taskset": {
                "family": "hidden_dag",
                "max_examples": 1,
                "task_params": {"n_core": 1, "depth": 1},
            },
            "harness": {"max_turns": 7},
        }
    )

    assert overridden.taskset.rows()[0]["max_turns"] == 7


def test_native_tool_updates_serialized_result_without_trace_delta():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))
    undertow_task = task_from_dict(task["info"]["undertow_task"])
    func = undertow_task.hidden_state["functions"][0]
    value = undertow_task.public_spec["initial_values"][func["inputs"][0]]

    output = asyncio.run(
        env.harness.runtime.call_tool(
            func["name"],
            task,
            state,
            **{func["inputs"][0]: value},
        )
    )
    parsed = json.loads(output)

    assert parsed["ok"] is True
    assert "trace_delta" not in parsed
    assert state["undertow_result"]["num_calls"] == 1
    assert state["undertow_result"]["trace"][-1]["tool_name"] == func["name"]


def test_native_tool_restores_hidden_dag_state_between_calls():
    env = load_small_env(
        {
            "taskset": {
                "family": "hidden_dag",
                "max_examples": 1,
                "task_params": {"n_core": 2, "depth": 2},
            }
        }
    )
    task, state = asyncio.run(setup_state(env))
    undertow_task = task_from_dict(task["info"]["undertow_task"])
    first, second = undertow_task.hidden_state["functions"][:2]
    first_input = first["inputs"][0]

    asyncio.run(
        env.harness.runtime.call_tool(
            first["name"],
            task,
            state,
            **{first_input: undertow_task.public_spec["initial_values"][first_input]},
        )
    )
    output = asyncio.run(
        env.harness.runtime.call_tool(
            second["name"],
            task,
            state,
            **{second["inputs"][0]: undertow_task.hidden_state["values"][second["inputs"][0]]},
        )
    )
    parsed = json.loads(output)

    assert parsed["ok"] is True
    assert parsed["variable"] == second["output"]


def test_native_tool_restores_repair_chain_location_between_calls():
    env = load_small_env(
        {
            "taskset": {
                "family": "repair_chain",
                "max_examples": 1,
                "task_params": {
                    "n_objects": 1,
                    "n_repair_steps": 1,
                    "n_tools": 2,
                    "n_distractor_tools": 0,
                },
            }
        }
    )
    task, state = asyncio.run(setup_state(env))
    undertow_task = task_from_dict(task["info"]["undertow_task"])
    object_id, obj = next(iter(undertow_task.hidden_state["objects"].items()))

    scan_output = asyncio.run(
        env.harness.runtime.call_tool("scan_object", task, state, object_id=object_id)
    )
    scan = json.loads(scan_output)
    asyncio.run(env.harness.runtime.call_tool("move_to", task, state, location=obj["location"]))
    output = asyncio.run(
        env.harness.runtime.call_tool(
            "apply_tool",
            task,
            state,
            object_id=object_id,
            tool_id=obj["required_tools"][0],
            observation_version=scan["observation_version"],
        )
    )
    parsed = json.loads(output)

    assert parsed["ok"] is True
    assert parsed["status"] == "repaired"


def test_submit_stops_and_metrics_read_serialized_result():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))
    undertow_task = task_from_dict(task["info"]["undertow_task"])

    output = asyncio.run(
        env.harness.runtime.call_tool(
            "submit",
            task,
            state,
            answer={"value": undertow_task.target["value"]},
        )
    )
    parsed = json.loads(output)

    assert parsed["ok"] is True
    assert state["stop_condition"] == "submitted"
    assert asyncio.run(standard_undertow.undertow_num_calls(state)) == 1.0
    assert asyncio.run(standard_undertow.undertow_reward(state)) == 1.0
