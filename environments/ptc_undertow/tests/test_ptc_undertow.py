import asyncio

import verifiers as vf

import ptc_undertow


def load_small_env(config=None):
    return ptc_undertow.load_environment(
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


def test_load_environment_uses_ptc_harness_and_public_rows():
    env = load_small_env()
    row = env.taskset.rows()[0]

    assert isinstance(env, vf.Env)
    assert isinstance(env.harness, ptc_undertow.PtcHarness)
    assert env.harness.sandbox["timeout_minutes"] == ptc_undertow.PTC_TIMEOUT_MINUTES
    assert env.harness.sandbox["command_timeout"] == ptc_undertow.PTC_TIMEOUT_SECONDS
    assert "hidden_state" not in str(row)
    assert "target': {'value'" not in str(row)


def test_only_python_is_model_visible():
    env = load_small_env()
    _task, state = asyncio.run(setup_state(env))

    tool_names = [tool.name for tool in env.harness.runtime.tool_defs(state) or []]

    assert tool_names == ["python"]
    assert ptc_undertow.RPC_TOOL not in tool_names


def test_prompts_show_exact_python_finish_shape():
    hidden_row = load_small_env().taskset.rows()[0]
    repair_row = load_small_env(
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
    ).taskset.rows()[0]

    assert (
        'Finish by running Python code that calls finish({"value": value}).'
        in hidden_row["prompt"][0]["content"]
    )
    assert (
        'Finish by running Python code that calls finish({"repaired": repaired_object_ids}).'
        in repair_row["prompt"][0]["content"]
    )
    assert "hidden_state" not in str(hidden_row)
    assert "hidden_state" not in str(repair_row)


def test_default_and_configured_max_turns():
    env = load_small_env()

    assert "max_turns" not in env.taskset.rows()[0]

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


def test_workspace_files_are_public_and_ergonomic():
    files = ptc_undertow.workspace_files(
        public_spec={"family": "hidden_dag", "tools": [{"name": "core_0"}]},
        task_id="task-1",
        rpc_config={"endpoint_root_url": "http://host/rollout/r", "api_key": "secret"},
    )

    joined = "\n".join(files.values())
    assert "/workspace/README.md" in files
    assert "/workspace/task.md" in files
    assert "/workspace/solver.py" in files
    assert "/workspace/ptc/__init__.py" in files
    assert "from ptc import task, tools, finish" in files["/workspace/solver.py"]
    assert "hidden_state" not in joined
    assert "solve(" not in joined
    assert "BFS" not in joined
    assert '"User-Agent": "undertow-ptc/1.0"' in files["/workspace/ptc/_client.py"]
    assert f"timeout={ptc_undertow.PTC_TIMEOUT_SECONDS}" in files["/workspace/ptc/_client.py"]


def test_private_rpc_routes_undertow_calls_host_side():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))
    harness = env.harness

    listed = asyncio.run(harness.handle_rpc(task, state, {"operation": "list_tools"}))
    first_tool = listed["tools"][0]
    schema = asyncio.run(
        harness.handle_rpc(task, state, {"operation": "tool_schema", "name": first_tool})
    )["schema"]
    args = {
        name: task["undertow_public_spec"]["initial_values"][name]
        for name in schema["parameters"]["required"]
    }
    called = asyncio.run(
        harness.handle_rpc(
            task,
            state,
            {"operation": "call_tool", "name": first_tool, "arguments": args},
        )
    )

    assert called["ok"] is True
    assert called["observation"]["ok"] is True
    assert "trace_delta" not in called
    assert state["undertow_result"]["calls"] == 1
    assert "hidden_state" not in str(state)


def test_finish_updates_host_session_and_stops():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))
    harness = env.harness

    finished = asyncio.run(
        harness.handle_rpc(
            task,
            state,
            {"operation": "finish", "answer": {"value": 0}},
        )
    )

    assert finished["ok"] is True
    assert state["undertow_result"]["answer"] == {"value": 0}
    assert state["stop_condition"] == "submitted"


def test_rpc_failures_are_controlled_state_errors():
    env = load_small_env()
    task, state = asyncio.run(setup_state(env))

    response = asyncio.run(
        env.harness.handle_rpc(task, state, {"operation": "not_real"})
    )

    assert response["ok"] is False
    assert state["ptc_rpc_failures"] == 1
