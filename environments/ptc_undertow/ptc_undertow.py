from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

import verifiers as vf
from pydantic import Field
from verifiers.types import Tool
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.v1.utils.endpoint_utils import (
    Endpoint,
    cancel_forwarders,
    endpoint_api_key,
    forward_request,
    raise_execution_error,
    raise_finished_forward_errors,
)

from undertow.core.runtime import env_for_task, generate_tasks
from undertow.core.types import FailureCode, JsonDict, UndertowTask

SESSION_KEY = "undertow_session"
RESULT_KEY = "undertow_result"
TASK_ID_KEY = "undertow_task_id"
PUBLIC_SPEC_KEY = "undertow_public_spec"
RPC_TOOL = "__undertow_rpc"
WORKSPACE = "/workspace"

PTC_SYSTEM_PROMPT = (
    "You are in a Programmatic Tool Calling environment. Use the python tool "
    "to build durable workspace cognition around primitive hidden-state tools. "
    "Create wrappers, caches, solvers, local verifiers, planners, logs, and "
    "experiments in files when useful. Python variables do not persist across "
    "calls; files under /workspace do. Hidden state and reward are protected. "
    "Local verifiers are useful checks but are not reward authorities. Finish "
    "by running finish(answer) from the ptc package."
)


class UndertowTasksetConfig(vf.TasksetConfig):
    family: str | None = "hidden_dag"
    families: list[str] | None = None
    seed: int = 0
    max_examples: int | None = None
    shuffle: bool = False
    task_params: dict[str, Any] | None = Field(default=None)


class UndertowHarnessConfig(vf.HarnessConfig):
    max_turns: int | None = None


class UndertowEnvConfig(vf.EnvConfig):
    taskset: UndertowTasksetConfig = Field(default_factory=UndertowTasksetConfig)
    harness: UndertowHarnessConfig = Field(default_factory=UndertowHarnessConfig)


async def python(code: str, sandbox: object, state: vf.State) -> str:
    """Execute Python code in /workspace. Files persist; variables do not."""
    state["ptc_python_executions"] = int(state.get("ptc_python_executions", 0)) + 1
    try:
        await ensure_workspace(sandbox, state)
        execution_id = uuid.uuid4().hex
        remote_path = f"{WORKSPACE}/.ptc/executions/{execution_id}.py"
        await sandbox.upload_bytes(remote_path, code.encode(), filename=f"{execution_id}.py")
        result = await sandbox.execute(
            f"python {remote_path}",
            working_dir=WORKSPACE,
            env={"PYTHONPATH": WORKSPACE},
        )
    except Exception as exc:
        state["ptc_workspace_setup_failures"] = int(
            state.get("ptc_workspace_setup_failures", 0)
        ) + 1
        return compact_json({"ok": False, "error": exc.__class__.__name__, "detail": str(exc)})

    payload: JsonDict = {"ok": result.exit_code == 0, "exit_code": result.exit_code}
    if result.stdout:
        payload["stdout"] = truncate_output(result.stdout, 12000)
    if result.stderr:
        payload["stderr"] = truncate_output(result.stderr, 4000)
    return compact_json(payload)


python.tool_def = Tool(
    name="python",
    description="Execute Python code in /workspace. Files persist across calls.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    strict=True,
)


class PtcHarness(vf.Harness):
    def __init__(
        self,
        *,
        undertow_tasks: Sequence[UndertowTask],
        max_turns: int | None = None,
        sandbox: Mapping[str, object] | None = None,
    ):
        self._undertow_tasks = {task.task_id: task for task in undertow_tasks}
        self._sessions: dict[str, Any] = {}
        super().__init__(
            program={"base": True, "sandbox": True},
            sandbox={
                "workdir": WORKSPACE,
                "scope": "rollout",
                "command_timeout": 600,
                **dict(sandbox or {}),
            },
            max_turns=max_turns,
            toolsets=[vf.Toolset(tools=[python], sandbox="program", write=True)],
            stops=[ptc_submitted],
            rewards=[ptc_reward],
            metrics=[
                ptc_success,
                ptc_num_calls,
                ptc_python_executions,
                ptc_budget,
                ptc_remaining_budget,
                ptc_budget_used_ratio,
                ptc_progress_score,
                ptc_useful_calls,
                ptc_wasted_calls,
                *[failure_metric(failure) for failure in FailureCode],
            ],
            cleanups=[self.cleanup_session],
        )

    async def setup_state(self, task: vf.Task, state: vf.State) -> vf.State:
        state = await super().setup_state(task, state)
        undertow_task = self.undertow_task_for_vf_task(task)
        env = env_for_task(undertow_task)
        self._sessions[str(state["trajectory_id"])] = env
        persist_session(state, env)
        return state

    async def run_program(self, task: vf.Task, state: vf.State) -> vf.State:
        endpoint = self.resolved_endpoint(state)
        result = await run_ptc_intercepted_program(
            self._program,
            endpoint,
            self.runtime,
            task,
            state,
            self.handle_tool_request,
        )
        if isinstance(result, vf.State):
            state = result
        elif isinstance(result, Mapping):
            state.update(dict(result))
        elif result is not None:
            raise TypeError("Harness program must return None, State, or a mapping.")
        self.refresh_state_from_session(task, state)
        return state

    async def handle_tool_request(
        self,
        name: str,
        arguments: Mapping[str, object],
        task: vf.Task,
        state: vf.State,
    ) -> object:
        if name == RPC_TOOL:
            return await self.handle_rpc(task, state, arguments)
        return await self.runtime.call_tool(name, task, state, **dict(arguments))

    async def handle_rpc(
        self,
        task: vf.Task,
        state: vf.State,
        arguments: Mapping[str, object],
    ) -> JsonDict:
        state["ptc_rpc_calls"] = int(state.get("ptc_rpc_calls", 0)) + 1
        try:
            operation = str(arguments.get("operation", ""))
            env = self.session_for(task, state)
            if operation == "list_tools":
                return {"ok": True, "tools": [tool["name"] for tool in env.tools()]}
            if operation == "tool_schema":
                name = require_string(arguments, "name")
                return {"ok": True, "schema": self.tool_schema(env, name)}
            if operation == "call_tool":
                name = require_string(arguments, "name")
                tool_args = arguments.get("arguments") or {}
                if not isinstance(tool_args, Mapping):
                    raise TypeError("arguments must be an object")
                observation = env.call(name, **dict(tool_args))
                persist_session(state, env)
                return {"ok": True, "observation": observation}
            if operation == "finish":
                observation = env.submit(arguments.get("answer"))
                persist_session(state, env)
                state.stop("submitted")
                return {"ok": True, "observation": observation}
            if operation == "status":
                persist_session(state, env)
                result = cast(Mapping[str, Any], state[RESULT_KEY])
                return {
                    "ok": True,
                    "submitted": bool(env.submitted),
                    "remaining_budget": env.remaining_budget(),
                    "success": bool(result.get("success")),
                }
            raise ValueError(f"unknown RPC operation {operation!r}")
        except Exception as exc:
            state["ptc_rpc_failures"] = int(state.get("ptc_rpc_failures", 0)) + 1
            return {"ok": False, "error": exc.__class__.__name__, "detail": str(exc)}

    def undertow_task_for_vf_task(self, task: Mapping[str, Any]) -> UndertowTask:
        task_id = task.get(TASK_ID_KEY)
        if not isinstance(task_id, str):
            info = task.get("info")
            if isinstance(info, Mapping):
                task_id = cast(str | None, info.get(TASK_ID_KEY))
        if not isinstance(task_id, str) or task_id not in self._undertow_tasks:
            raise KeyError(f"unknown Undertow task id {task_id!r}")
        return self._undertow_tasks[task_id]

    def session_for(self, task: vf.Task, state: vf.State):
        key = str(state["trajectory_id"])
        env = self._sessions.get(key)
        if env is None:
            env = env_for_task(
                self.undertow_task_for_vf_task(task),
                cast(Mapping[str, Any] | None, state.get(SESSION_KEY)),
            )
            self._sessions[key] = env
        return env

    def refresh_state_from_session(self, task: vf.Task, state: vf.State) -> None:
        env = self.session_for(task, state)
        persist_session(state, env)
        if env.submitted:
            state.stop("submitted")

    def tool_schema(self, env, name: str) -> JsonDict:
        for schema in env.tools():
            if schema.get("name") == name:
                return dict(schema)
        raise KeyError(f"unknown tool {name!r}")

    async def cleanup_session(
        self, state: vf.State | None = None, task: vf.Task | None = None
    ) -> None:
        _ = task
        if state is None:
            return
        self._sessions.pop(str(state.get("trajectory_id")), None)


async def run_ptc_intercepted_program(
    program,
    endpoint: Endpoint,
    runtime,
    task: vf.Task,
    state: vf.State,
    tool_handler,
) -> object:
    async def call_tool(name: str, arguments: Mapping[str, object]) -> object:
        return await tool_handler(name, dict(arguments), task, state)

    async def call_user(transcript: list[object]) -> object:
        return await runtime.user_messages(task, state, transcript=transcript)

    async def check_stop() -> object:
        return {
            "done": await runtime.is_completed(task, state),
            "stop_condition": state.get("stop_condition"),
        }

    await endpoint.register_rollout(
        state,
        tool_handler=call_tool,
        tool_defs=runtime.tool_defs(state),
        user_handler=call_user,
        stop_handler=check_stop,
    )
    execution = asyncio.create_task(maybe_call_with_named_args(program, task=task, state=state))
    rollout_key = str(state["endpoint_rollout_key"])
    queue = endpoint.rollout_queue(rollout_key)
    pending: set[asyncio.Task[None]] = set()
    try:
        while True:
            await raise_finished_forward_errors(pending)
            if execution.done():
                await raise_execution_error(execution)
                if not queue.empty():
                    request_id = queue.get_nowait()
                    pending.add(
                        asyncio.create_task(forward_request(endpoint, runtime, task, state, request_id))
                    )
                    continue
                if not pending:
                    break
                await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
                await endpoint.check_tunnel()
                continue
            queue_task = asyncio.create_task(queue.get())
            wait_set = {queue_task, execution, *pending}
            try:
                done, _ = await asyncio.wait(
                    wait_set,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    request_id = queue_task.result()
                    pending.add(
                        asyncio.create_task(forward_request(endpoint, runtime, task, state, request_id))
                    )
                    continue
                if execution in done or pending.intersection(done):
                    continue
                await endpoint.check_tunnel()
            finally:
                if not queue_task.done():
                    queue_task.cancel()
                    await asyncio.gather(queue_task, return_exceptions=True)
            if execution.done() and queue.empty() and not pending:
                break
        await raise_finished_forward_errors(pending)
        return await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await cancel_forwarders(pending)
        endpoint.unregister_rollout(rollout_key)


async def ptc_submitted(
    state: vf.State | None = None, task: vf.Task | None = None
) -> bool:
    _ = task
    if state is None:
        return False
    result = state.get(RESULT_KEY)
    return bool(isinstance(result, Mapping) and result.get("answer") is not None)


async def ptc_reward(
    state: vf.State | None = None, task: vf.Task | None = None
) -> float:
    _ = task
    if state is None:
        return 0.0
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    return 1.0 if result.get("success") else 0.0


async def ptc_success(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    return await ptc_reward(state=state, task=task)


async def ptc_num_calls(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    _ = task
    return result_number(state, "num_calls")


async def ptc_budget(task: vf.Task | None = None, state: vf.State | None = None) -> float:
    _ = task
    return result_number(state, "budget")


async def ptc_remaining_budget(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    _ = task
    return result_number(state, "remaining_budget")


async def ptc_budget_used_ratio(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    _ = task
    return result_number(state, "budget_used_ratio")


async def ptc_progress_score(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    _ = task
    return result_number(state, "progress_score")


async def ptc_useful_calls(task: vf.Task | None = None, state: vf.State | None = None) -> float:
    _ = task
    return result_number(state, "useful_calls")


async def ptc_wasted_calls(task: vf.Task | None = None, state: vf.State | None = None) -> float:
    _ = task
    return result_number(state, "wasted_calls")


def failure_metric(failure: FailureCode):
    async def metric(task: vf.Task | None = None, state: vf.State | None = None) -> float:
        _ = task
        if state is None:
            return 0.0
        result = state.get(RESULT_KEY)
        if not isinstance(result, Mapping):
            return 0.0
        counter = result.get("failure_counter")
        if not isinstance(counter, Mapping):
            return 0.0
        return float(counter.get(failure.value, 0))

    metric.__name__ = f"ptc_failure_{failure.value}"
    return metric


async def ptc_python_executions(
    task: vf.Task | None = None, state: vf.State | None = None
) -> float:
    _ = task
    if state is None:
        return 0.0
    return float(state.get("ptc_python_executions", 0))


def load_environment(config: UndertowEnvConfig | Mapping[str, object] | None = None) -> vf.Env:
    env_config = UndertowEnvConfig.from_config(config)
    tasks = generate_tasks(
        family=env_config.taskset.family,
        families=env_config.taskset.families,
        seed=env_config.taskset.seed,
        max_examples=env_config.taskset.max_examples,
        shuffle=env_config.taskset.shuffle,
        task_params=env_config.taskset.task_params,
    )
    rows = [
        task_row(
            task,
            example_id=index,
            max_turns=env_config.harness.max_turns,
        )
        for index, task in enumerate(tasks)
    ]
    taskset = vf.Taskset(rows, taskset_id="ptc-undertow")
    harness = PtcHarness(
        undertow_tasks=tasks,
        max_turns=env_config.harness.max_turns,
        sandbox=sandbox_config_without_none(env_config.harness.sandbox),
    )
    return vf.Env(taskset=taskset, harness=harness)


def task_row(task: UndertowTask, *, example_id: int, max_turns: int | None = None) -> JsonDict:
    public_spec = dict(task.public_spec)
    return {
        "example_id": example_id,
        TASK_ID_KEY: task.task_id,
        PUBLIC_SPEC_KEY: public_spec,
        "prompt": [{"role": "user", "content": ptc_user_prompt(task)}],
        "system_prompt": PTC_SYSTEM_PROMPT,
        "info": {
            TASK_ID_KEY: task.task_id,
            "family": task.family,
            "seed": task.seed,
            PUBLIC_SPEC_KEY: public_spec,
        },
        "max_turns": max_turns if max_turns is not None else task.budget + 4,
    }


def ptc_user_prompt(task: UndertowTask) -> str:
    return (
        "A persistent workspace is mounted at /workspace. The ptc package is "
        "available there after the first python call:\n"
        "  from ptc import task, tools, finish\n\n"
        "Use task.info(), task.spec(), tools.list(), tools.schema(name), "
        "tools.call(name, **kwargs), tools.<tool_name>(**kwargs), and finish(answer).\n"
        "Only public task data is exposed in the workspace and ptc package.\n\n"
        f"Public task spec:\n{json.dumps(task.public_spec, sort_keys=True)}"
    )


async def ensure_workspace(sandbox: object, state: vf.State) -> None:
    if state.get("ptc_workspace_ready"):
        return
    await sandbox.execute(
        "mkdir -p /workspace/cache /workspace/logs /workspace/experiments "
        "/workspace/ptc /workspace/.ptc/executions",
        working_dir=WORKSPACE,
    )
    config = {
        "endpoint_root_url": state["endpoint_root_url"],
        "api_key": endpoint_api_key(state),
    }
    files = workspace_files(
        public_spec=state["task"][PUBLIC_SPEC_KEY],
        task_id=state["task"][TASK_ID_KEY],
        rpc_config=config,
    )
    for path, content in files.items():
        await sandbox.upload_bytes(path, content.encode(), filename=path.rsplit("/", 1)[-1])
    state["ptc_workspace_ready"] = True


def workspace_files(
    *,
    public_spec: Mapping[str, Any],
    task_id: str,
    rpc_config: Mapping[str, str],
) -> dict[str, str]:
    public_json = json.dumps(dict(public_spec), indent=2, sort_keys=True)
    config_json = json.dumps(dict(rpc_config), sort_keys=True)
    return {
        f"{WORKSPACE}/README.md": PTC_README,
        f"{WORKSPACE}/task.md": f"# Undertow task\n\nTask id: `{task_id}`\n\n```json\n{public_json}\n```\n",
        f"{WORKSPACE}/solver.py": SOLVER_TEMPLATE,
        f"{WORKSPACE}/notes.md": "# Notes\n\n",
        f"{WORKSPACE}/.ptc/config.json": config_json,
        f"{WORKSPACE}/.ptc/public_spec.json": public_json,
        f"{WORKSPACE}/ptc/__init__.py": PTC_INIT,
        f"{WORKSPACE}/ptc/_client.py": PTC_CLIENT,
        f"{WORKSPACE}/ptc/task.py": PTC_TASK,
        f"{WORKSPACE}/ptc/tools.py": PTC_TOOLS,
    }


PTC_README = """# Programmatic Tool Calling

Use `from ptc import task, tools, finish`.

The workspace is durable across python calls. Python interpreter variables are not.
Hidden task state and reward are host-side only. Local checks are useful but not authoritative.
"""

SOLVER_TEMPLATE = """from ptc import task, tools, finish


def main():
    print(task.info())
    print(tools.list())


if __name__ == "__main__":
    main()
"""

PTC_INIT = """from . import task, tools
from ._client import finish

__all__ = ["task", "tools", "finish"]
"""

PTC_CLIENT = f'''import json
import urllib.error
import urllib.request
from pathlib import Path

_CONFIG_PATH = Path("{WORKSPACE}/.ptc/config.json")


def _config():
    return json.loads(_CONFIG_PATH.read_text())


def rpc(operation, **kwargs):
    config = _config()
    url = config["endpoint_root_url"].rstrip("/") + "/vf/tools/{RPC_TOOL}"
    body = json.dumps({{"arguments": {{"operation": operation, **kwargs}}}}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={{
            "Authorization": "Bearer " + config["api_key"],
            "Content-Type": "application/json",
            "User-Agent": "undertow-ptc/1.0",
            "Accept": "application/json",
        }},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"PTC RPC failed: {{exc.code}} {{detail}}") from exc
    if "error" in payload:
        raise RuntimeError(payload["error"])
    result = payload.get("result")
    if isinstance(result, dict) and result.get("ok") is False:
        raise RuntimeError(result.get("detail") or result.get("error") or "PTC RPC failed")
    return result


def finish(answer):
    return rpc("finish", answer=answer)
'''

PTC_TASK = f'''import json
from pathlib import Path

_TASK_PATH = Path("{WORKSPACE}/task.md")
_SPEC_PATH = Path("{WORKSPACE}/.ptc/public_spec.json")


def spec():
    if _SPEC_PATH.exists():
        return json.loads(_SPEC_PATH.read_text())
    text = _TASK_PATH.read_text()
    start = text.index("```json") + len("```json")
    end = text.index("```", start)
    return json.loads(text[start:end])


def info():
    data = spec()
    return {{
        "family": data.get("family"),
        "target": data.get("target"),
        "num_tools": len(data.get("tools", [])),
    }}
'''

PTC_TOOLS = """from ._client import rpc


def list():
    return rpc("list_tools")["tools"]


def schema(name):
    return rpc("tool_schema", name=name)["schema"]


def call(name, **kwargs):
    return rpc("call_tool", name=name, arguments=kwargs)["observation"]


def __getattr__(name):
    def invoke(**kwargs):
        return call(name, **kwargs)

    return invoke
"""


def persist_session(state: vf.State, env) -> None:
    state[SESSION_KEY] = env.snapshot()
    state[RESULT_KEY] = env.result(mode="ptc").to_dict()


def result_number(state: vf.State | None, key: str) -> float:
    if state is None:
        return 0.0
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    value = result.get(key, 0)
    return float(value) if isinstance(value, int | float) else 0.0


def require_string(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def sandbox_config_without_none(config: object) -> Mapping[str, object] | None:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        raw = config.model_dump(exclude_none=True)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        return None
    return {key: value for key, value in raw.items() if value is not None}


def truncate_output(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


__all__ = [
    "UndertowTasksetConfig",
    "UndertowHarnessConfig",
    "UndertowEnvConfig",
    "PtcHarness",
    "load_environment",
    "workspace_files",
]
