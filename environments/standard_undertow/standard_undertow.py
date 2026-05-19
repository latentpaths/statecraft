from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import verifiers as vf
from pydantic import Field
from verifiers.types import Tool

from undertow.core.runtime import env_for_task, generate_tasks, task_from_dict
from undertow.core.types import JsonDict, UndertowTask

SESSION_KEY = "undertow_session"
RESULT_KEY = "undertow_result"
TASK_INFO_KEY = "undertow_task"

STANDARD_SYSTEM_PROMPT = (
    "You are solving an Undertow task with native tools. Use the provided "
    "primitive tools to discover hidden task state, then call submit with the "
    "final answer. Tool observations are authoritative."
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


class UndertowTool:
    def __init__(self, schema: Mapping[str, Any]):
        self.name = str(schema["name"])
        self._tool_def = Tool(
            name=self.name,
            description=str(schema.get("description", "")),
            parameters=dict(schema.get("parameters") or {}),
            strict=bool(schema.get("strict", False)),
        )

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def tool_def(self) -> Tool:
        return self._tool_def

    async def __call__(self, state: vf.State, **kwargs: object) -> str:
        env = session_env(state)
        observation = env.call(self.name, **dict(kwargs))
        persist_session(state, env, mode="standard")
        return compact_json(observation)


class SubmitTool:
    name: ClassVar[str] = "submit"

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def tool_def(self) -> Tool:
        return Tool(
            name=self.name,
            description="Submit the final answer for scoring.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["integer", "null"],
                        "description": "Final numeric target value for hidden_dag tasks.",
                    },
                    "repaired": {
                        "type": ["array", "null"],
                        "description": "Object ids repaired for repair_chain tasks.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["value", "repaired"],
                "additionalProperties": False,
            },
            strict=True,
        )

    async def __call__(
        self,
        state: vf.State,
        answer: object | None = None,
        value: int | None = None,
        repaired: list[str] | None = None,
    ) -> str:
        if answer is None:
            answer = {"value": value} if value is not None else {"repaired": repaired or []}
        env = session_env(state)
        observation = env.submit(answer)
        persist_session(state, env, mode="standard")
        state.stop("submitted")
        return compact_json(observation)


async def setup_undertow_state(task: vf.Task, state: vf.State) -> None:
    undertow_task = undertow_task_from_vf_task(task)
    env = env_for_task(undertow_task)
    persist_session(state, env, mode="standard")


async def undertow_submitted(state: vf.State) -> bool:
    result = state.get(RESULT_KEY)
    return bool(isinstance(result, Mapping) and result.get("answer") is not None)


async def undertow_reward(state: vf.State) -> float:
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    return 1.0 if result.get("success") else 0.0


async def undertow_success(state: vf.State) -> float:
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    return 1.0 if result.get("success") else 0.0


async def undertow_num_calls(state: vf.State) -> float:
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    return float(result.get("num_calls", 0))


def load_environment(config: UndertowEnvConfig | Mapping[str, object] | None = None) -> vf.Env:
    env_config = UndertowEnvConfig.from_config(config)
    rows = task_rows(
        env_config.taskset,
        mode="standard",
        max_turns=env_config.harness.max_turns,
    )
    tools = native_tools_from_rows(rows)
    taskset = vf.Taskset(
        rows,
        taskset_id="standard-undertow",
        toolsets=[vf.Toolset(tools=tools)],
        setups=[setup_undertow_state],
        stops=[undertow_submitted],
        rewards=[undertow_reward],
        metrics=[undertow_success, undertow_num_calls],
    )
    harness = vf.Harness(max_turns=env_config.harness.max_turns)
    return vf.Env(taskset=taskset, harness=harness)


def task_rows(
    config: UndertowTasksetConfig,
    *,
    mode: str,
    max_turns: int | None = None,
) -> list[JsonDict]:
    tasks = generate_tasks(
        family=config.family,
        families=config.families,
        seed=config.seed,
        max_examples=config.max_examples,
        shuffle=config.shuffle,
        task_params=config.task_params,
    )
    return [
        task_row(task, mode=mode, example_id=index, max_turns=max_turns)
        for index, task in enumerate(tasks)
    ]


def task_row(
    task: UndertowTask,
    *,
    mode: str,
    example_id: int,
    max_turns: int | None = None,
) -> JsonDict:
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": standard_user_prompt(task)}],
        "system_prompt": STANDARD_SYSTEM_PROMPT,
        "answer": task.target,
        "info": {
            "task_id": task.task_id,
            "family": task.family,
            "seed": task.seed,
            "target": task.target,
            TASK_INFO_KEY: task.to_dict(),
        },
        "max_turns": max_turns if max_turns is not None else task.budget + 4,
    }


def native_tools_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[object]:
    schemas_by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task = task_from_dict(row["info"][TASK_INFO_KEY])  # type: ignore[index]
        for schema in task.public_spec.get("tools", []):
            schemas_by_name.setdefault(str(schema["name"]), schema)
    return [UndertowTool(schema) for schema in schemas_by_name.values()] + [SubmitTool()]


def session_env(state: vf.State):
    task = undertow_task_from_state(state)
    snapshot = state.get(SESSION_KEY)
    if snapshot is not None and not isinstance(snapshot, Mapping):
        raise TypeError(f"state.{SESSION_KEY} must be a mapping")
    return env_for_task(task, snapshot)


def persist_session(state: vf.State, env, *, mode: str) -> None:
    state[SESSION_KEY] = env.snapshot()
    state[RESULT_KEY] = env.result(mode=mode).to_dict()


def undertow_task_from_state(state: Mapping[str, Any]) -> UndertowTask:
    task_data = state.get("task", {})
    if not isinstance(task_data, Mapping):
        raise TypeError("state.task must be a mapping")
    return undertow_task_from_vf_task(task_data)


def undertow_task_from_vf_task(task: Mapping[str, Any]) -> UndertowTask:
    info = task.get("info")
    if not isinstance(info, Mapping):
        raise TypeError("task.info is missing Undertow task data")
    task_data = info.get(TASK_INFO_KEY)
    if not isinstance(task_data, Mapping):
        raise TypeError(f"task.info.{TASK_INFO_KEY} is missing")
    return task_from_dict(task_data)


def standard_user_prompt(task: UndertowTask) -> str:
    return (
        "Discover the target answer using the primitive tools, then submit it.\n\n"
        f"Public task spec:\n{json.dumps(task.public_spec, sort_keys=True)}"
    )


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


__all__ = [
    "UndertowTasksetConfig",
    "UndertowHarnessConfig",
    "UndertowEnvConfig",
    "UndertowTool",
    "SubmitTool",
    "load_environment",
]
