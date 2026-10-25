from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import verifiers as vf
from pydantic import Field
from verifiers.types import Tool

from statecraft.core.runtime import env_for_task, generate_tasks, task_from_dict
from statecraft.core.types import FailureCode, JsonDict, StatecraftTask

SESSION_KEY = "statecraft_session"
RESULT_KEY = "statecraft_result"
TASK_INFO_KEY = "statecraft_task"

STANDARD_SYSTEM_PROMPT = (
    "You are solving a Statecraft task with native tools. Use the provided "
    "primitive tools to discover hidden task state, then call submit with the "
    "final answer. Tool observations are authoritative."
)


class StatecraftTasksetConfig(vf.TasksetConfig):
    family: str | None = "hidden_dag"
    families: list[str] | None = None
    seed: int = 0
    max_examples: int | None = None
    shuffle: bool = False
    task_params: dict[str, Any] | None = Field(default=None)


class StatecraftHarnessConfig(vf.HarnessConfig):
    max_turns: int | None = None


class StatecraftEnvConfig(vf.EnvConfig):
    taskset: StatecraftTasksetConfig = Field(default_factory=StatecraftTasksetConfig)
    harness: StatecraftHarnessConfig = Field(default_factory=StatecraftHarnessConfig)


class StatecraftTool:
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
            description=(
                "Submit final answer: use value for hidden_dag, repaired for repair_chain, "
                "and None for the unused field."
            ),
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


async def setup_statecraft_state(task: vf.Task, state: vf.State) -> None:
    statecraft_task = statecraft_task_from_vf_task(task)
    env = env_for_task(statecraft_task)
    persist_session(state, env, mode="standard")


async def statecraft_submitted(state: vf.State) -> bool:
    result = state.get(RESULT_KEY)
    return bool(isinstance(result, Mapping) and result.get("answer") is not None)


async def statecraft_reward(state: vf.State) -> float:
    return result_number(state, "reward")


async def statecraft_solved(state: vf.State) -> float:
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    return 1.0 if result.get("solved") else 0.0


async def statecraft_calls(state: vf.State) -> float:
    return result_number(state, "calls")


async def statecraft_budget(state: vf.State) -> float:
    return result_number(state, "budget")


async def statecraft_reference_calls(state: vf.State) -> float:
    return result_number(state, "reference_calls")


async def statecraft_progress(state: vf.State) -> float:
    return result_number(state, "progress")


async def statecraft_failed_calls(state: vf.State) -> float:
    return result_number(state, "failed_calls")


def failure_metric(failure: FailureCode):
    async def metric(state: vf.State) -> float:
        result = state.get(RESULT_KEY)
        if not isinstance(result, Mapping):
            return 0.0
        counter = result.get("failure_counter")
        if not isinstance(counter, Mapping):
            return 0.0
        return float(counter.get(failure.value, 0))

    metric.__name__ = f"statecraft_failure_{failure.value}"
    return metric


def load_environment(config: StatecraftEnvConfig | Mapping[str, object] | None = None) -> vf.Env:
    env_config = StatecraftEnvConfig.from_config(config)
    rows = task_rows(
        env_config.taskset,
        mode="standard",
        max_turns=env_config.harness.max_turns,
    )
    tools = native_tools_from_rows(rows)
    taskset = vf.Taskset(
        rows,
        taskset_id="standard-statecraft",
        toolsets=[vf.Toolset(tools=tools)],
        setups=[setup_statecraft_state],
        stops=[statecraft_submitted],
        rewards=[statecraft_reward],
        metrics=[
            statecraft_solved,
            statecraft_calls,
            statecraft_budget,
            statecraft_reference_calls,
            statecraft_progress,
            statecraft_failed_calls,
            *[failure_metric(failure) for failure in FailureCode],
        ],
    )
    harness = vf.Harness(max_turns=env_config.harness.max_turns)
    return vf.Env(taskset=taskset, harness=harness)


def task_rows(
    config: StatecraftTasksetConfig,
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
    task: StatecraftTask,
    *,
    mode: str,
    example_id: int,
    max_turns: int | None = None,
) -> JsonDict:
    row = {
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
    }
    if max_turns is not None:
        row["max_turns"] = max_turns
    return row


def native_tools_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[object]:
    schemas_by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task = task_from_dict(row["info"][TASK_INFO_KEY])  # type: ignore[index]
        for schema in task.public_spec.get("tools", []):
            schemas_by_name.setdefault(str(schema["name"]), schema)
    return [StatecraftTool(schema) for schema in schemas_by_name.values()] + [SubmitTool()]


def session_env(state: vf.State):
    task = statecraft_task_from_state(state)
    snapshot = state.get(SESSION_KEY)
    if snapshot is not None and not isinstance(snapshot, Mapping):
        raise TypeError(f"state.{SESSION_KEY} must be a mapping")
    return env_for_task(task, snapshot)


def persist_session(state: vf.State, env, *, mode: str) -> None:
    state[SESSION_KEY] = env.snapshot()
    state[RESULT_KEY] = env.result(mode=mode).to_dict()


def result_number(state: vf.State, key: str) -> float:
    result = state.get(RESULT_KEY)
    if not isinstance(result, Mapping):
        return 0.0
    value = result.get(key, 0)
    return float(value) if isinstance(value, int | float) else 0.0


def statecraft_task_from_state(state: Mapping[str, Any]) -> StatecraftTask:
    task_data = state.get("task", {})
    if not isinstance(task_data, Mapping):
        raise TypeError("state.task must be a mapping")
    return statecraft_task_from_vf_task(task_data)


def statecraft_task_from_vf_task(task: Mapping[str, Any]) -> StatecraftTask:
    info = task.get("info")
    if not isinstance(info, Mapping):
        raise TypeError("task.info is missing Statecraft task data")
    task_data = info.get(TASK_INFO_KEY)
    if not isinstance(task_data, Mapping):
        raise TypeError(f"task.info.{TASK_INFO_KEY} is missing")
    return task_from_dict(task_data)


def standard_user_prompt(task: StatecraftTask) -> str:
    if task.family == "hidden_dag":
        finish = "submit(value=<integer>, repaired=None)"
    elif task.family == "repair_chain":
        finish = "submit(value=None, repaired=repaired_object_ids)"
    else:
        finish = "submit(answer=answer)"
    return (
        "Discover the target answer using the primitive tools.\n"
        f"Finish with {finish}.\n\n"
        f"Public task spec:\n{json.dumps(task.public_spec, sort_keys=True)}"
    )


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


__all__ = [
    "StatecraftTasksetConfig",
    "StatecraftHarnessConfig",
    "StatecraftEnvConfig",
    "StatecraftTool",
    "SubmitTool",
    "load_environment",
]
