from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from undertow.core.env import UndertowEnv
from undertow.core.families.hidden_dag import HiddenDagEnv, generate_hidden_dag_task
from undertow.core.families.repair_chain import RepairChainEnv, generate_repair_chain_task
from undertow.core.types import EpisodeResult, UndertowTask

InteractionMode = Literal["standard", "ptc"]
TaskFamily = Literal["hidden_dag", "repair_chain"]

TASK_FAMILIES: tuple[TaskFamily, ...] = ("hidden_dag", "repair_chain")
INTERACTION_MODES: tuple[InteractionMode, ...] = ("standard", "ptc")


def generate_task(
    family: str,
    *,
    seed: int,
    params: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> UndertowTask:
    task_params = {**dict(params or {}), **kwargs}
    if family == "hidden_dag":
        return generate_hidden_dag_task(seed=seed, **task_params)
    if family == "repair_chain":
        return generate_repair_chain_task(seed=seed, **task_params)
    raise ValueError(f"unknown family {family!r}; expected one of {list(TASK_FAMILIES)}")


def generate_tasks(
    *,
    family: str | None = None,
    families: Sequence[str] | None = None,
    seed: int = 0,
    max_examples: int | None = None,
    shuffle: bool = False,
    task_params: Mapping[str, Any] | Mapping[str, Mapping[str, Any]] | None = None,
) -> list[UndertowTask]:
    selected = _selected_families(family=family, families=families)
    total = 1 if max_examples is None else max_examples
    if total < 1:
        raise ValueError("max_examples must be >= 1")

    tasks = []
    for index in range(total):
        task_family = selected[index % len(selected)]
        tasks.append(
            generate_task(
                task_family,
                seed=seed + index,
                params=_params_for_family(task_params, task_family),
            )
        )
    if shuffle:
        random.Random(seed).shuffle(tasks)
    return tasks


def task_from_dict(data: Mapping[str, Any]) -> UndertowTask:
    return UndertowTask(
        task_id=str(data["task_id"]),
        family=str(data["family"]),
        seed=int(data["seed"]),
        params=dict(data["params"]),
        public_spec=dict(data["public_spec"]),
        hidden_state=dict(data["hidden_state"]),
        target=dict(data["target"]),
        budget=int(data["budget"]),
    )


def env_for_task(
    task: UndertowTask,
    snapshot: Mapping[str, Any] | None = None,
) -> UndertowEnv:
    if task.family == "hidden_dag":
        env = HiddenDagEnv(task)
    elif task.family == "repair_chain":
        env = RepairChainEnv(task)
    else:
        raise ValueError(f"unknown family {task.family!r}")
    env.restore(snapshot)
    return env


def run_oracle(env: UndertowEnv) -> EpisodeResult:
    if env.task.family == "hidden_dag":
        values = dict(env.task.public_spec["initial_values"])
        for func_name in env.task.hidden_state["solution"]:
            func = next(f for f in env.task.hidden_state["functions"] if f["name"] == func_name)
            args = {name: values[name] for name in func["inputs"]}
            obs = env.call(func_name, **args)
            values[obs["variable"]] = obs["value"]
        env.submit({"value": env.task.target["value"]})
        return env.result(mode="oracle")

    if env.task.family == "repair_chain":
        for object_id, obj in env.task.hidden_state["objects"].items():
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
                version = obs["observation_version"]
            env.call("test_object", object_id=object_id, observation_version=version)
        env.submit({"repaired": list(env.task.hidden_state["objects"])})
        return env.result(mode="oracle")

    raise ValueError(f"unknown family {env.task.family!r}")


def _selected_families(
    *, family: str | None = None, families: Sequence[str] | None = None
) -> tuple[str, ...]:
    if family and families:
        raise ValueError("pass either family or families, not both")
    selected = tuple(families or ((family,) if family else TASK_FAMILIES))
    if not selected:
        raise ValueError("at least one family is required")
    unknown = sorted(set(selected) - set(TASK_FAMILIES))
    if unknown:
        raise ValueError(f"unknown families {unknown}; expected one of {list(TASK_FAMILIES)}")
    return selected


def _params_for_family(
    task_params: Mapping[str, Any] | Mapping[str, Mapping[str, Any]] | None,
    family: str,
) -> Mapping[str, Any]:
    if not task_params:
        return {}
    family_params = task_params.get(family) if isinstance(task_params, Mapping) else None
    if isinstance(family_params, Mapping):
        return family_params
    if any(key in TASK_FAMILIES and isinstance(value, Mapping) for key, value in task_params.items()):
        return {}
    return task_params  # type: ignore[return-value]
