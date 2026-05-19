from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from undertow.core.runtime import generate_tasks
from undertow.core.types import UndertowTask

InteractionMode = str


def generate_dataset_records(
    *,
    mode: InteractionMode,
    family: str | None = None,
    families: Sequence[str] | None = None,
    seed: int = 0,
    max_examples: int | None = None,
    shuffle: bool = False,
    task_params: Mapping[str, Any] | Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, object]]:
    tasks = generate_tasks(
        family=family,
        families=families,
        seed=seed,
        max_examples=max_examples,
        shuffle=shuffle,
        task_params=task_params,
    )
    return records_for_tasks(tasks, mode=mode)


def records_for_tasks(tasks: Sequence[UndertowTask], *, mode: InteractionMode) -> list[dict[str, object]]:
    return [record_for_task(task, mode=mode, example_id=index) for index, task in enumerate(tasks)]


def record_for_task(task: UndertowTask, *, mode: InteractionMode, example_id: int) -> dict[str, object]:
    return {
        "example_id": example_id,
        "prompt": prompt_for_task(task, mode=mode),
        "answer": task.target,
        "info": {
            "task_id": task.task_id,
            "family": task.family,
            "seed": task.seed,
            "params": task.params,
            "target": task.target,
            "task": task.to_dict(),
        },
    }


def prompt_for_task(task: UndertowTask, *, mode: InteractionMode) -> list[dict[str, str]]:
    if mode == "standard":
        system = "You are solving an Undertow task with native tools."
        user = (
            "Call the available tools to discover hidden state, then call submit with the "
            f"final answer.\n\nTask spec: {task.public_spec}"
        )
    elif mode == "ptc":
        system = "You are in a Programmatic Tool Calling environment."
        user = (
            "Use Python to build durable wrappers, caches, solvers, local verifiers, "
            "planners, logs, and experiments around primitive hidden-state tools. "
            "Hidden state and reward are protected; local verifiers are not reward "
            "authorities. Finish with finish(answer) from the sandbox ptc package.\n\n"
            f"Task spec: {task.public_spec}"
        )
    else:
        raise ValueError("mode must be 'standard' or 'ptc'")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def to_hf_dataset(records: Sequence[Mapping[str, object]]):
    from datasets import Dataset

    return Dataset.from_list([dict(record) for record in records])


def generate_hf_dataset(**kwargs: Any):
    return to_hf_dataset(generate_dataset_records(**kwargs))


def save_hf_dataset(path: str | Path, **kwargs: Any) -> None:
    generate_hf_dataset(**kwargs).save_to_disk(str(path))


def push_hf_dataset(repo_id: str, **kwargs: Any) -> None:
    generate_hf_dataset(**kwargs).push_to_hub(repo_id)
