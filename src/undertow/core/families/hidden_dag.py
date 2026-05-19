from __future__ import annotations

import hashlib
import random
from typing import Any

from undertow.core.env import UndertowEnv
from undertow.core.types import FailureCode, JsonDict, UndertowTask


def generate_hidden_dag_task(
    *,
    seed: int,
    n_core: int = 10,
    depth: int = 5,
    n_connected_distractors: int = 0,
    n_disconnected_distractors: int = 0,
    observation_mode: str = "minimal",
    invalid_return_mode: str = "error",
    budget: int | None = None,
) -> UndertowTask:
    if n_core < 1:
        raise ValueError("n_core must be >= 1")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if observation_mode not in {"minimal", "reminder"}:
        raise ValueError("observation_mode must be 'minimal' or 'reminder'")
    if invalid_return_mode not in {"error", "noisy"}:
        raise ValueError("invalid_return_mode must be 'error' or 'noisy'")

    rng = random.Random(seed)
    chain_len = min(depth, n_core)
    initial_values: dict[str, int] = {"v0": rng.randint(100, 999)}
    functions: list[JsonDict] = []

    previous = "v0"
    for idx in range(chain_len):
        output = f"v{idx + 1}"
        functions.append(
            _function_spec(
                name=f"core_{idx}",
                inputs=[previous],
                output=output,
                node_type="core",
                description=f"Transforms {previous} into {output}.",
            )
        )
        previous = output

    # Extra core nodes are independent prerequisites for later chain functions.
    for idx in range(chain_len, n_core):
        source = f"source_{idx}"
        output = f"v{idx + 1}"
        initial_values[source] = rng.randint(100, 999)
        target_pos = rng.randint(1, chain_len - 1) if chain_len > 1 else 0
        functions[target_pos]["inputs"].append(output)
        functions[target_pos]["description"] = (
            f"Combines {', '.join(functions[target_pos]['inputs'])} into "
            f"{functions[target_pos]['output']}."
        )
        functions.append(
            _function_spec(
                name=f"core_{idx}",
                inputs=[source],
                output=output,
                node_type="core",
                description=f"Transforms {source} into {output}.",
            )
        )

    for idx in range(n_connected_distractors):
        input_var = rng.choice([f["output"] for f in functions if f["node_type"] == "core"] + list(initial_values))
        functions.append(
            _function_spec(
                name=f"connected_distractor_{idx}",
                inputs=[input_var],
                output=f"d_connected_{idx}",
                node_type="connected_distractor",
                description=f"Produces a type-compatible but irrelevant value from {input_var}.",
            )
        )

    for idx in range(n_disconnected_distractors):
        source = f"noise_source_{idx}"
        initial_values[source] = rng.randint(100, 999)
        functions.append(
            _function_spec(
                name=f"disconnected_distractor_{idx}",
                inputs=[source],
                output=f"d_disconnected_{idx}",
                node_type="disconnected_distractor",
                description=f"Produces an irrelevant value from {source}.",
            )
        )

    value_table, solution_order = _compute_values(seed, functions, initial_values)
    target_var = f"v{chain_len}"
    target_value = value_table[target_var]
    public_functions = [
        {
            "name": f["name"],
            "inputs": list(f["inputs"]),
            "output": f["output"],
            "description": f["description"],
        }
        for f in functions
    ]
    params = {
        "n_core": n_core,
        "depth": depth,
        "n_connected_distractors": n_connected_distractors,
        "n_disconnected_distractors": n_disconnected_distractors,
        "observation_mode": observation_mode,
        "invalid_return_mode": invalid_return_mode,
    }
    public_spec = {
        "family": "hidden_dag",
        "initial_values": dict(initial_values),
        "target": target_var,
        "functions": public_functions,
        "tools": [_tool_schema(f) for f in public_functions],
    }
    hidden_state = {
        "functions": functions,
        "values": value_table,
        "known_values": dict(initial_values),
        "solution": [name for name in solution_order if _function_by_name(functions, name)["node_type"] == "core"],
    }
    return UndertowTask(
        task_id=f"hidden_dag:{seed}:{_short_hash(params)}",
        family="hidden_dag",
        seed=seed,
        params=params,
        public_spec=public_spec,
        hidden_state=hidden_state,
        target={"variable": target_var, "value": target_value},
        budget=budget or max(8, (n_core + n_connected_distractors + n_disconnected_distractors) * 2),
    )


class HiddenDagEnv(UndertowEnv):
    def __init__(self, task: UndertowTask):
        super().__init__(task)
        self._functions = {f["name"]: dict(f) for f in task.hidden_state["functions"]}
        self._values = dict(task.hidden_state["values"])
        self._known_values = dict(task.hidden_state["known_values"])

    def _handle_call(
        self, tool_name: str, kwargs: JsonDict
    ) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        func = self._functions.get(tool_name)
        if func is None:
            return {"ok": False, "error": f"unknown function {tool_name}"}, FailureCode.INVALID_ACTION, {}

        expected_inputs = set(func["inputs"])
        if set(kwargs) != expected_inputs:
            return (
                {
                    "ok": False,
                    "error": "schema mismatch",
                    "required": sorted(expected_inputs),
                    "received": sorted(kwargs),
                },
                FailureCode.SCHEMA_ERROR,
                {},
            )

        for input_name in func["inputs"]:
            if input_name not in self._known_values:
                return self._invalid_input_observation(func, FailureCode.UNKNOWN_STATE_REFERENCE)
            if kwargs[input_name] != self._values[input_name]:
                return self._invalid_input_observation(func, FailureCode.INCORRECT_VALUE)

        output = func["output"]
        self._known_values[output] = self._values[output]
        observation = {"ok": True, "variable": output, "value": self._values[output]}
        if self.task.params["observation_mode"] == "reminder":
            observation["known_values"] = dict(sorted(self._known_values.items()))
        failure = (
            FailureCode.DISTRACTOR_ACTION
            if str(func["node_type"]).endswith("distractor")
            else None
        )
        return observation, failure, {"known": {output: self._values[output]}}

    def _handle_submit(self, answer: Any) -> tuple[JsonDict, FailureCode | None]:
        target_var = self.task.target["variable"]
        target_value = self.task.target["value"]
        submitted = _coerce_answer(answer, target_var)
        if submitted == target_value:
            return {"ok": True, "target": target_var, "value": target_value}, None
        if target_var not in self._known_values:
            return (
                {"ok": False, "error": "target value has not been computed", "target": target_var},
                FailureCode.PREMATURE_SUBMIT,
            )
        return (
            {"ok": False, "error": "incorrect target value", "target": target_var},
            FailureCode.INCORRECT_VALUE,
        )

    def _invalid_input_observation(
        self, func: JsonDict, failure: FailureCode
    ) -> tuple[JsonDict, FailureCode, JsonDict]:
        if self.task.params["invalid_return_mode"] == "noisy":
            noisy = _stable_value(self.task.seed, func["name"], "noisy", len(self._known_values))
            output = func["output"]
            self._known_values[output] = noisy
            observation = {"ok": True, "variable": output, "value": noisy, "warning": failure.value}
            if self.task.params["observation_mode"] == "reminder":
                observation["known_values"] = dict(sorted(self._known_values.items()))
            return observation, failure, {"known": {output: noisy}}
        return {"ok": False, "error": failure.value}, failure, {}


def _function_spec(
    *, name: str, inputs: list[str], output: str, node_type: str, description: str
) -> JsonDict:
    return {
        "name": name,
        "inputs": inputs,
        "output": output,
        "node_type": node_type,
        "description": description,
    }


def _tool_schema(func: JsonDict) -> JsonDict:
    return {
        "type": "function",
        "name": func["name"],
        "description": func["description"],
        "parameters": {
            "type": "object",
            "properties": {name: {"type": "integer"} for name in func["inputs"]},
            "required": list(func["inputs"]),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _compute_values(
    seed: int, functions: list[JsonDict], initial_values: dict[str, int]
) -> tuple[dict[str, int], list[str]]:
    values = dict(initial_values)
    pending = list(functions)
    order: list[str] = []
    while pending:
        progressed = False
        next_pending = []
        for func in pending:
            if all(name in values for name in func["inputs"]):
                inputs = [values[name] for name in func["inputs"]]
                values[func["output"]] = _stable_value(seed, func["name"], *inputs)
                order.append(func["name"])
                progressed = True
            else:
                next_pending.append(func)
        if not progressed:
            raise ValueError("generated hidden DAG has unresolved dependencies")
        pending = next_pending
    return values, order


def _function_by_name(functions: list[JsonDict], name: str) -> JsonDict:
    return next(func for func in functions if func["name"] == name)


def _stable_value(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).hexdigest()
    return int(digest[:12], 16) % 10000


def _short_hash(value: Any) -> str:
    return hashlib.sha1(repr(value).encode()).hexdigest()[:10]


def _coerce_answer(answer: Any, target_var: str) -> int | None:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        try:
            return int(answer.strip())
        except ValueError:
            return None
    if isinstance(answer, dict):
        for key in (target_var, "value", "answer"):
            if key in answer:
                return _coerce_answer(answer[key], target_var)
    return None
