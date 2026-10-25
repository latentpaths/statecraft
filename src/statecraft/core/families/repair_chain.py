from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

from statecraft.core.env import StatecraftEnv
from statecraft.core.types import FailureCode, JsonDict, StatecraftTask


def generate_repair_chain_task(
    *,
    seed: int,
    n_objects: int = 3,
    n_repair_steps: int = 3,
    n_tools: int = 6,
    n_distractor_tools: int = 2,
    staleness_rate: float = 0.2,
    partial_observation_rate: float = 0.0,
    shared_tool_ambiguity: float = 0.0,
    budget: int | None = None,
) -> StatecraftTask:
    if n_objects < 1:
        raise ValueError("n_objects must be >= 1")
    if n_repair_steps < 1:
        raise ValueError("n_repair_steps must be >= 1")
    if n_tools < 1:
        raise ValueError("n_tools must be >= 1")
    if n_distractor_tools < 0:
        raise ValueError("n_distractor_tools must be >= 0")

    rng = random.Random(seed)
    locations = [f"bay_{idx}" for idx in range(max(2, min(5, n_objects + 1)))]
    parts = [f"part_{idx}" for idx in range(max(n_objects, n_repair_steps))]
    real_tools = [f"tool_{idx}" for idx in range(n_tools)]
    distractor_tools = [f"distractor_tool_{idx}" for idx in range(n_distractor_tools)]
    objects: dict[str, JsonDict] = {}
    tool_catalog: dict[str, JsonDict] = {}

    for tool in real_tools:
        tool_catalog[tool] = {
            "id": tool,
            "kind": "repair",
            "compatible_parts": sorted(rng.sample(parts, k=rng.randint(1, len(parts)))),
        }
    for tool in distractor_tools:
        tool_catalog[tool] = {
            "id": tool,
            "kind": "distractor",
            "compatible_parts": [],
        }

    for idx in range(n_objects):
        object_id = f"machine_{idx}"
        part = parts[idx % len(parts)]
        sequence = _choose_tool_sequence(
            rng, real_tools, tool_catalog, part, n_repair_steps, shared_tool_ambiguity
        )
        objects[object_id] = {
            "id": object_id,
            "status": "broken",
            "part": part,
            "location": rng.choice(locations),
            "repair_stage": 0,
            "required_tools": sequence,
            "observation_version": 0,
            "critical": True,
        }

    _annotate_tool_stage_compatibility(tool_catalog, objects, parts)

    public_spec = {
        "family": "repair_chain",
        "objects": sorted(objects),
        "locations": locations,
        "tools_available": real_tools + distractor_tools,
        "target": "repair all critical machines",
        "tools": _operation_schemas(),
    }
    params = {
        "n_objects": n_objects,
        "n_repair_steps": n_repair_steps,
        "n_tools": n_tools,
        "n_distractor_tools": n_distractor_tools,
        "staleness_rate": staleness_rate,
        "partial_observation_rate": partial_observation_rate,
        "shared_tool_ambiguity": shared_tool_ambiguity,
    }
    hidden_state = {
        "objects": objects,
        "tool_catalog": tool_catalog,
        "current_location": None,
        "partial_fields": _partial_fields(rng, partial_observation_rate),
        "stale_after_stage": staleness_rate > 0,
    }
    return StatecraftTask(
        task_id=f"repair_chain:{seed}:{_short_hash(params)}",
        family="repair_chain",
        seed=seed,
        params=params,
        public_spec=public_spec,
        hidden_state=hidden_state,
        target={"repaired": sorted(objects)},
        budget=budget or _default_budget(
            n_objects=n_objects,
            n_repair_steps=n_repair_steps,
            n_tools=n_tools,
            n_distractor_tools=n_distractor_tools,
        ),
    )


class RepairChainEnv(StatecraftEnv):
    def __init__(self, task: StatecraftTask):
        super().__init__(task)
        self._objects = {key: dict(value) for key, value in task.hidden_state["objects"].items()}
        self._tool_catalog = {
            key: dict(value) for key, value in task.hidden_state["tool_catalog"].items()
        }
        self._current_location = task.hidden_state.get("current_location")
        self._partial_fields = set(task.hidden_state.get("partial_fields", []))
        self._stale_after_stage = bool(task.hidden_state.get("stale_after_stage"))

    def _handle_call(
        self, tool_name: str, kwargs: JsonDict
    ) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        if tool_name == "scan_object":
            return self._scan_object(kwargs)
        if tool_name == "inspect_tool":
            return self._inspect_tool(kwargs)
        if tool_name == "move_to":
            return self._move_to(kwargs)
        if tool_name == "apply_tool":
            return self._apply_tool(kwargs)
        if tool_name == "test_object":
            return self._test_object(kwargs)
        return {"ok": False, "error": f"unknown action {tool_name}"}, FailureCode.INVALID_ACTION, {}

    def _handle_submit(self, answer: Any) -> tuple[JsonDict, FailureCode | None]:
        repaired = _extract_repaired(answer)
        target = set(self.task.target["repaired"])
        actually_repaired = {
            object_id for object_id, obj in self._objects.items() if obj["status"] == "repaired"
        }
        if target <= repaired and target <= actually_repaired:
            return {"ok": True, "repaired": sorted(actually_repaired)}, None
        return (
            {
                "ok": False,
                "error": "not all critical machines are repaired",
                "reported": sorted(repaired),
                "actually_repaired": sorted(actually_repaired),
            },
            FailureCode.PREMATURE_SUBMIT,
        )

    def _scan_object(self, kwargs: JsonDict) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        object_id = kwargs.get("object_id")
        obj = self._objects.get(object_id)
        if obj is None:
            return {"ok": False, "error": f"unknown object {object_id}"}, FailureCode.UNKNOWN_STATE_REFERENCE, {}
        observation = {
            "ok": True,
            "object_id": object_id,
            "status": obj["status"],
            "repair_stage": obj["repair_stage"],
            "observation_version": obj["observation_version"],
        }
        if "part" not in self._partial_fields:
            observation["part"] = obj["part"]
        if "location" not in self._partial_fields:
            observation["location"] = obj["location"]
        return observation, None, {}

    def _inspect_tool(self, kwargs: JsonDict) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        tool_id = kwargs.get("tool_id")
        tool = self._tool_catalog.get(tool_id)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {tool_id}"}, FailureCode.UNKNOWN_STATE_REFERENCE, {}
        failure = FailureCode.DISTRACTOR_ACTION if tool["kind"] == "distractor" else None
        return {
            "ok": True,
            "tool_id": tool_id,
            "kind": tool["kind"],
            "compatible_parts": list(tool["compatible_parts"]),
            "compatible_stages_by_part": {
                part: list(stages)
                for part, stages in sorted(tool.get("compatible_stages_by_part", {}).items())
            },
        }, failure, {}

    def _move_to(self, kwargs: JsonDict) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        location = kwargs.get("location")
        if location not in self.task.public_spec["locations"]:
            return {"ok": False, "error": f"unknown location {location}"}, FailureCode.UNKNOWN_STATE_REFERENCE, {}
        previous = self._current_location
        self._current_location = location
        return {"ok": True, "location": location}, None, {"location": {"from": previous, "to": location}}

    def _apply_tool(self, kwargs: JsonDict) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        object_id = kwargs.get("object_id")
        tool_id = kwargs.get("tool_id")
        obj = self._objects.get(object_id)
        tool = self._tool_catalog.get(tool_id)
        if obj is None or tool is None:
            return {"ok": False, "error": "unknown object or tool"}, FailureCode.UNKNOWN_STATE_REFERENCE, {}
        if "observation_version" in kwargs and kwargs["observation_version"] != obj["observation_version"]:
            return {"ok": False, "error": "stale object observation"}, FailureCode.STALE_STATE_REFERENCE, {}
        if tool["kind"] == "distractor":
            return {"ok": False, "error": "tool cannot repair objects"}, FailureCode.DISTRACTOR_ACTION, {}
        if self._current_location != obj["location"]:
            return {
                "ok": False,
                "error": "wrong location",
                "current_location": self._current_location,
            }, FailureCode.VIOLATED_CONSTRAINT, {}
        if obj["status"] == "repaired":
            return {"ok": True, "status": "already repaired"}, FailureCode.REPEATED_CALL, {}
        expected_tool = obj["required_tools"][obj["repair_stage"]]
        if tool_id != expected_tool:
            return {
                "ok": False,
                "error": "wrong tool for current repair stage",
                "repair_stage": obj["repair_stage"],
            }, FailureCode.VIOLATED_CONSTRAINT, {}

        before_stage = obj["repair_stage"]
        obj["repair_stage"] += 1
        if self._stale_after_stage:
            obj["observation_version"] += 1
        if obj["repair_stage"] >= len(obj["required_tools"]):
            obj["status"] = "repaired"
        return {
            "ok": True,
            "object_id": object_id,
            "repair_stage": obj["repair_stage"],
            "status": obj["status"],
            "observation_version": obj["observation_version"],
        }, None, {
            object_id: {
                "repair_stage": {"from": before_stage, "to": obj["repair_stage"]},
                "status": obj["status"],
            }
        }

    def _test_object(self, kwargs: JsonDict) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        object_id = kwargs.get("object_id")
        obj = self._objects.get(object_id)
        if obj is None:
            return {"ok": False, "error": f"unknown object {object_id}"}, FailureCode.UNKNOWN_STATE_REFERENCE, {}
        if "observation_version" in kwargs and kwargs["observation_version"] != obj["observation_version"]:
            return {"ok": False, "error": "stale object observation"}, FailureCode.STALE_STATE_REFERENCE, {}
        ok = obj["status"] == "repaired"
        return {
            "ok": ok,
            "object_id": object_id,
            "status": obj["status"],
            "repair_stage": obj["repair_stage"],
            "required_stages": len(obj["required_tools"]),
            "observation_version": obj["observation_version"],
        }, None if ok else FailureCode.VIOLATED_CONSTRAINT, {}

    def _snapshot_state(self) -> JsonDict:
        return {
            "objects": {key: dict(value) for key, value in self._objects.items()},
            "current_location": self._current_location,
        }

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        objects = state.get("objects")
        if isinstance(objects, Mapping):
            self._objects = {
                str(key): dict(value)
                for key, value in objects.items()
                if isinstance(value, Mapping)
            }
        self._current_location = state.get("current_location")


def _choose_tool_sequence(
    rng: random.Random,
    real_tools: list[str],
    tool_catalog: dict[str, JsonDict],
    part: str,
    n_repair_steps: int,
    shared_tool_ambiguity: float,
) -> list[str]:
    compatible = [
        tool_id for tool_id in real_tools if part in tool_catalog[tool_id]["compatible_parts"]
    ]
    if not compatible:
        compatible = [rng.choice(real_tools)]
        tool_catalog[compatible[0]]["compatible_parts"].append(part)
        tool_catalog[compatible[0]]["compatible_parts"].sort()
    sequence = []
    for _ in range(n_repair_steps):
        if rng.random() < shared_tool_ambiguity:
            sequence.append(rng.choice(compatible))
        else:
            sequence.append(compatible[len(sequence) % len(compatible)])
    return sequence


def _annotate_tool_stage_compatibility(
    tool_catalog: dict[str, JsonDict],
    objects: dict[str, JsonDict],
    parts: list[str],
) -> None:
    stage_map: dict[str, dict[str, set[int]]] = {
        tool_id: {part: set() for part in parts} for tool_id in tool_catalog
    }
    for obj in objects.values():
        part = str(obj["part"])
        for stage, tool_id in enumerate(obj["required_tools"]):
            stage_map[tool_id][part].add(stage)
    for tool_id, tool in tool_catalog.items():
        if tool["kind"] == "distractor":
            tool["compatible_stages_by_part"] = {}
            continue
        tool["compatible_stages_by_part"] = {
            part: sorted(stage_map[tool_id][part])
            for part in tool["compatible_parts"]
            if stage_map[tool_id][part]
        }


def _default_budget(
    *,
    n_objects: int,
    n_repair_steps: int,
    n_tools: int,
    n_distractor_tools: int,
) -> int:
    inspect_calls = n_tools + n_distractor_tools
    scan_calls = n_objects
    movement_calls = n_objects
    repair_calls = n_objects * n_repair_steps
    test_calls = n_objects
    submit_calls = 1
    public_solver_calls = (
        inspect_calls + scan_calls + movement_calls + repair_calls + test_calls + submit_calls
    )
    return public_solver_calls * 3


def _partial_fields(rng: random.Random, rate: float) -> list[str]:
    fields = []
    if rng.random() < rate:
        fields.append("part")
    if rng.random() < rate:
        fields.append("location")
    return fields


def _operation_schemas() -> list[JsonDict]:
    return [
        _schema("scan_object", "Scan a machine and return its currently observable state.", {"object_id": "string"}),
        _schema("inspect_tool", "Inspect a tool and return its compatibility information.", {"tool_id": "string"}),
        _schema("move_to", "Move the agent to a location.", {"location": "string"}),
        _schema(
            "apply_tool",
            "Apply a tool to a machine at the current location.",
            {"tool_id": "string", "object_id": "string", "observation_version": "integer"},
            required=["tool_id", "object_id"],
        ),
        _schema(
            "test_object",
            "Test whether a machine is repaired.",
            {"object_id": "string", "observation_version": "integer"},
            required=["object_id"],
        ),
    ]


def _schema(
    name: str, description: str, properties: dict[str, str], required: list[str] | None = None
) -> JsonDict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": value} for key, value in properties.items()},
            "required": list(required if required is not None else properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _extract_repaired(answer: Any) -> set[str]:
    if isinstance(answer, str):
        return {answer}
    if isinstance(answer, list):
        return {str(item) for item in answer}
    if isinstance(answer, dict):
        value = answer.get("repaired", answer.get("objects", answer.get("answer", [])))
        return _extract_repaired(value)
    return set()


def _short_hash(value: Any) -> str:
    return hashlib.sha1(repr(value).encode()).hexdigest()[:10]
