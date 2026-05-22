from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

JsonDict = dict[str, Any]


class FailureCode(StrEnum):
    INVALID_ACTION = "invalid_action"
    SCHEMA_ERROR = "schema_error"
    UNKNOWN_STATE_REFERENCE = "unknown_state_reference"
    INCORRECT_VALUE = "incorrect_value"
    STALE_STATE_REFERENCE = "stale_state_reference"
    VIOLATED_CONSTRAINT = "violated_constraint"
    DISTRACTOR_ACTION = "distractor_action"
    PREMATURE_SUBMIT = "premature_submit"
    BUDGET_EXCEEDED = "budget_exceeded"
    REPEATED_CALL = "repeated_call"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class UndertowTask:
    task_id: str
    family: str
    seed: int
    params: JsonDict
    public_spec: JsonDict
    hidden_state: JsonDict
    target: JsonDict
    budget: int

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    order: int
    tool_name: str
    args: JsonDict
    observation: JsonDict
    failure: str | None = None
    state_delta: JsonDict = field(default_factory=dict)
    consumed_budget: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    family: str
    mode: str
    model: str | None
    params: JsonDict
    solved: bool
    reward: float
    answer: Any
    calls: int
    budget: int
    reference_calls: int
    progress: float
    failed_calls: int
    first_failure: str | None
    final_failure: str | None
    failure_counter: dict[str, int]
    trace: list[JsonDict]

    def to_dict(self) -> JsonDict:
        return asdict(self)
