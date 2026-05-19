from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from undertow.core.types import EpisodeResult, FailureCode, JsonDict, TraceEvent, UndertowTask


class UndertowEnv:
    """Base stateful benchmark environment."""

    def __init__(self, task: UndertowTask):
        self.task = task
        self._trace: list[TraceEvent] = []
        self._call_counts: dict[tuple[str, str], int] = {}
        self._submitted = False
        self._success = False
        self._answer: Any = None

    def public_spec(self) -> JsonDict:
        return self.task.public_spec

    def tools(self) -> list[JsonDict]:
        return list(self.task.public_spec.get("tools", []))

    def call(self, tool_name: str, **kwargs: Any) -> JsonDict:
        if self._submitted:
            return self._record(
                tool_name,
                kwargs,
                {"ok": False, "error": "episode already submitted"},
                FailureCode.INVALID_ACTION,
                consumed_budget=False,
            )
        if self.remaining_budget() <= 0:
            return self._record(
                tool_name,
                kwargs,
                {"ok": False, "error": "budget exhausted"},
                FailureCode.BUDGET_EXCEEDED,
                consumed_budget=False,
            )

        key = (tool_name, self._stable_args(kwargs))
        repeated = key in self._call_counts
        self._call_counts[key] = self._call_counts.get(key, 0) + 1

        observation, failure, delta = self._handle_call(tool_name, kwargs)
        if repeated and failure is None:
            failure = FailureCode.REPEATED_CALL
            observation = {**observation, "warning": FailureCode.REPEATED_CALL.value}
        return self._record(tool_name, kwargs, observation, failure, delta)

    def submit(self, answer: Any) -> JsonDict:
        if self._submitted:
            return self._record(
                "submit",
                {"answer": answer},
                {"ok": False, "error": "episode already submitted"},
                FailureCode.INVALID_ACTION,
                consumed_budget=False,
            )
        if self.remaining_budget() <= 0:
            return self._record(
                "submit",
                {"answer": answer},
                {"ok": False, "error": "budget exhausted"},
                FailureCode.BUDGET_EXCEEDED,
                consumed_budget=False,
            )
        observation, failure = self._handle_submit(answer)
        self._submitted = True
        self._success = failure is None
        self._answer = answer
        return self._record("submit", {"answer": answer}, observation, failure)

    def trace(self) -> list[JsonDict]:
        return [event.to_dict() for event in self._trace]

    def snapshot(self) -> JsonDict:
        """Return a JSON-serializable session snapshot."""
        return {
            "trace": self.trace(),
            "call_counts": [
                {"tool_name": tool_name, "args_key": args_key, "count": count}
                for (tool_name, args_key), count in sorted(self._call_counts.items())
            ],
            "submitted": self._submitted,
            "success": self._success,
            "answer": self._answer,
        }

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Restore a session snapshot produced by :meth:`snapshot`."""
        if not snapshot:
            return
        self._trace = [
            TraceEvent(**dict(event))
            for event in snapshot.get("trace", [])
            if isinstance(event, Mapping)
        ]
        self._call_counts = {}
        for item in snapshot.get("call_counts", []):
            if not isinstance(item, Mapping):
                continue
            tool_name = item.get("tool_name")
            args_key = item.get("args_key")
            count = item.get("count")
            if isinstance(tool_name, str) and isinstance(args_key, str) and isinstance(count, int):
                self._call_counts[(tool_name, args_key)] = count
        self._submitted = bool(snapshot.get("submitted", False))
        self._success = bool(snapshot.get("success", False))
        self._answer = snapshot.get("answer")

    @property
    def submitted(self) -> bool:
        return self._submitted

    def record_external(
        self,
        tool_name: str,
        args: JsonDict,
        observation: JsonDict,
        *,
        failure: FailureCode | None = None,
        state_delta: JsonDict | None = None,
        consumed_budget: bool = False,
    ) -> JsonDict:
        return self._record(
            tool_name,
            args,
            observation,
            failure,
            state_delta,
            consumed_budget=consumed_budget,
        )

    def result(
        self,
        mode: str = "scripted",
        model: str | None = None,
    ) -> EpisodeResult:
        failures: dict[str, int] = {}
        for event in self._trace:
            if event.failure:
                failures[event.failure] = failures.get(event.failure, 0) + 1
        return EpisodeResult(
            task_id=self.task.task_id,
            family=self.task.family,
            mode=mode,
            model=model,
            params=dict(self.task.params),
            success=self._success,
            answer=self._answer,
            num_calls=sum(1 for event in self._trace if event.consumed_budget),
            invalid_calls=failures.get(FailureCode.INVALID_ACTION.value, 0)
            + failures.get(FailureCode.SCHEMA_ERROR.value, 0),
            distractor_calls=failures.get(FailureCode.DISTRACTOR_ACTION.value, 0),
            stale_state_errors=failures.get(FailureCode.STALE_STATE_REFERENCE.value, 0),
            unknown_reference_errors=failures.get(FailureCode.UNKNOWN_STATE_REFERENCE.value, 0),
            repeated_calls=failures.get(FailureCode.REPEATED_CALL.value, 0),
            failure_counter=failures,
            trace=self.trace(),
        )

    def remaining_budget(self) -> int:
        used = sum(1 for event in self._trace if event.consumed_budget)
        return max(0, self.task.budget - used)

    def _handle_call(
        self, tool_name: str, kwargs: JsonDict
    ) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        raise NotImplementedError

    def _handle_submit(self, answer: Any) -> tuple[JsonDict, FailureCode | None]:
        raise NotImplementedError

    def _record(
        self,
        tool_name: str,
        args: JsonDict,
        observation: JsonDict,
        failure: FailureCode | None,
        state_delta: JsonDict | None = None,
        consumed_budget: bool = True,
    ) -> JsonDict:
        event = TraceEvent(
            order=len(self._trace) + 1,
            tool_name=tool_name,
            args=dict(args),
            observation=dict(observation),
            failure=failure.value if failure else None,
            state_delta=dict(state_delta or {}),
            consumed_budget=consumed_budget,
        )
        self._trace.append(event)
        return event.observation

    @staticmethod
    def _stable_args(args: JsonDict) -> str:
        return repr(sorted((key, repr(value)) for key, value in args.items()))
