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
        self._solved = False
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
        if self._budget_left() <= 0:
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
        if self._budget_left() <= 0:
            return self._record(
                "submit",
                {"answer": answer},
                {"ok": False, "error": "budget exhausted"},
                FailureCode.BUDGET_EXCEEDED,
                consumed_budget=False,
            )
        observation, failure = self._handle_submit(answer)
        self._submitted = True
        self._solved = failure is None
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
            "solved": self._solved,
            "answer": self._answer,
            "env_state": self._snapshot_state(),
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
        self._solved = bool(snapshot.get("solved", False))
        self._answer = snapshot.get("answer")
        env_state = snapshot.get("env_state", {})
        self._restore_state(env_state if isinstance(env_state, Mapping) else {})

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
        first_failure: str | None = None
        last_failure: str | None = None
        failed_calls = 0
        for event in self._trace:
            if event.failure:
                failures[event.failure] = failures.get(event.failure, 0) + 1
                if first_failure is None:
                    first_failure = event.failure
                last_failure = event.failure
                if event.consumed_budget:
                    failed_calls += 1
        calls = sum(1 for event in self._trace if event.consumed_budget)
        progress = self._progress()
        reference_calls = self._reference_calls()
        reward = reference_calls / max(calls, reference_calls) if self._solved and reference_calls else 0.0
        return EpisodeResult(
            task_id=self.task.task_id,
            family=self.task.family,
            mode=mode,
            model=model,
            params=dict(self.task.params),
            solved=self._solved,
            reward=reward,
            answer=self._answer,
            calls=calls,
            budget=self.task.budget,
            reference_calls=reference_calls,
            progress=progress,
            failed_calls=failed_calls,
            first_failure=first_failure,
            final_failure=None if self._solved else last_failure,
            failure_counter=failures,
            trace=self.trace(),
        )

    def _budget_left(self) -> int:
        used = sum(1 for event in self._trace if event.consumed_budget)
        return max(0, self.task.budget - used)

    def _progress(self) -> float:
        if self._solved:
            return 1.0
        if self.task.family == "hidden_dag":
            solution = self.task.hidden_state.get("solution", [])
            denominator = len(solution) + 1
            if denominator == 0:
                return 0.0
            solution_outputs = {}
            for func in self.task.hidden_state.get("functions", []):
                if func.get("name") in solution:
                    solution_outputs[str(func["output"])] = self.task.hidden_state.get(
                        "values", {}
                    ).get(func["output"])
            correct_outputs = set()
            for event in self._trace:
                known = event.state_delta.get("known")
                if isinstance(known, dict):
                    for key, value in known.items():
                        output = str(key)
                        if output in solution_outputs and value == solution_outputs[output]:
                            correct_outputs.add(output)
            return min(1.0 - (1.0 / denominator), len(correct_outputs) / denominator)
        if self.task.family == "repair_chain":
            objects = self.task.hidden_state.get("objects", {})
            total_stages = sum(len(obj.get("required_tools", [])) for obj in objects.values())
            denominator = total_stages + 1
            if total_stages == 0:
                return 0.0
            progressed: dict[str, int] = {}
            for event in self._trace:
                for object_id, delta in event.state_delta.items():
                    if not isinstance(delta, dict):
                        continue
                    stage_delta = delta.get("repair_stage")
                    if isinstance(stage_delta, dict) and isinstance(stage_delta.get("to"), int):
                        progressed[str(object_id)] = max(
                            progressed.get(str(object_id), 0),
                            int(stage_delta["to"]),
                        )
            return min(1.0 - (1.0 / denominator), sum(progressed.values()) / denominator)
        return 0.0

    def _reference_calls(self) -> int:
        if self.task.family == "hidden_dag":
            return len(self.task.public_spec.get("functions", [])) + 1
        if self.task.family == "repair_chain":
            objects = self.task.hidden_state.get("objects", {})
            n_objects = len(objects)
            n_repair_steps = sum(len(obj.get("required_tools", [])) for obj in objects.values())
            n_tools = len(self.task.public_spec.get("tools_available", []))
            return n_tools + n_objects + n_objects + n_repair_steps + n_objects + 1
        return self.task.budget

    def _handle_call(
        self, tool_name: str, kwargs: JsonDict
    ) -> tuple[JsonDict, FailureCode | None, JsonDict]:
        raise NotImplementedError

    def _handle_submit(self, answer: Any) -> tuple[JsonDict, FailureCode | None]:
        raise NotImplementedError

    def _snapshot_state(self) -> JsonDict:
        return {}

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        pass

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
