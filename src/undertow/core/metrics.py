from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from undertow.core.types import EpisodeResult, JsonDict


def summarize(results: Iterable[EpisodeResult | JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
    for result in results:
        row = result.to_dict() if isinstance(result, EpisodeResult) else dict(result)
        key = (
            str(row.get("family")),
            str(row.get("mode")),
            str(row.get("model")),
        )
        groups[key].append(row)

    summaries = []
    for (family, mode, model), rows in sorted(groups.items()):
        failures: Counter[str] = Counter()
        for row in rows:
            failures.update(row.get("failure_counter", {}))
        summaries.append(
            {
                "family": family,
                "mode": mode,
                "model": None if model == "None" else model,
                "num_episodes": len(rows),
                "solve_rate": _rate(rows, "solved"),
                "avg_reward": _avg(rows, "reward"),
                "avg_progress": _avg(rows, "progress"),
                "avg_efficiency": _avg(rows, "efficiency"),
                "avg_calls": _avg(rows, "calls"),
                "avg_state_changes": _avg(rows, "state_changes"),
                "avg_failure_calls": _avg(rows, "failure_calls"),
                "budget_exhaustion_rate": _budget_exhaustion_rate(rows),
                "failures": dict(failures),
            }
        )
    return summaries


def read_jsonl(path: str | Path) -> list[JsonDict]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_csv(path: str | Path, rows: Iterable[JsonDict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "mode",
        "model",
        "num_episodes",
        "solve_rate",
        "avg_reward",
        "avg_progress",
        "avg_efficiency",
        "avg_calls",
        "avg_state_changes",
        "avg_failure_calls",
        "budget_exhaustion_rate",
        "failures",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _avg(rows: list[JsonDict], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows) if rows else 0.0


def _rate(rows: list[JsonDict], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def _budget_exhaustion_rate(rows: list[JsonDict]) -> float:
    if not rows:
        return 0.0
    exhausted = sum(1 for row in rows if int(row["calls"]) >= int(row["budget"]))
    return exhausted / len(rows)
