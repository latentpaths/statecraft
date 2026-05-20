from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Json = dict[str, Any]


RAW_FIELDS = {
    "mode",
    "family",
    "model",
    "params",
    "solved",
    "reward",
    "calls",
    "budget",
    "reference_calls",
    "efficiency",
    "progress",
    "state_changes",
    "failure_calls",
    "failure_counter",
    "first_failure",
    "final_failure",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate normalized Undertow experiment rows.")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "results" / "raw")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)

    rows = read_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", summarize(rows), SUMMARY_FIELDS)
    write_csv(args.output / "by_config.csv", by_config(rows), BY_CONFIG_FIELDS)
    write_csv(args.output / "failures.csv", failures(rows), FAILURE_FIELDS)
    return 0


def read_rows(path: Path) -> list[Json]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows = []
    for file in files:
        with file.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = RAW_FIELDS - set(row)
                if missing:
                    raise ValueError(f"{file}:{line_number} missing fields: {sorted(missing)}")
                rows.append(row)
    return rows


def summarize(rows: list[Json]) -> list[Json]:
    return [summary_row(keys, values) for keys, values in group(rows, ["mode", "family", "model"]).items()]


def by_config(rows: list[Json]) -> list[Json]:
    return [
        summary_row(keys, values)
        for keys, values in group(rows, ["mode", "family", "model", "params"]).items()
    ]


def summary_row(keys: tuple[Any, ...], rows: list[Json]) -> Json:
    key_names = ["mode", "family", "model", "params"]
    out = {key_names[idx]: key for idx, key in enumerate(keys)}
    out.update(
        {
            "num_episodes": len(rows),
            "solve_rate": sum(1 for row in rows if row["solved"]) / len(rows) if rows else 0.0,
            "avg_reward": avg(rows, "reward"),
            "avg_progress": avg(rows, "progress"),
            "avg_efficiency": avg(rows, "efficiency"),
            "avg_calls": avg(rows, "calls"),
            "avg_state_changes": avg(rows, "state_changes"),
            "avg_failure_calls": avg(rows, "failure_calls"),
            "budget_exhaustion_rate": sum(1 for row in rows if row["calls"] >= row["budget"])
            / len(rows)
            if rows
            else 0.0,
        }
    )
    return out


def failures(rows: list[Json]) -> list[Json]:
    output = []
    for keys, values in group(rows, ["mode", "family", "model"]).items():
        counts: Counter[str] = Counter()
        for row in values:
            counts.update(row["failure_counter"])
        total = sum(counts.values())
        for failure, count in sorted(counts.items()):
            output.append(
                {
                    "mode": keys[0],
                    "family": keys[1],
                    "model": keys[2],
                    "failure": failure,
                    "count": count,
                    "share": count / total if total else 0.0,
                }
            )
    return output


def group(rows: list[Json], keys: list[str]) -> dict[tuple[Any, ...], list[Json]]:
    grouped: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])))


def avg(rows: list[Json], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def write_csv(path: Path, rows: list[Json], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


SUMMARY_FIELDS = [
    "mode",
    "family",
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
]
BY_CONFIG_FIELDS = ["params", *SUMMARY_FIELDS]
FAILURE_FIELDS = ["mode", "family", "model", "failure", "count", "share"]


if __name__ == "__main__":
    raise SystemExit(main())
