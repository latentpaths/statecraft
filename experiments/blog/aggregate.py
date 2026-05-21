from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import (  # noqa: E402
    DEFAULT_CONFIG,
    config_result_dir,
    model_slug,
)

Json = dict[str, Any]

RAW_FIELDS = {
    "config",
    "seed",
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
    "num_turns",
    "input_tokens",
    "output_tokens",
    "total_tokens",
}
METRIC_FIELDS = [
    "num_episodes",
    "solve_rate",
    "avg_reward",
    "avg_progress",
    "avg_efficiency",
    "avg_calls",
    "avg_reference_calls",
    "avg_state_changes",
    "avg_failure_calls",
    "budget_exhaustion_rate",
    "avg_call_ratio",
    "avg_reference_call_ratio",
    "failure_call_rate",
    "avg_success_calls",
    "avg_num_turns",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
]
BASE_EPISODE_FIELDS = [
    "config",
    "model",
    "mode",
    "family",
    "seed",
    "params",
    "solved",
    "reward",
    "progress",
    "efficiency",
    "calls",
    "budget",
    "reference_calls",
    "state_changes",
    "failure_calls",
    "num_turns",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "call_ratio",
    "reference_call_ratio",
    "failure_call_share",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate normalized Undertow experiment rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="raw dir, config result dir, or one JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional config result dir override; CSVs are still written under <output>/csvs",
    )
    args = parser.parse_args(argv)

    input_path = args.input or config_result_dir(args.config)
    files = raw_files(input_path)
    rows = read_rows(files)
    result_dir = args.output or infer_result_dir(input_path, args.config, rows)
    write_outputs(rows, result_dir)
    print(f"output_dir={result_dir / 'csvs'}")
    return 0


def raw_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path]
    if path.name == "raw":
        return sorted(path.glob("*.jsonl"))
    return sorted(path.glob("*/raw/*.jsonl"))


def infer_result_dir(input_path: Path, config_path: Path, rows: list[Json]) -> Path:
    path = input_path.resolve()
    if not input_path.exists():
        return config_result_dir(config_path)
    if path.is_file():
        path = path.parent
    if path.name == "raw":
        parent = path.parent
        models = {model_slug(str(row["model"])) for row in rows}
        if parent.name in models:
            return parent.parent
        return parent
    return path


def read_rows(files: list[Path]) -> list[Json]:
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
                rows.append(normalize_row(row))
    return rows


def normalize_row(row: Json) -> Json:
    out = dict(row)
    out["solved"] = bool(row["solved"])
    for key in [
        "reward",
        "calls",
        "budget",
        "reference_calls",
        "efficiency",
        "progress",
        "state_changes",
        "failure_calls",
    ]:
        out[key] = float(row[key])
    for key in ["num_turns", "input_tokens", "output_tokens", "total_tokens"]:
        out[key] = float(row[key])
    out["params_dict"] = json.loads(row["params"])
    out["budget_exhausted"] = out["calls"] >= out["budget"]
    out["call_ratio"] = safe_div(out["calls"], out["budget"])
    out["reference_call_ratio"] = safe_div(out["calls"], out["reference_calls"])
    out["failure_call_share"] = safe_div(out["failure_calls"], out["calls"])
    return out


def write_outputs(rows: list[Json], result_dir: Path) -> None:
    write_csv_set(rows, result_dir / "csvs")
    for model, model_rows in group(rows, ["model"]).items():
        write_csv_set(model_rows, result_dir / model_slug(str(model[0])) / "csvs")


def write_csv_set(rows: list[Json], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    param_fields = sorted({key for row in rows for key in row["params_dict"]})
    write_csv(output_dir / "episodes.csv", episode_rows(rows, param_fields), [*BASE_EPISODE_FIELDS, *param_fields])
    write_csv(output_dir / "summary.csv", summarize(rows), METRIC_FIELDS)
    write_grouped(output_dir / "by_config_name.csv", rows, ["config"])
    write_grouped(output_dir / "by_model.csv", rows, ["model"])
    write_grouped(output_dir / "by_mode.csv", rows, ["mode"])
    write_grouped(output_dir / "by_family.csv", rows, ["family"])
    write_grouped(output_dir / "by_model_mode.csv", rows, ["model", "mode"])
    write_grouped(output_dir / "by_model_family.csv", rows, ["model", "family"])
    write_grouped(output_dir / "by_model_mode_family.csv", rows, ["model", "mode", "family"])
    write_grouped(output_dir / "by_config.csv", rows, ["config", "model", "mode", "family", "params"])
    write_grouped(output_dir / "by_params.csv", rows, ["family", *param_fields])
    write_grouped(output_dir / "by_seed.csv", rows, ["config", "model", "mode", "family", "params", "seed"])
    write_csv(output_dir / "failures.csv", failures(rows, ["config", "model", "mode", "family"]), [
        "config",
        "model",
        "mode",
        "family",
        "failure",
        "count",
        "share",
    ])
    write_csv(output_dir / "failure_by_config.csv", failures(rows, ["config", "model", "mode", "family", "params"]), [
        "config",
        "model",
        "mode",
        "family",
        "params",
        "failure",
        "count",
        "share",
    ])


def episode_rows(rows: list[Json], param_fields: list[str]) -> list[Json]:
    output = []
    for row in sorted_rows(rows):
        item = {key: row.get(key) for key in BASE_EPISODE_FIELDS}
        item.update({key: row["params_dict"].get(key) for key in param_fields})
        output.append(item)
    return output


def summarize(rows: list[Json]) -> list[Json]:
    return [summary_metrics(rows)] if rows else []


def write_grouped(path: Path, rows: list[Json], keys: list[str]) -> None:
    grouped_rows = [summary_row(keys, group_key, values) for group_key, values in group(rows, keys).items()]
    write_csv(path, grouped_rows, [*keys, *METRIC_FIELDS])


def summary_row(keys: list[str], group_key: tuple[Any, ...], rows: list[Json]) -> Json:
    out = dict(zip(keys, group_key, strict=True))
    out.update(summary_metrics(rows))
    return out


def summary_metrics(rows: list[Json]) -> Json:
    return {
        "num_episodes": len(rows),
        "solve_rate": avg_bool(rows, "solved"),
        "avg_reward": avg(rows, "reward"),
        "avg_progress": avg(rows, "progress"),
        "avg_efficiency": avg(rows, "efficiency"),
        "avg_calls": avg(rows, "calls"),
        "avg_reference_calls": avg(rows, "reference_calls"),
        "avg_state_changes": avg(rows, "state_changes"),
        "avg_failure_calls": avg(rows, "failure_calls"),
        "budget_exhaustion_rate": avg_bool(rows, "budget_exhausted"),
        "avg_call_ratio": avg(rows, "call_ratio"),
        "avg_reference_call_ratio": avg(rows, "reference_call_ratio"),
        "failure_call_rate": avg(rows, "failure_call_share"),
        "avg_success_calls": avg([row for row in rows if row["solved"]], "calls"),
        "avg_num_turns": avg(rows, "num_turns"),
        "avg_input_tokens": avg(rows, "input_tokens"),
        "avg_output_tokens": avg(rows, "output_tokens"),
        "avg_total_tokens": avg(rows, "total_tokens"),
    }


def failures(rows: list[Json], keys: list[str]) -> list[Json]:
    output = []
    for group_key, values in group(rows, keys).items():
        counts: Counter[str] = Counter()
        for row in values:
            counts.update(row["failure_counter"])
        total = sum(counts.values())
        for failure, count in sorted(counts.items()):
            item = dict(zip(keys, group_key, strict=True))
            item.update({"failure": failure, "count": count, "share": count / total if total else 0.0})
            output.append(item)
    return output


def group(rows: list[Json], keys: list[str]) -> dict[tuple[Any, ...], list[Json]]:
    grouped: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        grouped[tuple(group_value(row, key) for key in keys)].append(row)
    return dict(sorted(grouped.items(), key=lambda item: tuple("" if part is None else str(part) for part in item[0])))


def group_value(row: Json, key: str) -> Any:
    if key in row:
        return row[key]
    return row["params_dict"].get(key)


def sorted_rows(rows: list[Json]) -> list[Json]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            str(row["mode"]),
            str(row["family"]),
            "" if row.get("seed") is None else str(row["seed"]),
            str(row["params"]),
        ),
    )


def avg(rows: list[Json], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def avg_bool(rows: list[Json], key: str) -> float:
    return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[Json], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
