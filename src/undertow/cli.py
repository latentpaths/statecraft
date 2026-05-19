from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from undertow.core import UndertowTask, save_hf_dataset, push_hf_dataset
from undertow.core.metrics import append_jsonl, read_jsonl, summarize, write_summary_csv
from undertow.core.runtime import TASK_FAMILIES, env_for_task, generate_task, run_oracle, task_from_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="undertow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--family", choices=TASK_FAMILIES, required=True)
    generate_parser.add_argument("--seed", type=int, required=True)
    generate_parser.add_argument("--output", type=Path)
    generate_parser.add_argument("--params", default="{}")

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("--mode", choices=["standard", "ptc"], required=True)
    dataset_parser.add_argument("--family", choices=TASK_FAMILIES)
    dataset_parser.add_argument("--families", default=None)
    dataset_parser.add_argument("--seed", type=int, default=0)
    dataset_parser.add_argument("--max-examples", type=int, default=5)
    dataset_parser.add_argument("--shuffle", action="store_true")
    dataset_parser.add_argument("--params", default="{}")
    dataset_parser.add_argument("--output", type=Path)
    dataset_parser.add_argument("--push-to")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", type=Path, required=True)
    run_parser.add_argument("--mode", choices=["oracle"], default="oracle")
    run_parser.add_argument("--output", type=Path)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--input", type=Path, required=True)
    summarize_parser.add_argument("--csv", type=Path)

    args = parser.parse_args(argv)
    if args.command == "generate":
        task = _generate(args.family, args.seed, json.loads(args.params))
        _emit(task.to_dict(), args.output)
        return 0
    if args.command == "dataset":
        _save_or_push_dataset(
            mode=args.mode,
            family=args.family,
            families=_split_csv(args.families),
            seed=args.seed,
            max_examples=args.max_examples,
            shuffle=args.shuffle,
            task_params=json.loads(args.params),
            output=args.output,
            push_to=args.push_to,
        )
        return 0
    if args.command == "run":
        task = _load_task(args.task)
        env = _env_for_task(task)
        if args.mode == "oracle":
            result = _run_oracle(env).to_dict()
        else:
            raise AssertionError(args.mode)
        if args.output:
            append_jsonl(args.output, [result])
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        rows = summarize(read_jsonl(args.input))
        if args.csv:
            write_summary_csv(args.csv, rows)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def _generate(family: str, seed: int, params: dict[str, Any]) -> UndertowTask:
    return generate_task(family, seed=seed, params=params)


def _load_task(path: Path) -> UndertowTask:
    data = json.loads(path.read_text())
    return task_from_dict(data)


def _env_for_task(task: UndertowTask):
    return env_for_task(task)


def _run_oracle(env):
    return run_oracle(env)


def _save_or_push_dataset(
    *,
    mode: str,
    family: str | None,
    families: list[str] | None,
    seed: int,
    max_examples: int,
    shuffle: bool,
    task_params: dict[str, Any],
    output: Path | None,
    push_to: str | None,
) -> None:
    if output is None and push_to is None:
        raise SystemExit("one of --output or --push-to is required")
    kwargs = {
        "mode": mode,
        "family": family,
        "families": families,
        "seed": seed,
        "max_examples": max_examples,
        "shuffle": shuffle,
        "task_params": task_params,
    }
    if output is not None:
        save_hf_dataset(output, **kwargs)
    if push_to is not None:
        push_hf_dataset(push_to, **kwargs)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _emit(data: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    raise SystemExit(main())
