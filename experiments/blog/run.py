from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from undertow.core.types import FailureCode

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import raw_dir_for_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Undertow blog experiment grid.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    raw_dir = raw_dir_for_config(args.config)
    jobs = list(expand_grid(config, args.model))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"raw_dir={raw_dir}")

    for job in jobs:
        command = prime_command(job)
        print(" ".join(command))
        if args.dry_run:
            continue
        started = time.time()
        subprocess.run(command, cwd=ROOT, check=True)
        result_path = latest_results(job["env_name"], job["model"], started)
        rows = [normalize_prime_row(row, job) for row in read_jsonl(result_path)]
        write_jsonl(raw_path(job, raw_dir), rows)
    return 0


def expand_grid(config: dict[str, Any], model: str) -> list[dict[str, Any]]:
    jobs = []
    for mode in config["modes"]:
        env_name = f"{mode}-undertow"
        for family, params_grid in config["families"].items():
            keys = list(params_grid)
            for values in itertools.product(*(params_grid[key] for key in keys)):
                task_params = dict(zip(keys, values, strict=True))
                for seed in config["seeds"]:
                    jobs.append(
                        {
                            "mode": mode,
                            "env_name": env_name,
                            "family": family,
                            "seed": seed,
                            "model": model,
                            "task_params": task_params,
                        }
                    )
    return jobs


def prime_command(job: dict[str, Any]) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run Prime evals")
    payload = {
        "config": {
            "taskset": {
                "family": job["family"],
                "seed": job["seed"],
                "max_examples": 1,
                "task_params": job["task_params"],
            }
        }
    }
    return [
        uv,
        "run",
        "prime",
        "eval",
        "run",
        job["env_name"],
        "--model",
        job["model"],
        "-a",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    ]


def latest_results(env_name: str, model: str, started: float) -> Path:
    model_slug = model.replace("/", "--")
    candidates = []
    for root in [ROOT / "outputs" / "evals", ROOT / "environments" / "outputs" / "evals"]:
        candidates.extend(root.glob(f"{env_name}--{model_slug}/*/results.jsonl"))
    candidates.extend((ROOT / "environments").glob(f"*/outputs/evals/{env_name}--{model_slug}/*/results.jsonl"))
    fresh = [path for path in candidates if path.stat().st_mtime >= started - 1]
    if not fresh:
        raise FileNotFoundError(f"no results.jsonl found for {env_name} {model}")
    return max(fresh, key=lambda path: path.stat().st_mtime)


def normalize_prime_row(row: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    prefix = "ptc" if job["mode"] == "ptc" else "undertow"
    failure_counter = {
        failure.value: int(metrics[f"{prefix}_failure_{failure.value}"])
        for failure in FailureCode
        if int(metrics[f"{prefix}_failure_{failure.value}"])
    }
    return {
        "mode": job["mode"],
        "family": job["family"],
        "model": job["model"],
        "params": json.dumps(job["task_params"], sort_keys=True, separators=(",", ":")),
        "solved": bool(metrics[f"{prefix}_solved"] == 1),
        "reward": float(row["reward"]),
        "calls": float(metrics[f"{prefix}_calls"]),
        "budget": float(metrics[f"{prefix}_budget"]),
        "reference_calls": float(metrics[f"{prefix}_reference_calls"]),
        "efficiency": float(metrics[f"{prefix}_efficiency"]),
        "progress": float(metrics[f"{prefix}_progress"]),
        "state_changes": float(metrics[f"{prefix}_state_changes"]),
        "failure_calls": float(metrics[f"{prefix}_failure_calls"]),
        "failure_counter": failure_counter,
        "first_failure": None,
        "final_failure": None,
    }


def raw_path(job: dict[str, Any], raw_dir: Path) -> Path:
    params = json.dumps(job["task_params"], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(params.encode()).hexdigest()[:10]
    model = job["model"].replace("/", "--")
    return raw_dir / (
        f"mode-{job['mode']}__family-{job['family']}__model-{model}"
        f"__seed-{job['seed']}__params-{digest}.jsonl"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
