<p>
  <a href="https://www.latentpaths.com/"><img src="https://raw.githubusercontent.com/latentpaths/.github/main/assets/latent-paths-wordmark.png" alt="Latent Paths" width="280"></a>
</p>

# Statecraft

How does tool access change an agent's ability to track hidden state? Statecraft
provides synthetic tasks with controlled dependencies and success conditions,
exposed through native tools or a Python workspace. Recorded runs let you inspect
dependency errors, bookkeeping failures, and decisions based on stale observations.

[Task environments](environments/) · [Recorded experiments](experiments/blog/README.md) · [Writeup](https://latentpaths.com/writings/statecraft)

## Try a task offline

With Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/latentpaths/statecraft.git
cd statecraft
uv sync --locked --group dev
uv run statecraft generate --family hidden_dag --seed 1 \
  --output /tmp/statecraft-task.json
uv run statecraft run --task /tmp/statecraft-task.json --mode oracle \
  --output /tmp/statecraft-results.jsonl
uv run statecraft summarize --input /tmp/statecraft-results.jsonl
```

This generates a task, solves it with the reference oracle, and summarizes the
record. Use a fresh output path for each independent run: records are append-only.
Oracle success checks task mechanics; it does not measure LLM performance.

## What the agent does

| Task family | Required actions |
| --- | --- |
| `hidden_dag` | Discover intermediate values and compute a target in dependency order |
| `repair_chain` | Inspect and repair machines while tracking locations, tools, and observation versions |

The core package implements task generation, hidden-state transitions, oracle
runs, datasets, and metrics. Two Verifiers packages expose those tasks:
[standard-statecraft](environments/standard_statecraft/README.md) uses native tools;
[ptc-statecraft](environments/ptc_statecraft/README.md) uses a Python workspace.

## Evaluate and inspect

The [experiment guide](experiments/blog/README.md) explains matched grids,
retained records, aggregation, and plots. These synthetic comparisons measure
performance under the recorded task and execution settings; they do not establish
general tool-use reliability. Some hard-grid PTC runs timed out and have no raw
results, so inspect missing-run coverage before comparing summaries.

See [evaluation and development](docs/evaluation.md) for dataset export, offline
grid previews, model evaluation commands, and package checks. Real evaluations
require inference access and may create paid sandboxes.

## License

The [existing license](LICENSE) reserves all rights. Use, reproduction,
modification, distribution, benchmarking, evaluation, dataset generation, model
training, and derivative works require prior written permission.
Contact [Ayush Bodade](mailto:ayush@latentpaths.com).
