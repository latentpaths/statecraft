<p>
  <a href="https://www.latentpaths.com/"><img src="https://raw.githubusercontent.com/latentpaths/.github/main/assets/latent-paths-wordmark.png" alt="Latent Paths" width="280"></a>
</p>

# Statecraft

How does tool access change an agent's ability to track hidden state? Statecraft
provides synthetic tasks with controlled dependencies and success conditions,
exposed through native tools or a Python workspace. Recorded runs let you inspect
dependency errors, bookkeeping failures, and decisions based on stale observations.

[Task environments](environments/) · [Recorded experiments](experiments/blog/README.md) · [Writeup](https://latentpaths.com/writings/statecraft)

[Try a task](#try-a-task-offline) · [Evaluate models](#evaluate-models) · [Develop](#development)

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

## Export a dataset

Run commands from the repository root. Export a generated Hugging Face dataset locally:

```sh
uv run statecraft dataset --mode standard --family hidden_dag \
  --max-examples 100 --output /tmp/statecraft-hidden-dag
```

## Evaluate models

Real evaluations require the [Prime CLI](https://docs.primeintellect.ai/),
configured inference access, and the workspace environment packages:

```sh
uv sync --locked --all-packages --group dev
uv run prime eval run standard-statecraft \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
uv run prime eval run ptc-statecraft \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
```

These commands make model calls and may create sandboxes, consuming provider
credits. They are separate from the offline oracle checks above.
See the [standard environment](environments/standard_statecraft/README.md) and
[PTC environment](environments/ptc_statecraft/README.md) for taskset and rollout settings.

### Preview and inspect experiment grids

Preview an experiment grid without launching evaluations:

```sh
uv run python experiments/blog/run.py \
  --config experiments/blog/configs/medium.json \
  --model openai/gpt-5.5 --limit 2 --dry-run
```

The preview writes local output directories but makes no inference calls.
Retained normalized episode records are under
`experiments/blog/results/<config>/<model>/raw/`; CSV summaries and figures are
retained alongside them. Historical logs describe the original runs and do not
imply that a provider, model, or sandbox setup remains available unchanged.

Aggregation validates the current normalized schema. Some retained legacy rows
predate that schema and are rejected; keep those records unchanged and aggregate
supported rows into a separate output directory.

## Development

```sh
uv run --locked pytest tests
uv build --package statecraft
uv build --package standard-statecraft
uv build --package ptc-statecraft
```

## License

The [existing license](LICENSE) reserves all rights. Use, reproduction,
modification, distribution, benchmarking, evaluation, dataset generation, model
training, and derivative works require prior written permission.
Contact [Ayush Bodade](mailto:ayush@latentpaths.com).
