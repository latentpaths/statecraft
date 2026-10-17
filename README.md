# Undertow

Hidden, stateful environments for comparing native and programmatic tool calling.
Undertow generates synthetic tasks whose hidden state and success conditions
are controlled, so agent traces can be examined for dependency, bookkeeping,
and stale-observation failures.

## What is included

| Task family | What the agent must do |
| --- | --- |
| `hidden_dag` | Discover intermediate values through tools and compute a target in dependency order |
| `repair_chain` | Inspect and repair machines while tracking locations, tools, and observation versions |

The core package supplies generation, runtime semantics, oracle runs, datasets,
and metrics. Two standalone Verifiers packages expose the same tasks through
native tools (`standard-undertow`) or a Python workspace (`ptc-undertow`).

## Installation

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/stochi0/undertow.git
cd undertow
uv sync --locked --group dev
```

The core CLI and its oracle checks require no inference credentials.
Dataset export additionally needs the `datasets` extra:
`uv sync --locked --extra datasets --group dev`.

## An offline smoke check

Generate a task, solve it with the reference oracle, and summarize the record:

```sh
uv run undertow generate --family hidden_dag --seed 1 --output /tmp/undertow-task.json
uv run undertow run --task /tmp/undertow-task.json --mode oracle --output /tmp/undertow-results.jsonl
uv run undertow summarize --input /tmp/undertow-results.jsonl
```

Replace `hidden_dag` with `repair_chain` to exercise the second family.
Result files are append-only; use a fresh output path for an independent run.
Oracle success checks task mechanics, not LLM performance.

Export a generated Hugging Face dataset locally:

```sh
uv run undertow dataset --mode standard --family hidden_dag \
  --max-examples 100 --output /tmp/undertow-hidden-dag
```

## Model evaluations

Real evaluations require the [Prime CLI](https://docs.primeintellect.ai/),
configured inference access, and the workspace environment packages:

```sh
uv sync --locked --all-packages --group dev
uv run prime eval run standard-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
uv run prime eval run ptc-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
```

These commands make model calls and may create sandboxes, consuming provider
credits. They are separate from the offline oracle checks above.
See the [standard environment](environments/standard_undertow/README.md) and
[PTC environment](environments/ptc_undertow/README.md) for taskset and rollout settings.

## Recorded experiments

The [experiment guide](experiments/blog/README.md) documents the matched grids,
retained measurements, aggregation, and plotting. The
[writeup](https://latentpaths.com/writings/undertow) explains the task design and
observed failure modes. These are synthetic environment experiments, not evidence
of general tool-use reliability.

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

## Repository guide

| Location | Contents |
| --- | --- |
| [src/undertow/core/](src/undertow/core/) | Task families, hidden-state runtime, datasets, and metrics |
| [src/undertow/cli.py](src/undertow/cli.py) | Task generation, oracle runs, dataset export, and summaries |
| [environments/](environments/) | Native-tool and programmatic-tool Verifiers packages |
| [experiments/blog/](experiments/blog/) | Experiment grids, recorded results, and analysis scripts |
| [tests/](tests/) | Core semantics, metrics, CLI, and experiment checks |

## Development

```sh
uv run --locked pytest tests
uv build --package undertow
uv build --package standard-undertow
uv build --package ptc-undertow
```

## License

Public source availability does not grant permission to use this project.
The existing [license](LICENSE) reserves all rights and requires prior written
permission for use, reproduction, modification, distribution, benchmarking,
evaluation, dataset generation, model training, or derivative works.
Contact [ayush@latentpaths.com](mailto:ayush@latentpaths.com) to request permission.
