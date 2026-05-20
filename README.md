# Undertow

Undertow is a controllable benchmark for comparing standard tool-calling agents
with programmatic tool-calling agents in hidden stateful environments.

Implemented v0 families:

- Hidden DAG Computation
- Stateful Repair Chains

Basic local smoke run:

```bash
undertow generate --family hidden_dag --seed 1 --output /tmp/hidden.json
undertow run --task /tmp/hidden.json --mode oracle --output /tmp/results.jsonl
undertow summarize --input /tmp/results.jsonl
```

Generate a Hugging Face `datasets` package on disk:

```bash
undertow dataset \
  --mode standard \
  --family hidden_dag \
  --max-examples 100 \
  --output /tmp/undertow-hidden-dag
```

Model evaluations run through the standalone Verifiers environment packages:

```bash
uv run prime eval run standard-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'

uv run prime eval run ptc-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
```

Blog experiment grids (standard vs PTC comparison, aggregation, figures) live in
[`experiments/blog/`](experiments/blog/). Each config file gets its own results
directory under `experiments/blog/results/<config-stem>/`.

```bash
uv run python experiments/blog/run.py --model openai/gpt-5.5
uv run python experiments/blog/aggregate.py --input experiments/blog/results/configs/raw
uv run experiments/blog/plot.py --input experiments/blog/results/configs/raw
```

The core `undertow` CLI only generates tasks/datasets, runs local oracle smoke
checks, and summarizes result JSONL files.

## Development

From the repo root:

```bash
uv sync --group dev
uv run pytest tests
uv build --package undertow
uv build --package standard-undertow
uv build --package ptc-undertow
```

## Package Layout

Undertow keeps task generation and environment semantics independent from the
framework used to run them. The core `undertow` package contains task families,
runtime primitives, dataset helpers, metrics, and the CLI:

- `undertow.core.datasets` builds JSON-serializable records and Hugging Face
  `Dataset` objects for generated Undertow tasks.
- `standard-undertow` and `ptc-undertow` are standalone Prime Verifiers
  environment packages that depend on the core `undertow` package.
- Verifiers environment implementation lives in the standalone environment
  packages under `environments`.
- Future integrations can reuse the same `UndertowTask`, `UndertowEnv`, native
  tool schemas, dataset records, and metrics without taking over rollout loops.

Dataset imports:

```python
from undertow import generate_hf_dataset, save_hf_dataset

dataset = generate_hf_dataset(mode="standard", family="hidden_dag", max_examples=100)
save_hf_dataset("/tmp/undertow-hidden-dag", mode="ptc", family="hidden_dag", max_examples=100)
```

Verifiers environment usage:

```python
import ptc_undertow
import standard_undertow

env = standard_undertow.load_environment(
    {"taskset": {"family": "hidden_dag", "max_examples": 5}}
)
ptc_env = ptc_undertow.load_environment(
    {"taskset": {"family": "hidden_dag", "max_examples": 5}}
)
```

```bash
uv run prime eval run standard-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'

uv run prime eval run ptc-undertow \
  -a '{"config":{"taskset":{"family":"hidden_dag","max_examples":5}}}'
```
Blog experiment code: https://github.com/stochi0/undertow/tree/4501d002420f9838590d0133868be1fc6d79bf01
