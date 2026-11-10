# Datasets and model evaluation

Run commands from the repository root after the [offline quickstart](../README.md#try-a-task-offline).

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
See the [standard environment](../environments/standard_statecraft/README.md) and
[PTC environment](../environments/ptc_statecraft/README.md) for taskset and rollout settings.

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
