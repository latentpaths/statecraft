# Evaluation and development

Run commands from the repository root.

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
See the [standard environment](../environments/standard_undertow/README.md) and
[PTC environment](../environments/ptc_undertow/README.md) for taskset and rollout settings.

## Recorded experiments

The [experiment guide](../experiments/blog/README.md) documents the matched grids,
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


## Development

```sh
uv run --locked pytest tests
uv build --package undertow
uv build --package standard-undertow
uv build --package ptc-undertow
```
