# Undertow Blog Experiments

This folder contains the fixed experiment grid for the Undertow writeup. It compares
the `standard-undertow` and `ptc-undertow` Verifiers environments on the same
generated task settings.

Run the full grid for a model:

```bash
uv run python experiments/undertow_blog/run.py --model openai/gpt-5.5
```

Run a quick command preview:

```bash
uv run python experiments/undertow_blog/run.py --model openai/gpt-5.5 --limit 2 --dry-run
```

Aggregate normalized raw rows:

```bash
uv run python experiments/undertow_blog/aggregate.py
```

Outputs:

- `results/raw/*.jsonl`: one normalized JSONL file per run.
- `results/summary.csv`: aggregate by mode, family, and model.
- `results/by_config.csv`: aggregate by mode, family, model, and generator params.
- `results/failures.csv`: failure taxonomy counts and shares.

The grid is defined in `configs.json`:

- modes: `standard`, `ptc`
- seeds: `0..4`
- `hidden_dag`: `n_core in {5, 10}`, `depth in {2, 5}`,
  `n_connected_distractors in {0, 5}`
- `repair_chain`: `n_objects in {1, 3}`, `n_repair_steps in {2, 3}`,
  `n_distractor_tools in {0, 2}`
