# Undertow Blog Experiments

This folder contains the experiment grids for the Undertow writeup. Each config
compares the `standard-undertow` and `ptc-undertow` Verifiers environments on the
same generated task settings.

## Configs

Grid definitions live in `configs/`:

- `configs/configs.json` — original blog grid (5 seeds, moderate difficulty)
- `configs/configs_hard.json` — single-seed stress test with large params,
  distractors, noisy observations, and partial/stale state

## Output layout

Each config gets its own results directory under `results/<config-stem>/`:

```
results/configs/
├── raw/              # one normalized JSONL file per run (written by run.py)
├── summary.csv       # aggregate by mode, family, and model
├── by_config.csv     # aggregate by mode, family, model, and generator params
├── failures.csv      # failure taxonomy counts and shares
└── figures/          # plots for the blog writeup (written by plot.py)

results/configs_hard/
├── raw/
├── summary.csv
├── by_config.csv
├── failures.csv
└── figures/
```

The raw directory is derived automatically from the config filename. For
example, `configs/configs_hard.json` writes to `results/configs_hard/raw/`.

## Run a grid

```bash
uv run python experiments/blog/run.py --model openai/gpt-5.5
```

Run the hard config:

```bash
uv run python experiments/blog/run.py \
  --model qwen/qwen3.6-35b-a3b \
  --config experiments/blog/configs/configs_hard.json
```

Preview commands without executing:

```bash
uv run python experiments/blog/run.py \
  --model openai/gpt-5.5 \
  --limit 2 \
  --dry-run
```

## Aggregate

Pass only the raw directory. CSVs are written to the parent directory
(`results/<config-stem>/`):

```bash
uv run python experiments/blog/aggregate.py \
  --input experiments/blog/results/configs/raw
```

```bash
uv run python experiments/blog/aggregate.py \
  --input experiments/blog/results/configs_hard/raw
```

Omit `--input` to use the default (`results/configs/raw`, from
`configs/configs.json`).

## Plot

Pass only the raw directory. Figures are written to
`results/<config-stem>/figures/`:

```bash
uv run experiments/blog/plot.py \
  --input experiments/blog/results/configs/raw
```

```bash
uv run experiments/blog/plot.py \
  --input experiments/blog/results/configs_hard/raw
```

Use `--output` on aggregate or plot to override the auto-generated destination.

## Grid definitions

`configs/configs.json`:

- modes: `standard`, `ptc`
- seeds: `0..4`
- `hidden_dag`: `n_core in {5, 10}`, `depth in {2, 5}`,
  `n_connected_distractors in {0, 5}`
- `repair_chain`: `n_objects in {1, 3}`, `n_repair_steps in {2, 3}`,
  `n_distractor_tools in {0, 2}`

`configs/configs_hard.json`:

- modes: `standard`, `ptc`
- seeds: `0`
- `hidden_dag`: `n_core=128`, `depth=32`, `n_connected_distractors=15`,
  `n_disconnected_distractors=10`, `observation_mode=minimal`,
  `invalid_return_mode=noisy`
- `repair_chain`: `n_objects=128`, `n_repair_steps=64`, `n_tools=16`,
  `n_distractor_tools=12`, `staleness_rate=0.5`,
  `partial_observation_rate=0.5`, `shared_tool_ambiguity=0.5`
