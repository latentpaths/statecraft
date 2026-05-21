# Undertow Blog Experiments

Experiment grids for the Undertow writeup. Each config compares `standard-undertow` and `ptc-undertow` Verifiers environments on the same generated task settings.

Grid definitions live in `configs/` and results are written to `results/<config-stem>/`:

```
results/<config>/
├── raw/              # normalized JSONL files (written by run.py)
├── summary.csv       # aggregate by mode, family, and model
├── by_config.csv     # aggregate by mode, family, model, and generator params
├── failures.csv      # failure taxonomy counts and shares
└── figures/          # plots (written by plot.py)
```

The raw directory is derived automatically from the config filename (e.g. `configs/configs_hard.json` writes to `results/configs_hard/raw/`).

## Commands

```bash
# Run a grid
uv run python experiments/blog/run.py --model <model> [--config <path>]

# Dry-run preview (limit 2, no execution)
uv run python experiments/blog/run.py --model <model> --limit 2 --dry-run

# Aggregate raw results into CSVs
uv run python experiments/blog/aggregate.py --input results/<config>/raw

# Plot figures from raw results
uv run python experiments/blog/plot.py --input results/<config>/raw
```

Omit `--input` on aggregate/plot to use `results/configs/raw` (from `configs/configs.json`). Use `--output` to override the auto-generated destination.

## Grid definitions

`configs/configs.json`:
- modes: `standard`, `ptc`
- seeds: `0..4`
- `hidden_dag`: `n_core in {5, 10}`, `depth in {2, 5}`, `n_connected_distractors in {0, 5}`
- `repair_chain`: `n_objects in {1, 3}`, `n_repair_steps in {2, 3}`, `n_distractor_tools in {0, 2}`

`configs/configs_hard.json`:
- modes: `standard`, `ptc`
- seeds: `0`
- `hidden_dag`: `n_core=128`, `depth=32`, `n_connected_distractors=15`, `n_disconnected_distractors=10`, `observation_mode=minimal`, `invalid_return_mode=noisy`
- `repair_chain`: `n_objects=128`, `n_repair_steps=64`, `n_tools=16`, `n_distractor_tools=12`, `staleness_rate=0.5`, `partial_observation_rate=0.5`, `shared_tool_ambiguity=0.5`
