# Undertow Blog Experiments

Experiment grids for the Undertow writeup. Each config compares `standard-undertow` and `ptc-undertow` Verifiers environments on the same generated task settings.

Grid definitions live in `configs/` and results are written to a hybrid config/model layout:

```
results/<config>/
├── csvs/             # cross-model paper tables (written by aggregate.py)
├── figures/          # cross-model plots (written by plot.py)
└── <model-slug>/
    ├── raw/          # normalized JSONL files (written by run.py)
    ├── csvs/         # per-model tables (written by aggregate.py)
    └── figures/      # optional per-model plots (written by plot.py --per-model)
```

The raw directory is derived automatically from the config filename and model name. For example, `configs/medium.json` and `deepseek/deepseek-v4-flash` write to `results/medium/deepseek--deepseek-v4-flash/raw/`.

## Commands

```bash
# Run a grid
uv run python experiments/blog/run.py --config experiments/blog/configs/medium.json --model <model>

# Dry-run preview (limit 2, no execution)
uv run python experiments/blog/run.py --model <model> --limit 2 --dry-run

# Aggregate all model raw results for a config into CSVs
uv run python experiments/blog/aggregate.py --config experiments/blog/configs/medium.json

# Plot config-level figures
uv run python experiments/blog/plot.py --config experiments/blog/configs/medium.json

# Also emit per-model figures
uv run python experiments/blog/plot.py --config experiments/blog/configs/medium.json --per-model
```

Omit `--config` on aggregate/plot to use `configs/medium.json`. Use `--input` to aggregate or plot an explicit JSONL file, one model `raw/` directory, or one config result directory.

## CSV outputs

`aggregate.py` writes the same research table set at the config level and for each model:

- `episodes.csv`: tidy episode rows with generator params expanded into columns
- `summary.csv`: global totals, headline metrics, call ratios, turn counts, and token averages
- `by_config_name.csv`: grouped by config name for combined inputs
- `by_model.csv`, `by_mode.csv`, `by_family.csv`
- `by_model_mode.csv`, `by_model_family.csv`, `by_model_mode_family.csv`
- `by_config.csv`: grouped by model, mode, family, and exact params
- `by_params.csv`: grouped by family and expanded parameter columns
- `by_seed.csv`: grouped by model, mode, family, params, and seed
- `failures.csv`, `failure_by_config.csv`: failure taxonomy counts and shares

## Grid definitions

Available grids are `easy.json`, `medium.json`, `hard.json`, and `v_hard.json`.
All compare `standard` and `ptc`; `medium.json` is the default because it has two
seeds and moderate task sizes.
