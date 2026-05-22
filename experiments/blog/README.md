# Undertow Blog Experiments

Experiment grids for the Undertow writeup. Each config compares `standard-undertow` and `ptc-undertow` Verifiers environments on the same generated task settings.

Grid definitions live in `configs/` and results are written to a hybrid config/model layout:

```
results/<config>/
├── csvs/             # cross-model paper tables (written by aggregate.py)
├── figures/          # cross-model plots (written by plot.py)
└── <model-slug>/
    ├── <run-id>/raw/ # timestamped normalized JSONL files (written by run.py)
    ├── csvs/         # per-model tables (written by aggregate.py)
    └── figures/      # optional per-model plots (written by plot.py --per-model)
```

The raw directory is derived automatically from the config filename, model name,
and a timestamped run id. For example, `configs/medium.json` and
`deepseek/deepseek-v4-flash` write to
`results/medium/deepseek--deepseek-v4-flash/<run-id>/raw/`. Logs use the same
run id under `logs/medium/deepseek--deepseek-v4-flash/<run-id>/`.

## Commands

```bash
# Run a grid
uv run python experiments/blog/run.py --config experiments/blog/configs/medium.json --model <model>

# Dry-run preview (limit 2, no execution)
uv run python experiments/blog/run.py --model <model> --limit 2 --dry-run

# Aggregate all model raw results for a config into CSVs
uv run python experiments/blog/aggregate.py --config experiments/blog/configs/medium.json

# Plot config-level figures (uses inline script deps via uv)
uv run experiments/blog/plot.py --config experiments/blog/configs/medium.json

# Also emit per-model figures
uv run experiments/blog/plot.py --config experiments/blog/configs/medium.json --per-model

# Aggregate + plot all grid configs (easy, medium, hard, v_hard)
./experiments/blog/aggregate_and_plot.sh
./experiments/blog/aggregate_and_plot.sh --per-model
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

## Run status

Last checked: 2026-05-22. Batch grid across three models (`deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`, `qwen/qwen3.6-35b-a3b`) and three configs (`easy`, `medium`, `hard`), launched in nine tmux sessions. 45/48 jobs complete. No sessions are currently executing.

| tmux session | model | config | jobs |
|---|---|---|---|
| `undertow-ds-flash-easy` | deepseek-v4-flash | easy | 4/4 |
| `undertow-ds-pro-easy` | deepseek-v4-pro | easy | 4/4 |
| `undertow-qwen-easy` | qwen3.6-35b-a3b | easy | 4/4 |
| `undertow-ds-flash-medium` | deepseek-v4-flash | medium | 8/8 |
| `undertow-ds-pro-medium` | deepseek-v4-pro | medium | 8/8 |
| `undertow-qwen-medium` | qwen3.6-35b-a3b | medium | 8/8 |
| `undertow-ds-flash-hard` | deepseek-v4-flash | hard | 3/4 |
| `undertow-ds-pro-hard` | deepseek-v4-pro | hard | 3/4 |
| `undertow-qwen-hard` | qwen3.6-35b-a3b | hard | 3/4 |

### Missing jobs

All three hard sessions failed on the same job:

- `mode-ptc__family-repair_chain__seed-0__params-178938bca8`

Logs are under `logs/hard/<model-slug>/`. Failure: sandbox `CommandTimeoutError` after 600s during PTC repair-chain rollout. Raw results were not written for these runs.
