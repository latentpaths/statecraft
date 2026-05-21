# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""Generate blog figures from blog rollout results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import DEFAULT_CONFIG, config_result_dir, model_slug  # noqa: E402

Json = dict[str, Any]

MODE_COLORS = {"ptc": "#2A9D8F", "standard": "#E76F51"}
MODE_ALPHA = {"ptc": 0.9, "standard": 0.72}
TEXT_COLOR = "#263238"
MUTED_COLOR = "#607D8B"
GRID_COLOR = "#E6EAED"
SUCCESS_COLOR = "#2A9D8F"
PROGRESS_COLOR = "#3A5A78"
WARNING_COLOR = "#F2A65A"
MODEL_LABELS = {
    "qwen/qwen3.6-35b-a3b": "Qwen3.6-35B",
    "z-ai/glm-5": "GLM-5",
}
FAMILY_LABELS = {"hidden_dag": "Hidden DAG", "repair_chain": "Repair Chain"}
FAILURE_LABELS = {
    "repeated_call": "Repeated call",
    "unknown_state_reference": "Unknown state ref",
    "violated_constraint": "Constraint violation",
    "budget_exceeded": "Budget exceeded",
    "premature_submit": "Premature submit",
    "incorrect_value": "Incorrect value",
    "distractor_action": "Distractor action",
    "stale_state_reference": "Stale state ref",
    "schema_error": "Schema error",
    "invalid_action": "Invalid action",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot blog experiment results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="raw dir, config result dir, or one JSONL file")
    parser.add_argument("--output", type=Path, help="config-level figures dir override")
    parser.add_argument("--per-model", action="store_true", help="also write figures under each model directory")
    args = parser.parse_args(argv)

    input_path = args.input or config_result_dir(args.config)
    rows = read_rows(input_path)
    result_dir = infer_result_dir(input_path, args.config, rows)
    output_dir = args.output or result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#CFD8DC",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    write_figures(rows, output_dir)
    if args.per_model:
        for model in sorted({row["model"] for row in rows}):
            model_dir = result_dir / model_slug(model) / "figures"
            model_dir.mkdir(parents=True, exist_ok=True)
            write_figures([row for row in rows if row["model"] == model], model_dir)

    print(f"output_dir={output_dir}")
    return 0


def read_rows(path: Path) -> list[Json]:
    files = raw_files(path)
    rows: list[Json] = []
    for file in files:
        with file.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(normalize_row(json.loads(line)))
    return rows


def raw_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path]
    if path.name == "raw":
        return sorted(path.glob("*.jsonl"))
    return sorted(path.glob("*/raw/*.jsonl"))


def infer_result_dir(input_path: Path, config_path: Path, rows: list[Json]) -> Path:
    path = input_path.resolve()
    if not input_path.exists():
        return config_result_dir(config_path)
    if path.is_file():
        path = path.parent
    if path.name == "raw":
        parent = path.parent
        models = {model_slug(str(row["model"])) for row in rows}
        if parent.name in models:
            return parent.parent
        return parent
    return path


def write_figures(rows: list[Json], output_dir: Path) -> None:
    plot_headline_comparison(rows, output_dir / "01_headline_comparison.png")
    plot_progress_gap(rows, output_dir / "02_progress_vs_success.png")
    plot_failure_taxonomy(rows, output_dir / "03_failure_taxonomy.png")
    plot_call_efficiency(rows, output_dir / "04_call_efficiency.png")
    plot_hidden_dag_heatmap(rows, output_dir / "05_hidden_dag_heatmap.png")
    plot_repair_chain_heatmap(rows, output_dir / "06_repair_chain_heatmap.png")
    plot_calls_distribution(rows, output_dir / "07_calls_distribution.png")
    plot_budget_exhaustion(rows, output_dir / "08_budget_exhaustion.png")


def normalize_row(row: Json) -> Json:
    out = dict(row)
    out["solved"] = bool(row["solved"])
    out["calls"] = float(row["calls"])
    out["budget"] = float(row["budget"])
    out["progress"] = float(row["progress"])
    out["state_changes"] = float(row["state_changes"])
    out["failure_calls"] = float(row["failure_calls"])
    out["params_dict"] = json.loads(row["params"])
    total = out["state_changes"] + out["failure_calls"]
    out["useful_rate"] = out["state_changes"] / total if total else 0.0
    out["budget_exhausted"] = out["calls"] >= out["budget"]
    return out


def group(rows: list[Json], keys: list[str]) -> dict[tuple[Any, ...], list[Json]]:
    grouped: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(sorted(grouped.items()))


def avg(rows: list[Json], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def label_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.split("/")[-1].replace("--", "/"))


def label_mode(mode: str) -> str:
    return "PTC" if mode == "ptc" else "Standard"


def present_modes(rows: list[Json]) -> list[str]:
    modes = {row["mode"] for row in rows}
    return [mode for mode in ["ptc", "standard"] if mode in modes]


def soften_axes(ax) -> None:
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFD8DC")
    ax.spines["bottom"].set_color("#CFD8DC")


def rate_axis(ax, ylabel: str = "Rate") -> None:
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    soften_axes(ax)


def _sorted_param_values(rows: list[Json], key: str) -> list[Any]:
    return sorted({row["params_dict"][key] for row in rows if key in row["params_dict"]})


def _param_heatmap_matrix(
    rows: list[Json],
    *,
    mode: str,
    row_key: str,
    col_key: str,
    value_fn,
) -> tuple[np.ndarray, list[Any], list[Any]]:
    subset = [row for row in rows if row["mode"] == mode]
    row_vals = _sorted_param_values(subset, row_key)
    col_vals = _sorted_param_values(subset, col_key)
    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    counts = np.zeros_like(matrix)

    for row in subset:
        p = row["params_dict"]
        if row_key not in p or col_key not in p:
            continue
        ri = row_vals.index(p[row_key])
        ci = col_vals.index(p[col_key])
        value = float(value_fn(row))
        matrix[ri, ci] = (0.0 if np.isnan(matrix[ri, ci]) else matrix[ri, ci]) + value
        counts[ri, ci] += 1

    result = np.zeros_like(matrix)
    with np.errstate(invalid="ignore"):
        np.divide(matrix, counts, out=result, where=counts > 0)
    return result, row_vals, col_vals


def _render_param_heatmaps(
    rows: list[Json],
    path: Path,
    *,
    row_key: str,
    col_key: str,
    row_label: str,
    col_label: str,
    value_fn,
    title: str,
    cbar_label: str,
    cmap: str,
    text_fmt: str,
    text_threshold: float | None = None,
) -> None:
    modes = present_modes(rows)
    if not modes:
        return

    width = max(4.4 * len(modes) + 1.0, 5.4)
    fig = plt.figure(figsize=(width, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, len(modes) + 1, width_ratios=[*[1.0] * len(modes), 0.045])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(len(modes))]
    cax = fig.add_subplot(grid[0, len(modes)])
    im = None

    for ax, mode in zip(axes, modes, strict=True):
        matrix, row_vals, col_vals = _param_heatmap_matrix(
            rows, mode=mode, row_key=row_key, col_key=col_key, value_fn=value_fn
        )
        if not row_vals or not col_vals:
            ax.axis("off")
            continue

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(col_vals)), labels=[str(v) for v in col_vals])
        ax.set_yticks(range(len(row_vals)), labels=[str(v) for v in row_vals])
        ax.set_xlabel(col_label)
        ax.set_ylabel(row_label if ax is axes[0] else "")
        ax.set_title(label_mode(mode), pad=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color("#D5DDE1")
            spine.set_linewidth(1.0)
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                color = "white" if text_threshold is not None and val > text_threshold else TEXT_COLOR
                ax.text(j, i, text_fmt.format(val), ha="center", va="center", color=color, fontsize=11.5, fontweight="semibold")

    if im is None:
        cax.axis("off")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    cbar.outline.set_edgecolor("#CFD8DC")
    fig.suptitle(title, fontsize=14, fontweight="semibold", color=TEXT_COLOR)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_headline_comparison(rows: list[Json], path: Path) -> None:
    """Solve rate and avg progress by mode, family, and model."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    metrics = [("solved", "Solve rate"), ("progress", "Avg progress")]
    modes = present_modes(rows)

    for ax, (key, ylabel) in zip(axes, metrics, strict=True):
        grouped = group(rows, ["family", "model", "mode"])
        families = [family for family in ["hidden_dag", "repair_chain"] if any(row["family"] == family for row in rows)]
        models = sorted({row["model"] for row in rows})
        x = np.arange(len(families))
        series_count = max(1, len(models) * len(modes))
        width = min(0.74 / series_count, 0.22)
        offsets = np.linspace(-width * (series_count - 1) / 2, width * (series_count - 1) / 2, series_count)

        series_idx = 0
        for idx, model in enumerate(models):
            for mode in modes:
                values = []
                for family in families:
                    subset = grouped.get((family, model, mode), [])
                    if key == "solved":
                        values.append(avg(subset, "solved") if subset else 0.0)
                    else:
                        values.append(avg(subset, "progress") if subset else 0.0)
                label = f"{label_model(model)} · {label_mode(mode)}"
                ax.bar(
                    x + offsets[series_idx],
                    values,
                    width,
                    label=label,
                    color=MODE_COLORS[mode],
                    alpha=MODE_ALPHA[mode],
                    edgecolor="white",
                    linewidth=0.8,
                )
                series_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in families])
        ax.set_title(ylabel)
        rate_axis(ax, "Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("Standard vs PTC: solve rate and progress", fontsize=14, fontweight="semibold", y=0.99)
    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.06)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_progress_gap(rows: list[Json], path: Path) -> None:
    """Highlight repair_chain episodes that finish the state machine but not exact solve."""
    repair = [r for r in rows if r["family"] == "repair_chain"]
    grouped = group(repair, ["mode", "model"])

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    labels = []
    solve_vals = []
    progress_vals = []
    colors_solve = []
    colors_progress = []

    for (mode, model), subset in grouped.items():
        labels.append(f"{label_model(model)}\n{label_mode(mode)}")
        solve_vals.append(avg(subset, "solved"))
        progress_vals.append(avg(subset, "progress"))
        colors_solve.append(MODE_COLORS[mode])
        colors_progress.append(MODE_COLORS[mode])

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, progress_vals, width, label="Avg progress", color=PROGRESS_COLOR, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, solve_vals, width, label="Solve rate", color=WARNING_COLOR, alpha=0.9, edgecolor="white", linewidth=0.8)

    for i, (p, s) in enumerate(zip(progress_vals, solve_vals, strict=True)):
        if p > 0.5 and s < 0.05:
            ax.annotate(
                "state done,\nsubmit failed",
                xy=(i - width / 2, p),
                xytext=(i - 0.55, p + 0.08),
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#555", "lw": 0.8},
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    rate_axis(ax, "Rate")
    ax.set_title("Repair chain: progress vs exact solve")
    ax.legend(loc="upper right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_failure_taxonomy(rows: list[Json], path: Path) -> None:
    """Stacked horizontal bars of failure shares by mode and family."""
    grouped = group(rows, ["mode", "family"])
    keys = list(grouped.keys())
    top_failures = top_failure_codes(rows, n=8)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    if not keys or not top_failures:
        ax.axis("off")
        ax.set_title("Failure taxonomy by task family and mode")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    y = np.arange(len(keys))
    left = np.zeros(len(keys))

    cmap = plt.colormaps["tab10"]
    for idx, failure in enumerate(top_failures):
        widths = []
        for key in keys:
            counts: Counter[str] = Counter()
            for row in grouped[key]:
                counts.update(row["failure_counter"])
            total = sum(counts.values())
            widths.append(counts[failure] / total if total else 0.0)
        ax.barh(
            y,
            widths,
            left=left,
            label=FAILURE_LABELS.get(failure, failure),
            color=cmap(idx / max(1, len(top_failures) - 1)),
            edgecolor="white",
            linewidth=0.4,
        )
        left += np.array(widths)

    ylabels = []
    for mode, family in keys:
        ylabels.append(f"{FAMILY_LABELS[family]} · {label_mode(mode)}")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Share of failure-tagged calls")
    ax.set_title("Failure taxonomy by task family and mode")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    soften_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=min(4, len(top_failures)))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_call_efficiency(rows: list[Json], path: Path) -> None:
    """Useful vs wasted call share."""
    grouped = group(rows, ["family", "model", "mode"])
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    labels = [f"{FAMILY_LABELS[k[0]]}\n{label_model(k[1])}\n{label_mode(k[2])}" for k in keys]

    useful = [avg(grouped[k], "useful_rate") for k in keys]
    wasted = [1.0 - u for u in useful]

    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    x = np.arange(len(keys))
    ax.bar(x, useful, label="State-changing calls", color=SUCCESS_COLOR, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.bar(x, wasted, bottom=useful, label="Failure / non-state calls", color=WARNING_COLOR, alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    rate_axis(ax, "Share of tool calls")
    ax.set_title("Call efficiency")
    ax.legend(loc="upper right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_dag_heatmap(rows: list[Json], path: Path) -> None:
    """Solve rate heatmap over depth × n_core for PTC vs standard."""
    subset = [r for r in rows if r["family"] == "hidden_dag"]
    _render_param_heatmaps(
        subset,
        path,
        row_key="depth",
        col_key="n_core",
        row_label="depth",
        col_label="n_core",
        value_fn=lambda row: float(row["solved"]),
        title="Hidden DAG: difficulty vs solve rate (averaged over seeds & distractors)",
        cbar_label="Solve rate",
        cmap="YlGn",
        text_fmt="{:.0%}",
        text_threshold=0.62,
    )


def plot_repair_chain_heatmap(rows: list[Json], path: Path) -> None:
    """Progress heatmap over n_objects × n_repair_steps."""
    subset = [r for r in rows if r["family"] == "repair_chain"]
    _render_param_heatmaps(
        subset,
        path,
        row_key="n_objects",
        col_key="n_repair_steps",
        row_label="objects",
        col_label="steps",
        value_fn=lambda row: float(row["progress"]),
        title="Repair chain: task size vs progress (averaged over seeds & distractors)",
        cbar_label="Avg progress",
        cmap="PuBu",
        text_fmt="{:.2f}",
        text_threshold=0.58,
    )


def plot_calls_distribution(rows: list[Json], path: Path) -> None:
    """Box plot of tool calls by mode and family."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True, constrained_layout=True)

    for ax, family in zip(axes, ["hidden_dag", "repair_chain"], strict=True):
        data = []
        positions = []
        tick_labels = []
        box_modes = []
        pos = 0
        for model in sorted({r["model"] for r in rows}):
            for mode in ["ptc", "standard"]:
                subset = [r["calls"] for r in rows if r["family"] == family and r["model"] == model and r["mode"] == mode]
                if subset:
                    data.append(subset)
                    positions.append(pos)
                    tick_labels.append(f"{label_model(model)}\n{label_mode(mode)}")
                    box_modes.append(mode)
                    pos += 1

        if not data:
            ax.axis("off")
            ax.set_title(FAMILY_LABELS[family])
            continue

        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, mode in zip(bp["boxes"], box_modes, strict=True):
            patch.set_facecolor(MODE_COLORS[mode])
            patch.set_alpha(MODE_ALPHA[mode])
            patch.set_edgecolor("#455A64")
            patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_title(FAMILY_LABELS[family])
        ax.set_ylabel("Tool calls per episode")
        soften_axes(ax)

    fig.suptitle("Distribution of tool-call budget usage", fontsize=14, fontweight="semibold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_exhaustion(rows: list[Json], path: Path) -> None:
    grouped = group(rows, ["family", "model", "mode"])
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    rates = [sum(1 for r in grouped[k] if r["budget_exhausted"]) / len(grouped[k]) for k in keys]
    labels = [f"{FAMILY_LABELS[k[0]]}\n{label_model(k[1])}\n{label_mode(k[2])}" for k in keys]

    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    colors = [MODE_COLORS[k[2]] for k in keys]
    alphas = [MODE_ALPHA[k[2]] for k in keys]
    bars = ax.bar(range(len(keys)), rates, color=colors, edgecolor="white", linewidth=0.8)
    for bar, alpha in zip(bars, alphas, strict=True):
        bar.set_alpha(alpha)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=0)
    rate_axis(ax, "Budget exhaustion rate")
    ax.set_title("Episodes that exhausted the tool-call budget")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def top_failure_codes(rows: list[Json], n: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["failure_counter"])
    return [code for code, _ in counts.most_common(n)]


if __name__ == "__main__":
    raise SystemExit(main())
