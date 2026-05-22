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
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import DEFAULT_CONFIG, config_result_dir, model_slug  # noqa: E402

Json = dict[str, Any]

MODE_COLORS = {"ptc": "#0072B2", "standard": "#D55E00"}
MODE_ALPHA = {"ptc": 0.92, "standard": 0.88}
TEXT_COLOR = "#22313A"
MUTED_COLOR = "#60717B"
GRID_COLOR = "#E7ECEF"
SPINE_COLOR = "#CBD5DA"
SUCCESS_COLOR = "#009E73"
REMAINDER_COLOR = "#DDE5EA"
PROGRESS_COLOR = "#6C8EBF"
SOLVE_COLOR = "#D55E00"
FAILURE_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F7F7F",
    "#332288",
    "#B8B8B8",
]
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
    "other": "Other",
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

    configure_style()
    write_figures(rows, output_dir)
    if args.per_model:
        for model in sorted({row["model"] for row in rows}):
            model_dir = result_dir / model_slug(model) / "figures"
            model_dir.mkdir(parents=True, exist_ok=True)
            write_figures([row for row in rows if row["model"] == model], model_dir)

    print(f"output_dir={output_dir}")
    return 0


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.75,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


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


def label_family(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())


def wrap_label(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def percent(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}%}"


def present_modes(rows: list[Json]) -> list[str]:
    modes = {row["mode"] for row in rows}
    return [mode for mode in ["ptc", "standard"] if mode in modes]


def present_families(rows: list[Json]) -> list[str]:
    families = {row["family"] for row in rows}
    return [family for family in ["hidden_dag", "repair_chain"] if family in families]


def present_models(rows: list[Json]) -> list[str]:
    return sorted({row["model"] for row in rows}, key=label_model)


def metric_value(rows: list[Json], key: str) -> float:
    if key == "solved":
        return avg(rows, "solved")
    return avg(rows, key)


def no_data_figure(path: Path, title: str, message: str = "No rows available for this figure.") -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    ax.set_facecolor("#FAFCFD")
    ax.text(0.5, 0.56, title, ha="center", va="center", fontsize=14, fontweight="semibold", color=TEXT_COLOR)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10.5, color=MUTED_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def soften_axes(ax, *, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.75)
    ax.grid(False, axis="y" if axis == "x" else "x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(length=0)


def rate_x_axis(ax, xlabel: str = "Rate") -> None:
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    soften_axes(ax, axis="x")


def rate_y_axis(ax, ylabel: str = "Rate") -> None:
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    soften_axes(ax, axis="y")


def horizontal_height(count: int, *, base: float = 2.1, per_row: float = 0.44, minimum: float = 4.0) -> float:
    return max(minimum, base + count * per_row)


def ordered_group_keys(rows: list[Json]) -> list[tuple[str, str, str]]:
    keys = []
    for family in present_families(rows):
        for model in present_models([row for row in rows if row["family"] == family]):
            for mode in present_modes([row for row in rows if row["family"] == family and row["model"] == model]):
                if any(row["family"] == family and row["model"] == model and row["mode"] == mode for row in rows):
                    keys.append((family, model, mode))
    return keys


def group_label(key: tuple[str, str, str], *, include_family: bool = True) -> str:
    family, model, mode = key
    parts = []
    if include_family:
        parts.append(label_family(family))
    parts.append(label_model(model))
    parts.append(label_mode(mode))
    return " | ".join(parts)


def group_label_tick(key: tuple[str, str, str]) -> str:
    """Two-line y-tick label: family | model, then mode on its own line."""
    family, model, mode = key
    head = " | ".join([label_family(family), label_model(model)])
    return f"{head}\n{label_mode(mode)}"


LEGEND_BELOW_AXES = -0.10
LEGEND_BELOW_FIG = 0.03


def bottom_legend(ax, *, ncol: int = 2, anchor_y: float = LEGEND_BELOW_AXES, **kwargs) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, anchor_y), ncol=ncol, **kwargs)


def add_mode_legend(fig, *, ncol: int = 2, anchor_y: float = LEGEND_BELOW_FIG) -> None:
    handles = [
        mlines.Line2D([], [], color=MODE_COLORS[mode], marker="o", linestyle="", markersize=8, label=label_mode(mode))
        for mode in ["ptc", "standard"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, anchor_y), ncol=ncol)


def reserve_bottom_margin(fig, *, bottom: float = 0.18, top: float = 0.90, left: float | None = None) -> None:
    kwargs: dict[str, float] = {"bottom": bottom, "top": top}
    if left is not None:
        kwargs["left"] = left
    fig.subplots_adjust(**kwargs)


def add_bar_end_labels(ax, bars, values: list[float], *, formatter: Callable[[float], str] = percent, min_value: float = 0.001) -> None:
    for bar, value in zip(bars, values, strict=True):
        if value < min_value:
            continue
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.annotate(
            formatter(value),
            xy=(x, y),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            color=TEXT_COLOR,
            clip_on=False,
        )


def plot_headline_comparison(rows: list[Json], path: Path) -> None:
    """Solve rate and avg progress by mode, family, and model."""
    if not rows:
        no_data_figure(path, "Standard vs PTC: solve rate and progress")
        return

    metrics = [("solved", "Solve rate"), ("progress", "Average progress")]
    families = present_families(rows)
    models = present_models(rows)
    grouped = group(rows, ["family", "model", "mode"])
    y_items = [
        (family, model)
        for family in families
        for model in models
        if any(row["family"] == family and row["model"] == model for row in rows)
    ]
    row_count = max(1, len(y_items))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, horizontal_height(row_count, per_row=0.5)), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    y_labels = [f"{label_family(family)} | {wrap_label(label_model(model), 18)}" for family, model in y_items]
    y_positions = list(range(len(y_items)))
    mode_offsets = {"ptc": 0.11, "standard": -0.11}

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for y, (family, model) in zip(y_positions, y_items, strict=True):
            values_by_mode = {}
            for mode in ["ptc", "standard"]:
                subset = grouped.get((family, model, mode), [])
                if subset:
                    values_by_mode[mode] = metric_value(subset, metric)
            if len(values_by_mode) == 2:
                ax.plot(
                    [values_by_mode["standard"], values_by_mode["ptc"]],
                    [y + mode_offsets["standard"], y + mode_offsets["ptc"]],
                    color="#B7C3CA",
                    linewidth=1.4,
                    zorder=1,
                )
            for mode, value in values_by_mode.items():
                ax.scatter(
                    value,
                    y + mode_offsets[mode],
                    s=86,
                    color=MODE_COLORS[mode],
                    alpha=MODE_ALPHA[mode],
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=3,
                )
        ax.set_title(title, pad=10)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        rate_x_axis(ax, "Rate")

    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    add_mode_legend(fig)
    fig.suptitle("Standard vs PTC: headline outcomes", fontsize=15, fontweight="semibold", y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.21, right=0.98, wspace=0.08)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_progress_gap(rows: list[Json], path: Path) -> None:
    """Highlight repair_chain episodes that finish state work but not exact solve."""
    repair = [r for r in rows if r["family"] == "repair_chain"]
    if not repair:
        no_data_figure(path, "Repair chain: progress vs exact solve", "No repair-chain rows available.")
        return

    grouped = group(repair, ["model", "mode"])
    keys: list[tuple[str, str]] = []
    for model in present_models(repair):
        for mode in present_modes([row for row in repair if row["model"] == model]):
            if grouped.get((model, mode)):
                keys.append((model, mode))

    labels = [f"{wrap_label(label_model(model), 20)} | {label_mode(mode)}" for model, mode in keys]
    progress_vals = [avg(grouped[key], "progress") for key in keys]
    solve_vals = [avg(grouped[key], "solved") for key in keys]
    y = np.arange(len(keys))
    height = 0.28

    fig, ax = plt.subplots(figsize=(10.4, horizontal_height(len(keys), per_row=0.55)), constrained_layout=False)
    progress_bars = ax.barh(
        y - height / 2,
        progress_vals,
        height,
        color=PROGRESS_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.8,
        label="Avg progress",
    )
    solve_bars = ax.barh(
        y + height / 2,
        solve_vals,
        height,
        color=SOLVE_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.8,
        label="Solve rate",
    )
    add_bar_end_labels(ax, progress_bars, progress_vals)
    add_bar_end_labels(ax, solve_bars, solve_vals)

    for idx, (progress, solved) in enumerate(zip(progress_vals, solve_vals, strict=True)):
        gap = progress - solved
        if gap >= 0.45:
            ax.annotate(
                f"gap {gap:.0%}",
                xy=(min(progress, 1.0), idx),
                xytext=(14, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color=MUTED_COLOR,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    rate_x_axis(ax, "Rate")
    ax.set_title("Repair chain: progress vs exact solve", pad=12)
    bottom_legend(ax, ncol=2)
    reserve_bottom_margin(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_failure_taxonomy(rows: list[Json], path: Path) -> None:
    """Stacked horizontal bars of failure shares by mode and family."""
    grouped = group(rows, ["family", "mode"])
    keys = [(family, mode) for family in present_families(rows) for mode in present_modes(rows) if grouped.get((family, mode))]
    top_failures = top_failure_codes(rows, n=6)
    if not keys or not top_failures:
        no_data_figure(path, "Failure taxonomy by task family and mode", "No failure-tagged calls available.")
        return

    failures = [*top_failures, "other"]
    y = np.arange(len(keys))
    left = np.zeros(len(keys))
    fig, ax = plt.subplots(figsize=(11.2, horizontal_height(len(keys), per_row=0.6)), constrained_layout=False)

    totals_by_key = {}
    counts_by_key = {}
    for key in keys:
        counts: Counter[str] = Counter()
        for row in grouped[key]:
            counts.update(row["failure_counter"])
        counts_by_key[key] = counts
        totals_by_key[key] = sum(counts.values())

    for idx, failure in enumerate(failures):
        widths = []
        for key in keys:
            counts = counts_by_key[key]
            total = totals_by_key[key]
            if not total:
                widths.append(0.0)
            elif failure == "other":
                widths.append(sum(count for code, count in counts.items() if code not in top_failures) / total)
            else:
                widths.append(counts[failure] / total)
        ax.barh(
            y,
            widths,
            left=left,
            label=FAILURE_LABELS.get(failure, failure),
            color=FAILURE_COLORS[idx % len(FAILURE_COLORS)],
            edgecolor="white",
            linewidth=0.5,
            height=0.66,
        )
        left += np.array(widths)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{label_family(family)} | {label_mode(mode)}" for family, mode in keys])
    ax.invert_yaxis()
    rate_x_axis(ax, "Share of failure-tagged calls")
    ax.set_title("Failure taxonomy by task family and mode", pad=12)
    bottom_legend(ax, ncol=min(4, len(failures)), anchor_y=-0.14)
    reserve_bottom_margin(fig, bottom=0.22)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_call_efficiency(rows: list[Json], path: Path) -> None:
    """Useful vs wasted call share."""
    keys = ordered_group_keys(rows)
    if not keys:
        no_data_figure(path, "Call efficiency")
        return

    grouped = group(rows, ["family", "model", "mode"])
    useful = [avg(grouped[k], "useful_rate") for k in keys]
    wasted = [1.0 - value for value in useful]
    y = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11.2, horizontal_height(len(keys), per_row=0.52)), constrained_layout=False)
    ax.barh(
        y,
        useful,
        color=SUCCESS_COLOR,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        height=0.62,
        label="State-changing calls",
    )
    ax.barh(
        y,
        wasted,
        left=useful,
        color=REMAINDER_COLOR,
        edgecolor="white",
        linewidth=0.8,
        height=0.62,
        label="Failure / non-state calls",
    )

    for idx, (state_share, waste_share) in enumerate(zip(useful, wasted, strict=True)):
        if state_share >= 0.12:
            ax.text(state_share / 2, idx, percent(state_share), ha="center", va="center", fontsize=9, fontweight="semibold", color="white")
        if waste_share >= 0.14:
            ax.text(state_share + waste_share / 2, idx, percent(waste_share), ha="center", va="center", fontsize=9, color=TEXT_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels([group_label_tick(key) for key in keys])
    ax.invert_yaxis()
    rate_x_axis(ax, "Share of tool calls")
    ax.set_title("Call efficiency", pad=14)
    bottom_legend(ax, ncol=2)
    reserve_bottom_margin(fig, bottom=0.20, left=0.30)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _sorted_param_values(rows: list[Json], key: str) -> list[Any]:
    return sorted({row["params_dict"][key] for row in rows if key in row["params_dict"]})


def _param_heatmap_matrix(
    rows: list[Json],
    *,
    mode: str,
    row_key: str,
    col_key: str,
    value_fn: Callable[[Json], float],
) -> tuple[np.ndarray, list[Any], list[Any]]:
    subset = [row for row in rows if row["mode"] == mode]
    row_vals = _sorted_param_values(subset, row_key)
    col_vals = _sorted_param_values(subset, col_key)
    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    counts = np.zeros_like(matrix)

    row_index = {value: idx for idx, value in enumerate(row_vals)}
    col_index = {value: idx for idx, value in enumerate(col_vals)}
    for row in subset:
        p = row["params_dict"]
        if row_key not in p or col_key not in p:
            continue
        ri = row_index[p[row_key]]
        ci = col_index[p[col_key]]
        value = float(value_fn(row))
        matrix[ri, ci] = (0.0 if np.isnan(matrix[ri, ci]) else matrix[ri, ci]) + value
        counts[ri, ci] += 1

    result = np.full_like(matrix, np.nan)
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
    value_fn: Callable[[Json], float],
    title: str,
    cbar_label: str,
    cmap: str,
    text_fmt: str,
) -> None:
    modes = present_modes(rows)
    if not modes:
        no_data_figure(path, title, "No matching rows available.")
        return

    shapes = [
        _param_heatmap_matrix(rows, mode=mode, row_key=row_key, col_key=col_key, value_fn=value_fn)
        for mode in modes
    ]
    max_rows = max((len(row_vals) for _, row_vals, _ in shapes), default=1)
    max_cols = max((len(col_vals) for _, _, col_vals in shapes), default=1)
    width = max(4.4 * len(modes) + 1.0, 2.0 + max_cols * 1.0 * len(modes))
    height = max(4.5, 2.3 + max_rows * 0.82)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = fig.add_gridspec(1, len(modes) + 1, width_ratios=[*[1.0] * len(modes), 0.055])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(len(modes))]
    cax = fig.add_subplot(grid[0, len(modes)])
    im = None

    for ax, mode, (matrix, row_vals, col_vals) in zip(axes, modes, shapes, strict=True):
        if not row_vals or not col_vals:
            ax.axis("off")
            ax.set_title(label_mode(mode), pad=10)
            continue

        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(col_vals)), labels=[str(v) for v in col_vals])
        ax.set_yticks(range(len(row_vals)), labels=[str(v) for v in row_vals])
        ax.set_xlabel(col_label)
        ax.set_ylabel(row_label if ax is axes[0] else "")
        ax.set_title(label_mode(mode), pad=10)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(col_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_vals), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(1.0)
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val >= 0.58 else TEXT_COLOR
                ax.text(j, i, text_fmt.format(val), ha="center", va="center", color=color, fontsize=10.5, fontweight="semibold")

    if im is None:
        cax.axis("off")
    else:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)
        cbar.outline.set_edgecolor(SPINE_COLOR)
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(title, fontsize=14, fontweight="semibold", color=TEXT_COLOR)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_dag_heatmap(rows: list[Json], path: Path) -> None:
    """Solve rate heatmap over depth x n_core for PTC vs standard."""
    subset = [r for r in rows if r["family"] == "hidden_dag"]
    _render_param_heatmaps(
        subset,
        path,
        row_key="depth",
        col_key="n_core",
        row_label="Depth",
        col_label="Core functions",
        value_fn=lambda row: float(row["solved"]),
        title="Hidden DAG: difficulty vs solve rate",
        cbar_label="Solve rate",
        cmap="YlGnBu",
        text_fmt="{:.0%}",
    )


def plot_repair_chain_heatmap(rows: list[Json], path: Path) -> None:
    """Progress heatmap over n_objects x n_repair_steps."""
    subset = [r for r in rows if r["family"] == "repair_chain"]
    _render_param_heatmaps(
        subset,
        path,
        row_key="n_objects",
        col_key="n_repair_steps",
        row_label="Objects",
        col_label="Repair steps",
        value_fn=lambda row: float(row["progress"]),
        title="Repair chain: task size vs progress",
        cbar_label="Average progress",
        cmap="PuBuGn",
        text_fmt="{:.0%}",
    )


def plot_calls_distribution(rows: list[Json], path: Path) -> None:
    """Horizontal box/strip plot of tool calls by mode and family."""
    keys = ordered_group_keys(rows)
    if not keys:
        no_data_figure(path, "Distribution of tool-call budget usage")
        return

    grouped = group(rows, ["family", "model", "mode"])
    data = [np.array([row["calls"] for row in grouped[key]], dtype=float) for key in keys]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11.2, horizontal_height(len(keys), per_row=0.44)), constrained_layout=False)

    bp = ax.boxplot(
        data,
        vert=False,
        positions=y,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F2933", "linewidth": 1.4},
        whiskerprops={"color": "#7B8A93", "linewidth": 1.0},
        capprops={"color": "#7B8A93", "linewidth": 1.0},
    )
    for patch, key in zip(bp["boxes"], keys, strict=True):
        patch.set_facecolor(MODE_COLORS[key[2]])
        patch.set_alpha(0.30)
        patch.set_edgecolor(MODE_COLORS[key[2]])
        patch.set_linewidth(1.1)

    for idx, values in enumerate(data):
        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            values,
            np.full(len(values), idx) + jitter,
            s=36,
            color=MODE_COLORS[keys[idx][2]],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.88,
            zorder=3,
        )
        ax.annotate(
            f"avg {np.mean(values):.0f}",
            xy=(float(np.max(values)), idx),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=MUTED_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(group_label(key), 34) for key in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Tool calls per episode")
    ax.set_title("Distribution of tool-call budget usage", pad=12)
    soften_axes(ax, axis="x")
    add_mode_legend(fig)
    reserve_bottom_margin(fig, bottom=0.18, left=0.30)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_exhaustion(rows: list[Json], path: Path) -> None:
    keys = ordered_group_keys(rows)
    if not keys:
        no_data_figure(path, "Episodes that exhausted the tool-call budget")
        return

    grouped = group(rows, ["family", "model", "mode"])
    rates = [sum(1 for r in grouped[k] if r["budget_exhausted"]) / len(grouped[k]) for k in keys]
    y = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11.2, horizontal_height(len(keys), per_row=0.42)), constrained_layout=False)
    bars = ax.barh(
        y,
        rates,
        color=[MODE_COLORS[key[2]] for key in keys],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        height=0.62,
    )
    add_bar_end_labels(ax, bars, rates)
    if all(rate == 0 for rate in rates):
        ax.scatter(
            np.full(len(keys), 0.002),
            y,
            s=38,
            color=[MODE_COLORS[key[2]] for key in keys],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            clip_on=False,
        )
        ax.text(
            0.015,
            -0.7,
            "No episodes exhausted the budget",
            ha="left",
            va="center",
            fontsize=10,
            color=MUTED_COLOR,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(group_label(key), 34) for key in keys])
    ax.invert_yaxis()
    rate_x_axis(ax, "Budget exhaustion rate")
    ax.set_title("Episodes that exhausted the tool-call budget", pad=12)
    add_mode_legend(fig)
    reserve_bottom_margin(fig, bottom=0.18, left=0.30)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def top_failure_codes(rows: list[Json], n: int = 6) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["failure_counter"])
    return [code for code, _ in counts.most_common(n)]


if __name__ == "__main__":
    raise SystemExit(main())
