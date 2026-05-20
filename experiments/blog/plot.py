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
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import DEFAULT_CONFIG, figures_dir_for_raw, raw_dir_for_config  # noqa: E402

DEFAULT_RAW_DIR = raw_dir_for_config(DEFAULT_CONFIG)

Json = dict[str, Any]

MODE_COLORS = {"ptc": "#2A9D8F", "standard": "#E76F51"}
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
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, help="defaults to <input-parent>/figures when input ends in /raw")
    args = parser.parse_args(argv)

    output_dir = args.output or figures_dir_for_raw(args.input)
    rows = read_rows(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
        }
    )

    plot_headline_comparison(rows, output_dir / "01_headline_comparison.png")
    plot_progress_gap(rows, output_dir / "02_progress_vs_success.png")
    plot_failure_taxonomy(rows, output_dir / "03_failure_taxonomy.png")
    plot_call_efficiency(rows, output_dir / "04_call_efficiency.png")
    plot_hidden_dag_heatmap(rows, output_dir / "05_hidden_dag_heatmap.png")
    plot_repair_chain_heatmap(rows, output_dir / "06_repair_chain_heatmap.png")
    plot_calls_distribution(rows, output_dir / "07_calls_distribution.png")
    plot_budget_exhaustion(rows, output_dir / "08_budget_exhaustion.png")

    print(f"output_dir={output_dir}")
    return 0


def read_rows(path: Path) -> list[Json]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows: list[Json] = []
    for file in files:
        with file.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(normalize_row(json.loads(line)))
    return rows


def normalize_row(row: Json) -> Json:
    """Support both legacy and current raw schemas."""
    out = dict(row)
    out["solved"] = bool(row.get("solved", row.get("success", False)))
    out["calls"] = float(row.get("calls", row.get("num_calls", 0.0)))
    out["progress"] = float(row.get("progress", row.get("progress_score", 0.0)))
    out["params_dict"] = json.loads(row["params"])
    useful = float(row.get("useful_calls", row.get("state_changes", 0.0)))
    wasted = float(row.get("wasted_calls", row.get("failure_calls", 0.0)))
    total = useful + wasted
    out["useful_rate"] = useful / total if total else 0.0
    out["budget_exhausted"] = row.get("calls", row.get("num_calls", 0.0)) >= row["budget"]
    return out


def group(rows: list[Json], keys: list[str]) -> dict[tuple[Any, ...], list[Json]]:
    grouped: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(sorted(grouped.items()))


def avg(rows: list[Json], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def plot_headline_comparison(rows: list[Json], path: Path) -> None:
    """Solve rate and avg progress by mode, family, and model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    metrics = [("solved", "Solve rate"), ("progress", "Avg progress")]

    for ax, (key, ylabel) in zip(axes, metrics, strict=True):
        grouped = group(rows, ["family", "model", "mode"])
        families = ["hidden_dag", "repair_chain"]
        models = sorted({row["model"] for row in rows})
        x = np.arange(len(families))
        width = 0.18
        offsets = np.linspace(-(len(models) * 1.5 - 0.5) * width / 2, (len(models) * 1.5 - 0.5) * width / 2, len(models) * 2)

        for idx, model in enumerate(models):
            for mode_idx, mode in enumerate(["ptc", "standard"]):
                values = []
                for family in families:
                    subset = grouped.get((family, model, mode), [])
                    if key == "solved":
                        values.append(avg(subset, "solved") if subset else 0.0)
                    else:
                        values.append(avg(subset, "progress") if subset else 0.0)
                label = f"{MODEL_LABELS.get(model, model)} ({mode})"
                ax.bar(
                    x + offsets[idx * 2 + mode_idx],
                    values,
                    width,
                    label=label,
                    color=MODE_COLORS[mode],
                    alpha=0.85 if mode == "ptc" else 0.55,
                    edgecolor="white",
                    linewidth=0.6,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in families])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.axhline(1.0, color="#ccc", linewidth=0.8, linestyle="--")

    axes[0].legend(loc="upper left", ncol=2, frameon=True)
    fig.suptitle("Standard vs PTC: solve rate and progress (pre-fix diagnostic run)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_progress_gap(rows: list[Json], path: Path) -> None:
    """Highlight repair_chain episodes that finish the state machine but not exact solve."""
    repair = [r for r in rows if r["family"] == "repair_chain"]
    grouped = group(repair, ["mode", "model"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    solve_vals = []
    progress_vals = []
    colors_solve = []
    colors_progress = []

    for (mode, model), subset in grouped.items():
        labels.append(f"{MODEL_LABELS.get(model, model)}\n({mode})")
        solve_vals.append(avg(subset, "solved"))
        progress_vals.append(avg(subset, "progress"))
        colors_solve.append(MODE_COLORS[mode])
        colors_progress.append(MODE_COLORS[mode])

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, progress_vals, width, label="Avg progress", color="#264653", alpha=0.9)
    ax.bar(x + width / 2, solve_vals, width, label="Solve rate", color="#E9C46A", edgecolor="#333", linewidth=0.5)

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
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.set_title("Repair chain: high progress, zero exact solve (PTC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_failure_taxonomy(rows: list[Json], path: Path) -> None:
    """Stacked horizontal bars of failure shares by mode and family."""
    grouped = group(rows, ["mode", "family"])
    keys = list(grouped.keys())
    top_failures = top_failure_codes(rows, n=8)

    fig, ax = plt.subplots(figsize=(10, 5))
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
            color=cmap(idx),
            edgecolor="white",
            linewidth=0.4,
        )
        left += np.array(widths)

    ylabels = []
    for mode, family in keys:
        ylabels.append(f"{FAMILY_LABELS[family]} · {mode}")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Share of failure-tagged calls")
    ax.set_title("Failure taxonomy by task family and mode")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_call_efficiency(rows: list[Json], path: Path) -> None:
    """Useful vs wasted call share."""
    grouped = group(rows, ["family", "model", "mode"])
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    labels = [f"{FAMILY_LABELS[k[0]]}\n{MODEL_LABELS.get(k[1], k[1])} ({k[2]})" for k in keys]

    useful = [avg(grouped[k], "useful_rate") for k in keys]
    wasted = [1.0 - u for u in useful]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(keys))
    ax.bar(x, useful, label="Useful calls", color="#2A9D8F")
    ax.bar(x, wasted, bottom=useful, label="Wasted / failure calls", color="#F4A261")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of tool calls")
    ax.set_title("Call efficiency: useful vs wasted")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_dag_heatmap(rows: list[Json], path: Path) -> None:
    """Solve rate heatmap over depth × n_core for PTC vs standard."""
    subset = [r for r in rows if r["family"] == "hidden_dag"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mode in zip(axes, ["ptc", "standard"], strict=True):
        matrix = np.full((2, 2), np.nan)
        counts = np.zeros((2, 2))
        depth_vals = [2, 5]
        core_vals = [5, 10]

        for row in subset:
            if row["mode"] != mode:
                continue
            p = row["params_dict"]
            di = depth_vals.index(p["depth"])
            ci = core_vals.index(p["n_core"])
            matrix[di, ci] = (0.0 if np.isnan(matrix[di, ci]) else matrix[di, ci]) + float(row["solved"])
            counts[di, ci] += 1

        result = np.zeros_like(matrix)
        with np.errstate(invalid="ignore"):
            np.divide(matrix, counts, out=result, where=counts > 0)
        matrix = result

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], labels=["n_core=5", "n_core=10"])
        ax.set_yticks([0, 1], labels=["depth=2", "depth=5"])
        ax.set_title(f"{mode.upper()} solve rate")
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center", color="black", fontsize=11)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Solve rate")
    fig.suptitle("Hidden DAG: difficulty vs solve rate (averaged over seeds & distractors)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_repair_chain_heatmap(rows: list[Json], path: Path) -> None:
    """Progress heatmap over n_objects × n_repair_steps."""
    subset = [r for r in rows if r["family"] == "repair_chain"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mode in zip(axes, ["ptc", "standard"], strict=True):
        matrix = np.full((2, 2), np.nan)
        counts = np.zeros((2, 2))
        obj_vals = [1, 3]
        step_vals = [2, 3]

        for row in subset:
            if row["mode"] != mode:
                continue
            p = row["params_dict"]
            oi = obj_vals.index(p["n_objects"])
            si = step_vals.index(p["n_repair_steps"])
            matrix[oi, si] = (0.0 if np.isnan(matrix[oi, si]) else matrix[oi, si]) + row["progress"]
            counts[oi, si] += 1

        result = np.zeros_like(matrix)
        with np.errstate(invalid="ignore"):
            np.divide(matrix, counts, out=result, where=counts > 0)
        matrix = result

        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], labels=["2 steps", "3 steps"])
        ax.set_yticks([0, 1], labels=["1 object", "3 objects"])
        ax.set_title(f"{mode.upper()} avg progress")
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.5 else "black")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Avg progress")
    fig.suptitle("Repair chain: task size vs progress (averaged over seeds & distractors)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_calls_distribution(rows: list[Json], path: Path) -> None:
    """Box plot of tool calls by mode and family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, family in zip(axes, ["hidden_dag", "repair_chain"], strict=True):
        data = []
        positions = []
        tick_labels = []
        pos = 0
        for model in sorted({r["model"] for r in rows}):
            for mode in ["ptc", "standard"]:
                subset = [r["calls"] for r in rows if r["family"] == family and r["model"] == model and r["mode"] == mode]
                if subset:
                    data.append(subset)
                    positions.append(pos)
                    tick_labels.append(f"{MODEL_LABELS.get(model, model)}\n{mode}")
                    pos += 1

        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, label in zip(bp["boxes"], tick_labels, strict=True):
            mode = "ptc" if "ptc" in label else "standard"
            patch.set_facecolor(MODE_COLORS[mode])
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_title(FAMILY_LABELS[family])
        ax.set_ylabel("Tool calls per episode")

    fig.suptitle("Distribution of tool-call budget usage", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_exhaustion(rows: list[Json], path: Path) -> None:
    grouped = group(rows, ["family", "model", "mode"])
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    rates = [sum(1 for r in grouped[k] if r["budget_exhausted"]) / len(grouped[k]) for k in keys]
    labels = [f"{FAMILY_LABELS[k[0]]}\n{MODEL_LABELS.get(k[1], k[1])}\n({k[2]})" for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [MODE_COLORS[k[2]] for k in keys]
    ax.bar(range(len(keys)), rates, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Budget exhaustion rate")
    ax.set_title("Episodes that exhausted the tool-call budget")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def top_failure_codes(rows: list[Json], n: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["failure_counter"])
    return [code for code, _ in counts.most_common(n)]


if __name__ == "__main__":
    raise SystemExit(main())
