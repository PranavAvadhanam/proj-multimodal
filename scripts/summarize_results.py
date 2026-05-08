"""Summarize the most recent metrics from each pipeline into a comparison table.

Usage:
    python scripts/summarize_results.py              # terminal table only
    python scripts/summarize_results.py --png        # also save outputs/results_summary.png
    python scripts/summarize_results.py --png -o my_results.png
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings

TASK_CODES = ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM"]
TASK_FULL_NAMES = {
    "AIE": "Audio Info\nExtraction",
    "ACC": "Audio\nCounting",
    "AEL": "Audio Event\nLocalization",
    "AVCM": "AV Character\nMatching",
    "AVOM": "AV Object\nMatching",
    "AVTM": "AV Text\nMatching",
}


def _task_abbrev_lines_for_key() -> str:
    """One source of truth with ``TASK_CODES`` / ``TASK_FULL_NAMES`` (footer key in PNG)."""
    lines: list[str] = []
    for tc in TASK_CODES:
        lines.append(f"{tc} = {TASK_FULL_NAMES[tc].replace(chr(10), ' ')}")
    return "\n".join(lines)


PIPELINE_CONFIGS = [
    ("vanilla", "Vanilla"),
    ("vanilla_cot", "Vanilla + Reason"),
    ("idea2", "ModalitySeparation"),
    ("idea2_cot", "ModalitySeparation + Reason"),
    ("idea2_mis", "ModalitySeparation + MIS"),
    ("idea2_mis_cot", "ModalitySeparation + MIS + Reason"),
]

# Output folder names still use the historical ``idea2`` prefix in ``outputs/``.
MODSEP_OUTPUT_SUBDIRS = frozenset({"idea2", "idea2_cot", "idea2_mis", "idea2_mis_cot"})


@lru_cache(maxsize=1)
def _default_max_output_tokens_describe() -> int:
    """Per-modality describe cap; same as ``Settings.max_output_tokens_describe`` (see ``src.config``)."""
    return get_settings().max_output_tokens_describe


@lru_cache(maxsize=1)
def _settings_gemini_model() -> str:
    return get_settings().gemini_model


def _describe_max_tokens_cell(subdir_name: str, m: dict) -> str:
    """Cell text for per-modality describe max-output cap.

    Uses ``max_output_tokens_describe`` from metrics when present; otherwise the
    current ``GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE`` resolution via ``get_settings()``, as in config.
    """
    if subdir_name not in MODSEP_OUTPUT_SUBDIRS:
        return "—"
    raw = m.get("max_output_tokens_describe")
    if raw is not None:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = _default_max_output_tokens_describe()
    else:
        n = _default_max_output_tokens_describe()
    return str(n)


def _find_latest_metrics(subdir: Path) -> dict | None:
    """Find the most recently modified *_metrics_av_human*.json in a directory."""
    if not subdir.is_dir():
        return None
    candidates = sorted(subdir.glob("*_metrics_av_human*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_results(output_root: Path) -> list[tuple[str, dict, str]]:
    """Return (display_name, metrics_dict, output_subdir_name) for each pipeline with results."""
    results = []
    for subdir_name, display_name in PIPELINE_CONFIGS:
        metrics = _find_latest_metrics(output_root / subdir_name)
        if metrics is not None:
            results.append((display_name, metrics, subdir_name))
    return results


def _subtitle_describe_caps(
    results: list[tuple[str, dict, str]],
) -> tuple[str, list[int]]:
    """Return subtitle line listing describe caps observed for ModalitySeparation rows."""

    vals: list[int] = []
    for _, m, subdir in results:
        if subdir not in MODSEP_OUTPUT_SUBDIRS:
            continue
        raw = m.get("max_output_tokens_describe")
        if raw is not None:
            try:
                vals.append(int(raw))
            except (TypeError, ValueError):
                vals.append(_default_max_output_tokens_describe())
        else:
            vals.append(_default_max_output_tokens_describe())

    fallback = _default_max_output_tokens_describe()
    if not vals:
        return (
            f"No ModalitySeparation rows in this summary — vanilla-only tables. Default describe cap "
            f"when you run MS pipelines: {fallback} tok (GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE).",
            [fallback],
        )

    uniq = sorted(set(vals))
    if len(uniq) == 1:
        return (
            f"ModalitySeparation describe cap: {uniq[0]} max output tokens per modality describe call "
            f"(GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE unless overridden).",
            uniq,
        )
    pretty = ", ".join(str(v) for v in uniq)
    return (
        f"Mixed ModalitySeparation describe caps in this table — max output tokens per modality describe: "
        f"{pretty}.",
        uniq,
    )


def print_terminal_table(results: list[tuple[str, dict, str]]) -> None:
    """Print a clean terminal table of results."""
    if not results:
        print("No results found in outputs/. Run some evaluations first.")
        return

    col_w = 12
    name_w = max(len(name) for name, _, _ in results) + 2
    header_cols = ["Overall"] + TASK_CODES + ["N", "Time (s)", "Desc max"]
    header = f"{'Pipeline':<{name_w}}" + " " + " ".join(f"{c:>{col_w}}" for c in header_cols)
    sep = "─" * len(header)

    print(f"\n{sep}")
    print("  AVUT Multimodal QA — Results Summary")
    print(sep)
    print(header)
    print(sep)

    for name, m, subdir in results:
        acc = m.get("accuracy", 0)
        task_acc = m.get("task_accuracy", {})
        n = m.get("n_samples_scored", 0)
        latency = m.get("pipeline_latency_s", 0)
        desc_tok = _describe_max_tokens_cell(subdir, m)
        row = f"{name:<{name_w}}"
        cells = [f"{acc * 100:>{col_w}.1f}%"]
        for tc in TASK_CODES:
            val = task_acc.get(tc)
            if val is not None:
                cells.append(f"{val * 100:>{col_w}.1f}%")
            else:
                cells.append(f"{'—':>{col_w}}")
        cells.append(f"{n:>{col_w}}")
        cells.append(f"{latency:>{col_w}.0f}")
        cells.append(f"{desc_tok:>{col_w}}")
        row += " " + " ".join(cells)
        print(row)

    print(f"{sep}\n")


def render_png(results: list[tuple[str, dict, str]], output_path: Path) -> None:
    """Render a polished matplotlib table and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results to render.", file=sys.stderr)
        return

    col_labels = (
        ["Overall"]
        + [TASK_FULL_NAMES[tc] for tc in TASK_CODES]
        + ["N", "Time (s)", "Describe\nmax (tok)"]
    )
    row_labels = [name for name, _, _ in results]

    cell_data: list[list[str]] = []
    cell_values: list[list[float | None]] = []
    for _, m, subdir in results:
        acc = m.get("accuracy", 0)
        task_acc = m.get("task_accuracy", {})
        n = m.get("n_samples_scored", 0)
        latency = m.get("pipeline_latency_s", 0)
        desc_tok = _describe_max_tokens_cell(subdir, m)
        row_text = [f"{acc * 100:.1f}%"]
        row_vals: list[float | None] = [acc]
        for tc in TASK_CODES:
            val = task_acc.get(tc)
            if val is not None:
                row_text.append(f"{val * 100:.1f}%")
                row_vals.append(val)
            else:
                row_text.append("—")
                row_vals.append(None)
        row_text.append(str(n))
        row_vals.append(None)
        row_text.append(f"{latency:.0f}")
        row_vals.append(None)
        row_text.append(desc_tok)
        row_vals.append(None)
        cell_data.append(row_text)
        cell_values.append(row_vals)

    n_rows = len(results)
    n_cols = len(col_labels)
    fig_w = max(14, n_cols * 1.55)
    # Footer + table + titles — keep figure height modest; tuck table close to key.
    fig_h = max(10.0, 4.2 + n_rows * 0.48)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    header_bottom_fig = 0.88
    key_region_top_fig = 0.30
    gap_table_key = 0.012
    ax_bottom_fig = key_region_top_fig + gap_table_key
    ax_height_fig = max(0.28, header_bottom_fig - ax_bottom_fig)
    ax.set_position([0.05, ax_bottom_fig, 0.9, ax_height_fig])

    sub_desc, describe_vals = _subtitle_describe_caps(results)
    uniq_desc = sorted(set(describe_vals))
    if len(uniq_desc) == 1:
        title_caps = f"{uniq_desc[0]} tokens / modality describe"
    else:
        title_caps = "mixed caps — see Describe column"

    fig.text(
        0.5,
        0.98,
        f"AVUT Multimodal QA — Results Summary  |  {title_caps}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.933,
        sub_desc + f"   |   Model: {_settings_gemini_model()}",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
        fontfamily="sans-serif",
    )
    fig.text(
        0.5,
        0.896,
        '`"+ Reason"` variants: Gemini reasoning / thinking budget on for final MCQ (768).',
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        fontfamily="sans-serif",
        style="italic",
    )

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    cmap = LinearSegmentedColormap.from_list("acc", ["#fee2e2", "#fef9c3", "#dcfce7"], N=256)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.5)

        if row_idx == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
            cell.set_height(0.12)
        elif col_idx == -1:
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(fontweight="bold", fontsize=9.5)
        else:
            data_row = row_idx - 1
            data_col = col_idx
            if 0 <= data_row < len(cell_values) and 0 <= data_col < len(cell_values[data_row]):
                val = cell_values[data_row][data_col]
                if val is not None:
                    cell.set_facecolor(cmap(val))
                    if data_col == 0:
                        cell.set_text_props(fontweight="bold")
                else:
                    cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#ffffff")

            if row_idx % 2 == 0 and cell.get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                cell.set_facecolor("#f9fafb")

    fig.text(
        0.03,
        0.015,
        "Key:\n"
        "• Heatmap colouring applies only under task columns and Overall — red ≈ lower accuracy; green ≈ higher.\n"
        "• Accuracy is fraction correct on scored rows; blanks under a task mean that task had zero scored items in this run.\n"
        "• N = prompts scored OK; Time (s) ≈ aggregate wall time recorded in that metrics file.\n"
        "• Describe max (tok) = Gemini max_output_tokens limit for ModalitySeparation describe calls (each of text/audio/visual); Vanilla shows “—.”\n"
        "• MIS rows may show different numbers per modality if recorded in metrics; otherwise GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE fallback.\n\n"
        "Task abbreviations:\n"
        f"{_task_abbrev_lines_for_key()}\n\n"
        "Environment fallback — GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE: "
        f"{_default_max_output_tokens_describe()}",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
        fontfamily="sans-serif",
        linespacing=1.35,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AVUT pipeline results.")
    parser.add_argument(
        "--output-root", default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Save a PNG comparison table",
    )
    parser.add_argument(
        "-o", "--output-file", default=None,
        help="PNG output path (default: outputs/results_summary.png)",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    results = collect_results(root)

    print_terminal_table(results)

    if args.png:
        out_path = Path(args.output_file) if args.output_file else root / "results_summary.png"
        render_png(results, out_path)


if __name__ == "__main__":
    main()
