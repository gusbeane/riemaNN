"""Scan all outputs/*/metrics.json and write a summary table to outputs/summary.txt.

Can be run standalone:
    venv/bin/python -m scripts.summarize

Also called automatically at the end of every Experiment.run() / run_custom().
"""

from __future__ import annotations

import json
from pathlib import Path

from riemann_pinn import paths


# Metric columns to show, in display order.
_METRIC_COLS = [
    ("med |f|",        "median_abs_fstar"),
    ("p95 |f|",        "p95_abs_fstar"),
    ("med |Δlgp|",     "median_abs_delta_log10_p"),
    ("p95 |Δlgp|",     "p95_abs_delta_log10_p"),
    ("med rel",         "median_rel_abs_p_err"),
    ("p95 rel",         "p95_rel_abs_p_err"),
    ("frac p>0",        "frac_both_p_positive"),
]

# Config fields used for the sort key (group similar experiments together).
_SORT_FIELDS = [
    ("loss_impl",),
    ("model", "depth"),
    ("model", "width"),
    ("optimizer", "learning_rate"),
    ("n_epochs",),
]


def _deep_get(d: dict, keys: tuple) -> str:
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, "")
        else:
            return ""
    return str(cur)


def _sort_key(entry: dict) -> tuple:
    cfg = entry.get("config", {})
    parts: list = []
    for field_path in _SORT_FIELDS:
        val = _deep_get(cfg, field_path)
        # Try numeric sort first, fall back to string
        try:
            parts.append((0, float(val)))
        except (ValueError, TypeError):
            parts.append((1, val))
    return tuple(parts)


def _fmt(val: float | None, width: int) -> str:
    if val is None:
        return "-".center(width)
    if abs(val) >= 1.0 or val == 0.0:
        return f"{val:.3f}".rjust(width)
    return f"{val:.2e}".rjust(width)


def _config_summary(cfg: dict) -> str:
    """One-line summary of the key config axes."""
    parts = []
    m = cfg.get("model", {})
    parts.append(f"w{m.get('width', '?')}")
    parts.append(f"d{m.get('depth', '?')}")
    loss = cfg.get("loss_impl", "?")
    if loss != "residual_loss":
        parts.append(loss.replace("_loss", ""))
    opt = cfg.get("optimizer", {})
    lr = opt.get("learning_rate", "?")
    if lr != 0.001:
        parts.append(f"lr={lr}")
    ep = cfg.get("n_epochs", "?")
    if ep != 5000:
        parts.append(f"{ep}ep")
    return " ".join(parts)


def collect_entries(root: Path = paths.OUTPUTS_ROOT) -> list[dict]:
    """Gather metrics + config from every experiment that has metrics.json.

    Searches both `outputs/*/metrics.json` (active) and
    `outputs/archive/*/metrics.json` (archived).
    """
    entries = []
    # Search both top-level and archive/ subdirectory
    globs = [root.glob("*/metrics.json"), root.glob("archive/*/metrics.json")]
    for pattern in globs:
        for metrics_file in sorted(pattern):
            exp_dir = metrics_file.parent
            config_file = exp_dir / "config.json"
            # Build the name relative to outputs root
            rel = exp_dir.relative_to(root)
            name = str(rel)
            if name.startswith("_"):
                continue
            try:
                metrics = json.loads(metrics_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            config = {}
            if config_file.is_file():
                try:
                    config = json.loads(config_file.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            entries.append({"name": name, "metrics": metrics, "config": config})
    return entries


def _build_section(entries: list[dict], title: str) -> list[str]:
    """Format a group of entries into header + rows."""
    if not entries:
        return []

    entries.sort(key=_sort_key)

    col_w = 10
    name_w = max(len(e["name"]) for e in entries)
    cfg_w = max(len(_config_summary(e["config"])) for e in entries)

    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
        lines.append("")

    hdr = "name".ljust(name_w) + "  " + "config".ljust(cfg_w)
    for label, _ in _METRIC_COLS:
        hdr += "  " + label.rjust(col_w)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for entry in entries:
        # Strip archive/ prefix from display name for cleaner output
        display_name = entry["name"]
        if display_name.startswith("archive/"):
            display_name = display_name[len("archive/"):]
        row = display_name.ljust(name_w) + "  " + _config_summary(entry["config"]).ljust(cfg_w)
        for _, key in _METRIC_COLS:
            val = entry["metrics"].get(key)
            row += "  " + _fmt(val, col_w)
        lines.append(row)

    return lines


_LEGEND = """\
Column definitions:
  med |f|    — median absolute fstar residual (should be 0 at true p*)
  p95 |f|    — 95th percentile absolute fstar residual
  med |Δlgp| — median |log10(p_NN) − log10(p_true)| (log-scale pressure error)
  p95 |Δlgp| — 95th percentile log-scale pressure error
  med rel    — median |p_NN − p_true| / |p_true| (relative pressure error)
  p95 rel    — 95th percentile relative pressure error
  frac p>0   — fraction of holdout samples where both p_NN and p_true are positive
"""


def build_table(entries: list[dict]) -> str:
    """Format entries into a fixed-width text table with active/archived sections."""
    if not entries:
        return "(no experiments found)\n"

    active = [e for e in entries if not e["name"].startswith("archive/")]
    archived = [e for e in entries if e["name"].startswith("archive/")]

    lines: list[str] = [_LEGEND]
    lines.extend(_build_section(active, "Active experiments"))

    if archived:
        lines.append("")
        lines.append("")
        lines.extend(_build_section(archived, f"Archived experiments ({len(archived)})"))

    total = len(active) + len(archived)
    lines.append("")
    lines.append(f"{total} experiments ({len(active)} active, {len(archived)} archived)")
    return "\n".join(lines) + "\n"


def write_summary() -> Path:
    entries = collect_entries()
    table = build_table(entries)
    out = paths.OUTPUTS_ROOT / "summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    return out


if __name__ == "__main__":
    out = write_summary()
    print(out.read_text())
