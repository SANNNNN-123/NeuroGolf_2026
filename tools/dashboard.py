"""Overview dashboard page for the NeuroGolf Streamlit app.

Aggregates the per-task scoreboard (`reports/manifest.json`) across all 400 tasks:
score-band and operator-mechanism grouping, a sortable/filterable task table, and
drill-down into the per-task Task Viewer.

The pure aggregation helpers (`band_of`, `build_rows`, `score_bands`, `kpis`,
`op_breakdown`) take plain data and return plain data, so they are unit-tested in
`tests/test_dashboard.py` without a running Streamlit server.
"""

from __future__ import annotations

import html
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

MANIFEST_PATH = REPORTS / "manifest.json"
INVENTORY_PATH = REPORTS / "global_layer_inventory.json"
ARC_MAPPING_PATH = REPORTS / "arc_mapping.json"

MAX_POINTS = 25.0
# Ordered worst-last so the summary reads high -> low.
BANDS = ["25 (max)", "22-25", "20-22", "18-20", "<18"]


# ---------------------------------------------------------------------------
# Pure aggregation helpers (unit-tested)
# ---------------------------------------------------------------------------
def band_of(points: float) -> str:
    """Bucket a task's points into one of BANDS."""
    p = float(points)
    if p >= 24.999:
        return "25 (max)"
    if p >= 22.0:
        return "22-25"
    if p >= 20.0:
        return "20-22"
    if p >= 18.0:
        return "18-20"
    return "<18"


def _task_op_map(by_op: dict[str, list[int]]) -> dict[int, set[str]]:
    """Invert {op: [task numbers]} into {task number: {ops}}."""
    task_ops: dict[int, set[str]] = {}
    for op, tasks in by_op.items():
        for t in tasks:
            task_ops.setdefault(int(t), set()).add(op)
    return task_ops


def build_rows(
    manifest_tasks: dict[str, Any],
    by_op: dict[str, list[int]],
    arc_mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge manifest + operator inventory + arc mapping into one row per task."""
    task_ops = _task_op_map(by_op)
    rows: list[dict[str, Any]] = []
    for key, value in manifest_tasks.items():
        n = int(key)
        ops = sorted(task_ops.get(n, set()))
        points = float(value.get("points", 0.0))
        memory = int(value.get("memory", 0))
        params = int(value.get("params", 0))
        method = str(value.get("method", ""))
        arc_row = arc_mapping.get(str(n)) or arc_mapping.get(key) or {}
        rows.append(
            {
                "task": n,
                "arc_id": str(arc_row.get("arc_id", "")),
                "points": points,
                "memory": memory,
                "params": params,
                "mem_params": memory + params,
                "method": method,
                "method_prefix": method.split(":")[0] if method else "",
                "headroom": round(MAX_POINTS - points, 6),
                "n_ops": len(ops),
                "ops": ", ".join(ops),
                "band": band_of(points),
            }
        )
    rows.sort(key=lambda r: r["task"])
    return rows


def score_bands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-band count, share, and summed points, in BANDS order."""
    total = max(1, len(rows))
    out = []
    for band in BANDS:
        members = [r for r in rows if r["band"] == band]
        out.append(
            {
                "band": band,
                "count": len(members),
                "pct": round(100.0 * len(members) / total, 1),
                "sum_points": round(sum(r["points"] for r in members), 2),
            }
        )
    return out


def kpis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline totals: standing, gap to a perfect board, group counts."""
    count = len(rows)
    total = sum(r["points"] for r in rows)
    theoretical = count * MAX_POINTS
    return {
        "task_count": count,
        "total_points": round(total, 2),
        "theoretical_max": round(theoretical, 2),
        "gap_to_max": round(theoretical - total, 2),
        "max_count": sum(1 for r in rows if r["points"] >= 24.999),
        "low_count": sum(1 for r in rows if r["points"] < 18.0),
    }


def points_to_color(points: float, lo: float = 14.0, hi: float = MAX_POINTS) -> str:
    """Map a score to a red -> yellow -> green HSL colour (higher = greener)."""
    span = max(1e-9, hi - lo)
    t = min(1.0, max(0.0, (float(points) - lo) / span))
    hue = t * 130.0  # 0 = red (low), 130 = green (high)
    return f"hsl({hue:.0f}, 68%, 42%)"


def order_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Order tiles by task number or by score."""
    if mode == "score (low first)":
        return sorted(rows, key=lambda r: r["points"])
    if mode == "score (high first)":
        return sorted(rows, key=lambda r: r["points"], reverse=True)
    return sorted(rows, key=lambda r: r["task"])


def op_breakdown(by_op: dict[str, list[int]]) -> list[dict[str, Any]]:
    """Number of tasks using each operator, most-used first."""
    out = [{"op": op, "count": len(tasks)} for op, tasks in by_op.items()]
    out.sort(key=lambda r: r["count"], reverse=True)
    return out


def method_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Task count and average points per method prefix."""
    prefixes: dict[str, list[float]] = {}
    for r in rows:
        prefixes.setdefault(r["method_prefix"] or "(none)", []).append(r["points"])
    out = [
        {
            "method": prefix,
            "count": len(pts),
            "avg_points": round(sum(pts) / len(pts), 2),
        }
        for prefix, pts in prefixes.items()
    ]
    out.sort(key=lambda r: r["count"], reverse=True)
    return out


def points_histogram(rows: list[dict[str, Any]], step: float = 0.5) -> pd.DataFrame:
    """Binned points distribution as a DataFrame indexed by bin label."""
    points = np.array([r["points"] for r in rows], dtype=float)
    if points.size == 0:
        return pd.DataFrame({"tasks": []})
    lo = float(np.floor(points.min()))
    hi = MAX_POINTS
    edges = np.arange(lo, hi + step, step)
    counts, _ = np.histogram(points, bins=edges)
    labels = [f"{edges[i]:.1f}" for i in range(len(counts))]
    return pd.DataFrame({"tasks": counts}, index=labels)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def _load_json(path_str: str) -> Any:
    path = pathlib.Path(path_str)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_all() -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    manifest = _load_json(str(MANIFEST_PATH)) or {}
    tasks = manifest.get("tasks", {}) if isinstance(manifest, dict) else {}
    inventory = _load_json(str(INVENTORY_PATH)) or {}
    by_op = inventory.get("by_op", {}) if isinstance(inventory, dict) else {}
    arc_mapping = _load_json(str(ARC_MAPPING_PATH)) or {}
    rows = build_rows(tasks, by_op, arc_mapping)
    return rows, by_op


# ---------------------------------------------------------------------------
# Render layer
# ---------------------------------------------------------------------------
def _legend_html(lo: float = 14.0, hi: float = MAX_POINTS) -> str:
    stops = ", ".join(points_to_color(lo + t * (hi - lo)) for t in (0, 0.25, 0.5, 0.75, 1.0))
    return (
        "<div style='display:flex;align-items:center;gap:8px;font-size:12px;color:#6b7280'>"
        f"<span>low ({lo:.0f})</span>"
        f"<div style='height:12px;width:200px;border-radius:6px;"
        f"background:linear-gradient(to right,{stops})'></div>"
        f"<span>high ({hi:.0f})</span></div>"
    )


TILE_HEIGHT = 46
TILE_GAP = 3


def _tiles_html(rows: list[dict[str, Any]], columns: int = 20) -> str:
    """Self-contained grid of clickable task tiles for a components.html iframe.

    Each tile navigates the *top* window (same tab) to /viewer?task=N via JS, so
    it works even though the iframe document's own base URL is about:srcdoc.
    """
    cells = []
    for r in rows:
        n = r["task"]
        p = r["points"]
        color = points_to_color(p)
        title = html.escape(
            f"task {n} · {p:.3f} pts · mem {r['memory']:,} · "
            f"params {r['params']:,} · {r['method']}"
        )
        cells.append(
            f'<a href="/viewer?task={n}" '
            "onclick=\"window.open(top.location.origin+'/viewer?task="
            f"{n}','neurogolf_viewer');return false;\" "
            f'title="{title}" '
            "style='display:flex;flex-direction:column;align-items:center;justify-content:center;"
            f"height:{TILE_HEIGHT}px;background:{color};border-radius:4px;cursor:pointer;"
            "text-decoration:none;color:#fff;font-family:Menlo,monospace;line-height:1.05;"
            "box-shadow:inset 0 0 0 1px rgba(255,255,255,.10)'>"
            f"<span style='font-size:12px;font-weight:700'>{n}</span>"
            f"<span style='font-size:9px;opacity:.9'>{p:.1f}</span></a>"
        )
    return (
        "<div style='font-family:Menlo,monospace;'>"
        f"<div style='display:grid;grid-template-columns:repeat({columns},1fr);gap:{TILE_GAP}px'>"
        + "".join(cells)
        + "</div></div>"
    )


def _tiles_height(count: int, columns: int = 20) -> int:
    rows = (count + columns - 1) // columns
    return rows * (TILE_HEIGHT + TILE_GAP) + TILE_GAP + 6


def render_task_tiles(rows: list[dict[str, Any]], columns: int = 20) -> None:
    st.subheader("Task tiles")
    c1, c2 = st.columns([1, 2])
    order = c1.radio(
        "Order",
        ["task number", "score (low first)", "score (high first)"],
        horizontal=False,
    )
    with c2:
        st.caption("Each tile is one task, coloured by score. Click a tile to open it in the Task Viewer.")
        st.markdown(_legend_html(), unsafe_allow_html=True)
    ordered = order_rows(rows, order)
    components.html(
        _tiles_html(ordered, columns),
        height=_tiles_height(len(ordered), columns),
        scrolling=True,
    )


def render_dashboard() -> None:
    st.title("📊 NeuroGolf Dashboard")

    rows, by_op = _load_all()
    if not rows:
        st.warning(f"No scoreboard found. Expected {MANIFEST_PATH}.")
        return

    k = kpis(rows)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("total points", f"{k['total_points']:,.2f}")
    c2.metric("tasks", f"{k['task_count']}")
    c3.metric("theoretical max", f"{k['theoretical_max']:,.0f}")
    c4.metric("gap to max", f"{k['gap_to_max']:,.2f}")
    c5.metric("at 25 (max)", f"{k['max_count']}")
    c6.metric("below 18 (low)", f"{k['low_count']}")

    st.divider()
    render_task_tiles(rows)

    st.divider()
    st.subheader("Score bands")
    bands = score_bands(rows)
    b1, b2 = st.columns([1, 1])
    with b1:
        st.dataframe(bands, use_container_width=True, hide_index=True)
    with b2:
        band_df = pd.DataFrame(
            {"tasks": [b["count"] for b in bands]}, index=[b["band"] for b in bands]
        )
        st.bar_chart(band_df, use_container_width=True)
    st.caption("Points distribution")
    st.bar_chart(points_histogram(rows), use_container_width=True)

    st.divider()
    st.subheader("Method / mechanism")
    m1, m2 = st.columns([1, 2])
    with m1:
        st.caption("Method prefix")
        st.dataframe(method_breakdown(rows), use_container_width=True, hide_index=True)
    with m2:
        st.caption("Operator usage (tasks using each op)")
        ops = op_breakdown(by_op)
        if ops:
            top = ops[:20]
            op_df = pd.DataFrame(
                {"tasks": [o["count"] for o in top]}, index=[o["op"] for o in top]
            )
            st.bar_chart(op_df, use_container_width=True)
            with st.expander("All operators"):
                st.dataframe(ops, use_container_width=True, hide_index=True)
        else:
            st.info(f"No operator inventory at {INVENTORY_PATH}.")

    st.divider()
    st.subheader("All tasks")
    f1, f2, f3 = st.columns([1, 1, 1])
    band_filter = f1.multiselect("Score band", BANDS, default=BANDS)
    prefixes = sorted({r["method_prefix"] or "(none)" for r in rows})
    method_filter = f2.multiselect("Method", prefixes, default=prefixes)
    search = f3.text_input("Search task # or arc_id", "").strip().lower()

    df = pd.DataFrame(rows)
    df = df[df["band"].isin(band_filter)]
    df = df[df["method_prefix"].replace("", "(none)").isin(method_filter)]
    if search:
        mask = df["task"].astype(str).str.contains(search) | df["arc_id"].str.lower().str.contains(search)
        df = df[mask]
    df = df.sort_values("points", ascending=True).reset_index(drop=True)

    display_cols = [
        "task", "arc_id", "points", "memory", "params",
        "mem_params", "method", "headroom", "n_ops", "ops",
    ]
    st.caption(f"{len(df)} tasks (worst-scoring first). Select a row to open it in the Task Viewer.")
    event = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected = event.selection.rows if event and event.selection else []
    if selected:
        task = int(df.iloc[selected[0]]["task"])
        # Guard against re-triggering the switch when the user navigates back.
        if task != st.session_state.get("_drilled_task"):
            st.session_state["_drilled_task"] = task
            st.session_state["viewer_task"] = task
            viewer_page = st.session_state.get("_viewer_page")
            if viewer_page is not None:
                st.switch_page(viewer_page)
