#!/usr/bin/env python3
"""Backfill validation gate: each mechanism's saved query must surface ALL of its known-applied
(seeded) tasks from the real task index. This is the regression test that keeps saved queries
honest as task_index.json / probes evolve."""
import importlib.util, json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]


def _mod(name, rel):
    import sys; sys.path.append(str(ROOT / "reports/scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.mark.parametrize(
    "mechanism",
    [
        "topk_width",
        "walk_einsum",
        "signed_rect",
        "einsum_vs_free_input",
        "qlinearconv_render",
        pytest.param(
            "gridsample_warp",
            marks=pytest.mark.xfail(
                reason=(
                    "saved query d4_transform_of_input can't surface crop-shaped seed task209; "
                    "needs a geometric-warp semantic probe — see .superpowers/sdd/progress.md follow-up"
                ),
                strict=True,
            ),
        ),
    ],
)
def test_saved_query_surfaces_known_applied_tasks(mechanism):
    if not (ROOT / "reports/task_index.json").exists():
        pytest.skip("task_index.json not built yet")
    match = _mod("match_insight", "reports/scripts/match_insight.py")
    cov = _mod("coverage_lib", "reports/scripts/coverage_lib.py")
    query = cov.load().get(mechanism, {}).get("query")
    if not query:
        pytest.skip(f"no saved query for {mechanism}")
    known = set(cov.SEED[mechanism])
    surfaced = {t for t, _ in match.match(query, mechanism=mechanism, include_resolved=True)}
    missing = known - surfaced
    assert not missing, f"{mechanism} query failed to surface known tasks {missing}"
