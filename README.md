# NeuroGolf 2026

**Goal:** design the *smallest* neural networks that solve ARC-AGI image transformations. Correctness is required; score is almost entirely about how cheap the nets are.

## Competition in one page

| | |
|---|---|
| Tasks | 400 ARC-AGI-style grid transforms (`task001` … `task400`) |
| Deliverable | One ONNX net per task (`taskNNN.onnx`), packed as `submission.zip` |
| Max per file | 1.44 MB |
| Daily submits | 100 |
| Prize pool | $50,000 |

Each task is a few train input→output grids plus a held-out test. Your ONNX model must map one-hot grid tensors to the correct output grid.

### Scoring

```
points = max(1, 25 - ln(params + memory))
```

- **params** — element count of all ONNX initializers
- **memory** — bytes of intermediate tensors (shape-inferred; **input/output are free**)
- Unsolved / failing task → **0** points
- Theoretical max ≈ **10,000** (400 × ~25)

Leaderboard tops sit around **~8000**. This repo’s north star is to push toward that ceiling.

## Repository layout

```
data/taskNNN.json          # ARC task JSON (train/test examples)
src/custom/taskNNN.py      # Source of truth — build logic per task
networks/taskNNN.onnx      # Built nets (local artifact; usually gitignored)
reports/                   # Scoreboard, tasklogs, insight registry, scripts
submission/                # Active submit pack + helpers
tools/                     # Local dashboard / ONNX viewer
skills/                    # Agent playbook for recursive score work
```

**Source-owned workflow:** edit `src/custom/`, rebuild ONNX, verify, then pack. Public nets are teachers only — don’t blind-import them as the final answer.

## Quick start

```bash
# measure one task
PYTHONPATH=. .venv/bin/python reports/scripts/measure_task.py 42

# rebuild nets from source builders
PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py

# keep source ↔ live ONNX reconciled (must stay mismatches: 0)
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py

# pack + submit (archive name must be exactly submission.zip)
PYTHONPATH=. .venv/bin/python -c "from src.pipeline import pack; pack()"
kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "note"
```

More detail: [`AGENTIC_WORKFLOW.md`](AGENTIC_WORKFLOW.md)

## Strategy (this repo)

1. **Exploit free `input`/`output`** — route work so counted intermediate planes disappear.
2. **Just pass the LB gate** — bundled fail=0 + smaller mem/params; don’t over-engineer past what scores.
3. **Mine mechanisms, not single nets** — when a public solution wins a task, extract *why*, log it in `reports/insight_registry.yaml`, and apply it across every matching task.
4. **Submit freely** — 100/day; Kaggle keeps the best.

Local score state lives in `reports/manifest.json` / `reports/SCOREBOARD.md`.
