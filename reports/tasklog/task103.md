# task103 — compare left/right 3-col slices, emit equality/hole flags

## 2026-07-03 S13 — UNKNOWN triage: AT-FLOOR (int64→int32 lever = MEASURED ZERO)

**Cost (grader mem 36, params 24):** 5× INT64[3] Slice-index inits + Pad INT64[8].
Rule: compare left vs right 3-column slices, emit equality/hole flags.

**KILL — false lever:** a triage agent flagged "convert 5 Slice-index inits INT64→INT32,
120→60B, +0.317". MEASURED REFUTED: `src.harness.calculate_params` counts
`math.prod(init.dims)` = ELEMENT COUNT (dtype-independent). Converting all 6 int64 inits to
int32 → Δparams = 0 (24→24). Slice-index inits are not node outputs → not in `calculate_memory`
either. The agent's "+0.317" was raw-onnx byte-counting, which does NOT match the grader.
This is the anti-pattern already in memory safe-golf-301-400 line 53–55. AT-FLOOR. Do not re-probe.