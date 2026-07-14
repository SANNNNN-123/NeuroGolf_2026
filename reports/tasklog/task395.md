# task395 — OR top/bottom slices → occupied/hole/zero small grid, pad out

## 2026-07-03 S13 — UNKNOWN triage: AT-FLOOR (int64→int32 lever = MEASURED ZERO)

**Cost (grader mem 144, params 21):** 5× INT64[3] slice inits (top/bottom start/end + shared
`axes`) + Pad INT64[6]. Rule: OR top/bottom slices → occupied/hole/zero 3-channel small grid, pad.

**KILL — false lever:** triage agent flagged "slice inits INT64→INT32 (split shared axes),
+0.123". MEASURED REFUTED: params = `math.prod(dims)` = element count, dtype-agnostic →
Δparams = 0 (21→21) even converting every int64 init. See task103 S13 for the full reasoning.
AT-FLOOR. Do not re-probe.