# task295 — bbc9ae5d

## 2026-06-29 final-expansion screen

Current source score: 17.619744 @ mem 1557 params 47.

Dominant tensor is `full_label` [30,30] uint8 = 900 B, followed by the 9x18
working label/mask planes.  This is already the cheaper final-output route for
this rule: replacing the label plane with a compact one-hot before padding would
materialize roughly 9x18x10 bool cells, larger than the single uint8 label plane.

No rewrite adopted.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** a 9×18 working region's per-column/row label is expanded to the final 30×30 output; the net builds a full label plane then one-hots it.

**Cost (grader mem 1557, params 47):** ops Where×4/ReduceSum×3/Cast×3/Less×3/ArgMax/Mul/Add/Equal/Pad. Counted intermediates: `full_label` [30,30] uint8 900B (full-canvas label), `filled` [9,18] bool 162B, `default_label`/`small_label` [9,18] uint8 162B each. Params: `col_index` [1,18] fp16 36B, `pad_spec` [4] int64 32B, `row_index` [9,1] fp16 18B. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** full-output-carrier. The 900B `full_label` [30,30] uint8 is the dominant counted plane. The prior log established that a compact one-hot before padding would materialise ~9×18×10 bool cells (>900B) — the single uint8 label plane is already the cheaper final-expansion route.

**Lever:** no lever visible (log-confirmed floor; index params already fp16).
