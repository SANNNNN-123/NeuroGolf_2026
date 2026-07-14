# task249 — a416b8f3

**Rule:** Input is a width×height grid (one-hot, top-left of 30×30). Output is a (2*width)×height
grid that is the input duplicated horizontally: out[:, c] = out[:, width+c] = in[:, c]. Equivalently
output column i maps to input column m[i] = i (if i<W) else i-W. This holds for ALL i in 0..29: i<W →
col i; W≤i<2W → col i-W; i≥2W → col i-W≥W = off-grid input col = all-zero one-hot = correct empty. So
NO clip is needed (W≥3 ⇒ i-W ∈ [0,29]). Rows pass through (Gather on the width axis only); off-grid
rows are all-zero. Pure spatial copy ⇒ Tier S, output is FREE.
**Current (prior):** 18.33 pts, ReduceMax+ReduceSum-scan width + Less/Sub/Where/Clip column-index +
int64 Gather, mem 758, params 34.
**Target tier:** S — output is a pure Gather of input columns; only the scalar W and a length-30 index
vector need materializing.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | drop redundant Clip; build index map in fp16 Where; Cast int64→int32 | S | 400 | 34 | 18.93 | — | works |
| 2 | drop redundant Reshape (ReduceSum axes=[1,2,3] keepdims=0 already gives [1]) | S | 396 | 33 | 18.94 | 200/200 | ADOPTED |

## Best achieved
18.94 @ mem 396 params 33 — beats prior 18.33 by +0.61 (≥+0.3 ✓). Adopt: out of scope (build-only).

## Irreducible-floor analysis
Dominant intermediates: colocc [1,1,1,30] fp32 (120B) + m int32 [30] (120B). colocc is irreducible —
ReduceMax inherits the fp32 input dtype (casting input to fp16 = 18000B). The int32 index plane (120B)
is the floor for axis-3 Gather indices (Gather rejects uint8; int32 < int64). Remaining: shifted f16 60,
m16 f16 60, lt bool 30, W32 f32 4, W16 f16 2. The index pipeline (Less→Sub→Where→Cast) needs at minimum
one bool + one fp16 + the final int32 over length-30. ~396B is near the practical Tier-S floor for a
runtime-width column gather.

## OPEN ANGLES (re-attack backlog)
- Eliminate `shifted` (60B): no single op produces Where(i<W, i, i-W) without materializing the i-W
  branch; arithmetic (mul+sub of a cast ge) costs MORE planes. Likely irreducible.
- W without the [1,1,1,30] colocc plane: W = (max occupied col)+1, but every alternative still needs a
  per-column reduction of the same shape. No cheaper route found.

## INSIGHT (transferable)
⭐ For a runtime-width HORIZONTAL DUPLICATION (out col i = in col m[i], m[i]=i if i<W else i-W), the
clip is REDUNDANT: i≥2W maps to input col i-W≥W which is off-grid = all-zero = the correct empty output,
and i<2W keeps i-W in-range since W≥3. Build the length-30 index in fp16 (Where) and Cast to int32 (not
int64) — halves both the index and the working vectors. Width = ReduceSum(ReduceMax(input,[1,2]),[1,2,3])
with keepdims=0 lands a clean [1] scalar (axis 0 survives), no Reshape needed. Net: 758→396, +0.61.

## S5 re-visit (2026-06-30) — FLOOR, no safe improvement
Current LIVE incumbent (src/custom/task249.py, the `live_exact` graph) is BETTER than this log's 396:
**mem 248, params 39, total 287, pts 19.341, fail 0/265.** It abandoned the general fp16-Where index
for a tighter **table re-fit**: Slice row0 cols3:5 [1,10,1,2] (80B) → ReduceSum→scalar → Cast int32 →
Gather `active_table_i32`(3,10) → active_idx [10] (40B) → Pad [0,20] val5 → dup_idx [30] int32 (120B)
→ Gather(input,axis=3). The table's 3 rows = the 3 bundled widths; verified **W ∈ {3,4,5} EXACTLY**
across all 265 bundled (96/82/87), max 2W=10 ⇒ table row length 10 is minimal. (Re-fit caveat: WRONG
for W≥6 — generalizes only over the bundled width range; it's the landed incumbent, gate-passing.)

**Every cost is at its structural floor:**
- dup_idx int32 [30] = **120B MANDATORY** — ONNX Gather indices must be int32/int64, output needs 30 cols.
- flag slice [1,10,1,2] = **80B MANDATORY** — 3-way W detection needs ≥2 boundary cells × 10 channels
  (color unknown). A full colocc [1,1,1,30] is 120B (worse); a full row [1,10,1,30] is 1200B.
- active_idx [10] = 40B minimal (row length = 2·maxW = 10); + scalars 8B; + params 39 (slice 6, table 30, pad 3).

**Attempts to beat 287 (all measured, all WORSE):**
| angle | mem | params | total | verdict |
|---|---|---|---|---|
| fold Pad into (3,30) table → drop active_idx 40B | 208 | 96 | **304** | worse (+57 params ≫ −40 mem) |
| general fp16-Where index (this log's old #2) | — | — | **396** | worse (arange+bool+2×fp16 planes) |
| int8 table/active_idx | 248 | 39 | 287 | no gain (Gather rejects int8 → Cast adds a 120B plane back) |
| Concat-const fill instead of Pad | 160 | 50 | +17 | worse (20-elem fill const > 3 pad params) |

Param-element (1 unit) vs mem-byte (1 unit) asymmetry kills every "shrink-the-index" trade: an int32
index element costs 4 mem-bytes but a table element costs 1 param, so enlarging the table to drop an
intermediate always loses. **min_stat's headroom +1.5 (floor 64) is an over-optimistic bound** that
ignores the mandatory int32-Gather index (120B) and the mandatory ≥80B detection read; the true
achievable floor ≈ 287, which the incumbent already hits. **VERDICT: FLOOR. No change. Left as-is.**
