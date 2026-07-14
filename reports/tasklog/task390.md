# task390 — f8a8fe49 (eject the gray contents out of the red box)

**Rule:** A red(2) box (full horizontal red edges at rows brow and B=brow+tall-1, plus
4 red pixels on the second/second-last rows of the left/right walls) sits on a 0
background. gray(5) "contents" are packed INSIDE the box in the INPUT. OUTPUT = red box
copied UNCHANGED + every interior gray pixel reflected OUTWARD across the NEARER
horizontal red edge: Rout = 2*brow - Rin if 2*Rin < brow+B else 2*B - Rin; COLUMN
preserved. brow = min full-red row, B = max full-red row. Half of instances are
TRANSPOSED (xpose=1): box has vertical red edges (cols bcol..Bc), gray reflects across
the nearer vertical edge with ROW preserved. Grid always 15x15; colours {0,2,5}.
Verified on 3000 fresh instances (both orientations) that the formula is exact.
**Current (prior):** ~14.89 pts (public/stored), tier A label.
**Target tier:** B — a data-dependent 1-D reflection on the box-edge axis (column/row
preserved). The reflection axis (brow/B or bcol/Bc) varies across the grid so no fixed
Conv/permute → S/A blocked; the remap is row-independent AND col-independent (each gray
moves on exactly one axis) so it IS the boolean double-MatMul Rmat@gray@CmatT, NOT a
rowcond⊗colcond Tier-A rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | red/gray fp32 slices; full-red-edge ReduceSum→min/max edge scalars; per-axis reflection matrix (ramp+Where+Equal); Where-gate FULL matrices vs EYE init by orientation; fp16 double-MatMul; uint8 label + Pad + Equal | B | 8235 | 301 | 15.95 | 200/200 | works |
| 2 | gate orientation on the tiny [W] remap VECTOR (active? refl(i):i) instead of two full matrices + EYE init → one Equal per axis builds Rmat/CmatT directly | B | **7456** | **76** | **16.07** | 300/300 | FINAL |

## Best achieved
**16.07 pts @ mem 7456, params 76 — 266/266 stored, fresh 300/300.** Adopted? **N**
(orchestrator gates adoption). Beats prior ~14.89? **Y (+1.18).** GENERALIZES (both
train+test orientations; 300/300 fresh isolated).

## Irreducible-floor analysis
Dominant intermediates (all structural): red fp32 slice 900B (load-bearing twice — the
edge-count ReduceSum needs float, and the same plane is the box mask `redb`; ReduceSum
rejects uint8 so it cannot be the uint8 box plane), gray fp32 slice 900B (Slice preserves
fp32; cast to fp16 for the MatMul), the uint8 L Pad 900B (output is 30x30 — the off-canvas
sentinel is irreducible), and the fp16 MatMul chain gray16/Rmat/CmatT/rowmapped/colmapped
= 5×450B. Both MatMuls are needed because the active reflection axis differs per instance
(orientation is data-dependent), and a single MatMul contracts one fixed axis only; the
identity-gated branch still materializes its output. fp16 is exact ({0,1} permutation,
sums < 2^11). Everything else is ≤60B ([W] remap vectors / scalars).

## OPEN ANGLES (re-attack backlog)
- Shrink the MatMul canvas below 15×15: gray ejects to grid rows {0,1,...,14} (top edge
  brow≈3, ejected up to row 0; bottom ejected to ~row 14) so the full height is used — a
  data-dependent crop would be a Gather (≥100B + index machinery), likely net-neutral.
- Single-MatMul orientation fold: route gray through one reflection on the correct axis
  via a transpose chosen by orientation — would drop one MatMul (~450B) but adds a
  data-dependent Transpose/Where over a full plane; net unclear, untried.
- Avoid the gray fp32 slice by reflecting a combined colour-index plane and re-splitting —
  no byte win (still one 900B fp32 plane + the fp16 cast).

## INSIGHT (transferable)
⭐ **An ORIENTATION-DEPENDENT 1-D reflection (xpose flips which axis the box-edge
reflection acts on) is the task250 double-MatMul idiom with PER-AXIS GATING folded into
the tiny [W] remap VECTOR, not a second full matrix:** build refl(i) = Where(2i<lo+hi,
2*lo-i, 2*hi-i) from the min/max FULL-edge scalars, then `rvec = Where(active, refl, i)`
so the inactive axis becomes identity — ONE Equal(rvec, out_arange) yields the gating-built
permutation matrix. Detect orientation as "does a full red row (ReduceSum≥edge_width)
exist", and reuse `Not(is_row_box)` for the column axis. This kills the separate EYE
identity initializer (−225 params) and the two full-matrix Where selects (−2×450B mem)
vs gating at the matrix level — gate scalars/vectors, never full planes.
⭐ Recover a box's reflection AXES (not just its corner) from FULL-edge detection:
per-row red count = ReduceSum(red,[col-axis]); a "full edge" row = count ≥ box-width(5),
and min/max of the full-edge indices give the two reflection axes directly — this also
distinguishes orientation for free (full edges are horizontal ⇔ row-box).

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2624/16.96; color_padded 900B min pre-Equal carrier (15x15 Equal=2250B worse), red_core_code_f 400B forced-fp32, gray_indices ScatterND minimal. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.000)
**Mechanism (op-census diff):** Fused And+Not into one `Xor` (And 17→16, Not 4→3, +Xor; 75→74 nodes). −1B.
**Old→new:** mem 2624→2623, params 467→467.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task390_pre_s10.onnx`; source `public_candidates/bobmyers7186/task390.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
