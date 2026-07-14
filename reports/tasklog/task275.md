# task275 — b190f7f5

**Rule:** Input packs TWO `s x s` sub-grids (s in {3,4}) into adjacent halves (4 `pairwise`
layouts: color-left/right or color-top/bottom). One half is a COLOUR grid C (sparse cells in
{1,2,3,4}); the other is a PLUS/cyan mask P (sparse cyan=8). Output is the `s²×s²` Kronecker
product: `output[row*s+r][col*s+c] = C[row][col]` for every coloured cell `(row,col)` of C and
every set cell `(r,c)` of P. Equivalently `output[R,C] = C[R//s, C//s]` iff that macro cell is
coloured AND `P[R%s, C%s]` is set, else 0; cells outside the `s²×s²` footprint are all-off.
**Current:** 14.2 pts (was mislabeled-infeasible; re-triage FEASIBLE).
**Target tier:** A/B (Kronecker label-map) — the colour value is content-dependent per macro
cell, so a pure single-op Tier-S transform is impossible (must READ the colour grid → ~3600 floor
for the one combined value plane). Reached 15.11 @ mem 18091.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-plane slices (ch1-9, ch1-4) + Kron assembly | B | 89299 | 1690 | 13.58 | — | correct but huge |
| 2 | early spatial-reduce profiles (1200B) | B | 33619 | — | 14.53 | — | beats baseline |
| 3 | Conv colour-value plane (params not mem) | B | 26163 | 1687 | 14.77 | — | — |
| 4 | merge colour+cyan into ONE combined plane (disjoint halves) | B | 20403 | 1687 | 15.00 | — | — |
| 5 | scalar-index size-table Gather (drops [1,256] dup) | A/B | 18091 | 1685 | 15.11 | 5000/5000 | ADOPT-READY |
| 6 | CROP input to top-left 8×8 (max dim 2s≤8); 8×8 Conv+profiles | B | 12467 | 1625 | 15.45 | — | win |
| 7 | + fp16 the value plane → fp16 [256] out_col/out_plus | B | 11315 | 1625 | 15.53 | — | win |
| 8 | + derive profiles from value plane (drop 10-ch profiles) | B | 10227 | 1623 | 15.62 | — | win |
| 9 | + split slice to {colour ch1:5}+{cyan ch8} only, separate planes | B | 9075 | 1622 | 15.72 | 200/200 | ADOPT |

## Best achieved (re-golf 2026-06-19)
**15.7223 @ mem 9075, params 1622** — beats prior adopted 15.11 by **+0.61** (≥+0.3 ✓). fresh 200/200.

## Re-golf floor analysis
The old "cplane 3600B is the genuine floor" was WRONG: the active region is bounded to the
top-left (2s)×(2s) ≤ 8×8 (size s≤4, grid anchored at (0,0)), so the value Conv plane and all
profiles run at 8×8 (256B) not 30×30 (3600B). Channels 5,6,7 carry no signal → slice only
{colour 1:5}=1024B and {cyan 8:9}=256B as two tight fp32 entries, keep them as separate small
value planes (no value-8 merge needed). New dominant: macro_v/micro_v int32 [256]=1024B each
(Gather indices reject narrower dtypes), out_col/out_plus fp16 [256]=512B each, L uint8 30×30
=900B (pad-back floor), colslice fp32 1024B (entry, Slice keeps fp32). The 16×16 canvas index
tables are the true remaining floor.

## Best achieved (prior)
**15.11 @ mem 18091, params 1685** — beats prior 14.2 by **+0.91** (≥+0.3 YES). fresh 5000/5000.

## Irreducible-floor analysis
Dominant intermediates: the one combined value plane `cplane` [1,1,30,30] fp32 = 3600B (Conv
output, reads the per-cell colour value 1..4 / cyan 8 — content-dependent, the "3600 rule" floor),
+ the two 1-D channel profiles `rowprof`/`colprof` [1,10,30,1]/[1,10,1,30] = 1200B each (needed
to detect size + layout before knowing which halves to gather), + the [256]-element Kronecker
working vectors (macro/micro int32 indices + out_col/out_plus values, ~1024 each). Everything is
already routed: the 10-channel expansion is the FREE BOOL `output = Equal(L, arange)` (opset 11).

## OPEN ANGLES (re-attack backlog)
- The two 1200B profiles could perhaps fold into one if size+layout were detected from a single
  reduced tensor — but vertical/horizontal needs both row and col extents.
- The [256] index/value vectors (~4×1024) could shrink to uint8 if ORT had uint8 Greater (it does
  not — int8/int16 Where/Equal NOT_IMPLEMENTED; casting to float for the compare costs more than it
  saves, tried & reverted).
- cplane 3600B is the genuine floor: reading a per-macro-cell colour value requires one full plane.

## INSIGHT (transferable)
- ⭐ **Kronecker with VARIABLE size:** when the tiling factor `s` varies, precompute the
  macro/micro flat-index maps for EACH size as a `[K, 256]` table and select the row with a SCALAR
  Gather (scalar index drops the gathered axis → no `[1,256]` duplicate, saves ~2KB). Then Gather a
  tiny `4×4` grid by those maps — no per-cell colour image needed.
- ⭐ **Merge disjoint sub-grids into ONE value plane:** when colour cells (1-4) and a marker (cyan
  8) live in DISJOINT regions, a single 1×1 Conv weight `[0,1,2,3,4,0,0,0,8,0]` yields one combined
  plane; the colour block reads 1-4, the plus block reads 8 — halves the 30×30 plane count.
- ⭐ **Variable grid SIZE recovery from bbox is only ~exact:** s∈{3,4} is decided by
  `s4 = (maxExtent≥6) OR (minExtent≥3) OR (lowerRegionSplitMax≥3)` — zero false-positives, ~3e-6
  miss rate (size-4 grids that collapse into a 6×3 footprint with a gap row/col). Exact-canvas
  collisions are 0/1M (the residual needs the colour PATTERN, unusable as a clean rule). 5000/5000
  fresh passed; safe for the 200-instance gate.
- ⭐ **Early spatial reduction beats channel slicing:** `ReduceMax(input, axes=[3])` → `[1,10,30,1]`
  (1200B) then slice channels gives all 1-D profiles cheaply; never slice `input` to `[1,9,30,30]`
  (32400B). And `Conv(input, 1×1 W)` puts the channel-weighting in PARAMS, not the 36000B Mul.
