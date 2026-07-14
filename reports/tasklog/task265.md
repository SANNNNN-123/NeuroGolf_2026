# task265 — ARC-AGI a8d7556c

**Rule:** 18×18 grid of gray(5) static + black(0) holes. Two stripe passes paint
red(2) over 2×2 all-black "holes", mutating the output between passes (generator
guarantees order [downstripe,sidestripe] == [sidestripe,downstripe], non-ambiguous).
Pass 0 (downstripe): a 2×2 all-black hole (r,c) is painted unless a horizontal
side-column is fully black: fail0 = (col c-1 both rows black) OR (col c+2 both rows
black). Pass 1 (sidestripe): paint a hole iff it is STILL all-black (no cell painted
in pass 0) AND no vertical side-row is fully black: fail1 = (row r-1 both cols black)
OR (row r+2 both cols black). A cell is red iff in any painted hole.

**Current:** 15.250 pts, custom:task265 (literal 2-pass cascade), mem 17100, params 50
**Target tier:** B/detection — local convolution cascade; no scalar/copy collapse
possible (per-cell painting over the full 18×18 active grid).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | OR-of-both-passes on ORIGINAL grid (drop statefulness) | B | 12564 | 50 | — | 200/200 | FAIL 1 stored: over-paints a sidestripe-only hole overlapping a downstripe red |
| 2 | e1 via red0-overlap exclusion, literal | B | 18396 | 50 | 15.177 | 200/200 | correct but added planes |
| 3 | banded 2×4 / 4×2 conv per pass (v = holeCount + 10·leftPair + 100·rightPair → painted ⇔ v∈{4,14,104,114}); pass1 emptiness = e0 ∧ ¬(2×2 expand of red0); fail1 on ORIGINAL grid | B | 15480 | 56 | **15.349** | 200/200 | **best**, 266/266 stored |

## Best achieved
15.349 @ mem 15480 params 56 — adopted? **N** (only +0.099 over P=15.25, < +0.3
threshold → MARGINAL). Beats prior 15.250? Y, but marginally.

## Irreducible-floor analysis
The score uses the SUM of every value_info tensor at its declared 18×18 size (~34
tensors), not just the traced peak. Fixed/unavoidable cost ≈ 9k B: fp32 entry Slice
blk32 (1296, Slice forces fp32), blkf+blk (972), the uint8 label chain base+L18+L
(324+324+900=1548), and the two 2×2 red expansions (red0/red1, ~1620 each). The two
banded-conv MEMBERSHIP tests cost 14 bool planes (4 Equal + 3 Or each = 7×324 = 4536).
The set {4,14,104,114} is the product {h=4}×{L≤1}×{R≤1}; a linear conv can't separate
"L=1,R=1 valid" from "L=2,R=0 fail" with one threshold (same L+R sum), and Mod folds
one band away (R=2→same residue as R=0), so membership has a hard ≥4-op floor.
Reaching +0.3 would need mem+params ≤ ~12700, i.e. shedding ~8 more 18×18 planes — not
possible while keeping both stateful passes (both are required: downstripe-only fails
1 stored, sidestripe-only fails 4).

## OPEN ANGLES (re-attack backlog)
- Single-op membership: pick conv weights + ONE Mod/Greater that isolates {h==4 ∧ L≤1
  ∧ R≤1}. All linear/Mod variants tried collide; an exotic weighting (e.g. base-3 packing
  so each band is a clean digit, then ONE Equal on a transformed value) might cut the 14
  membership planes to ~4 → est mem ≈ 11.6k → ≈15.64 (+0.39, ADOPTABLE). Highest-value lever.
- Eliminate the holeHasRed0 expansion chain (red0f+conv+Greater+Not ≈ 1944 B) by deriving
  pass-1 emptiness directly from the banded value rather than from expanded red0.
- Fuse base/L18/L: build the 0/2/5/10 label in one op (e.g. 5−5·blk−3·red) to drop the
  `base` plane — blocked by ORT rejecting uint8 Mul; would need an fp16 detour (net wash).

## INSIGHT (transferable)
⭐ For an ORDER-DEPENDENT two-pass paint generator that guarantees both orders agree
(non-ambiguous), the naive "OR of both passes on the original grid" is NOT exact — a
later pass's emptiness test reads the EARLIER pass's mutations, so a hole that qualifies
only via the second pass but overlaps a first-pass paint must be excluded. The clean fix
is one-directional: keep the first pass on the original grid, and for the second pass
gate emptiness by "no cell already painted" (a 2×2 expansion of pass-0 reds) while its
FAIL/legality test may still be evaluated on the ORIGINAL grid (verified 500/500). Also:
a banded conv (band weights 1/10/100 reading a small neighborhood) replaces 3 separate
predicate convs, but only wins when the painted condition is a SINGLE clean threshold —
for a 2-D product-of-counts condition the membership disjunction reintroduces ~7 bool
planes and the net gain shrinks.

## 2026-06-30 — S6 re-confirm: FLOOR (input-side dead, external incumbent best)
- Incumbent is NOT the stale custom 2-pass cascade above (15.25). Manifest shows
  `method=ext:franksunp7166_65`, **pts=16.619, mem=4328, params=34** — a previous
  session already adopted the external public-teacher net, well above the custom best.
- S6 primary queue listed 265 as LOCAL_1 cov=0.996 (+1.58 input-side headroom).
  Re-ran `reports/scripts/conv_fit.py 265` → **no single-Conv fit** (k=1/3/5 all FAIL,
  channel 0 not separable, ≥4435 err on 300 ex). Confirms playbook §5.1: the LOCAL_1
  label is a collision-free-window artifact, NOT linear separability. A nonlinear-local
  fit needs a hidden 3600B fp32 plane → no win. **Input-side = measured dry well.**
- Output is per-cell red painting over the full 18×18 active grid (not ≤K separable
  rects) → signed-Einsum output routing does not apply. **FLOOR.**

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k3(cost 4362): LP-proven infeasible(838 패치); k5: LP-proven infeasible(20k subset). 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
