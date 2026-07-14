# task202 — 855e0971

## 2026-06-30 (S5) — masked-profile sentinel fold (ADOPTED, +0.20)
**Before (main working tree):** mem 9513, params 23, pts 15.837.
The grid-extent mask `ingrid = rpos & cpos` was a 900B 30×30 broadcast plane, AND'd with `mark` into
`markgrid` (another 900B plane).
**Change:** fold padding-exclusion into the two 1-D profiles *before* the outer `Equal`:
`Am = rpos ? A : 99`, `Bm = cpos ? B : 100` (sentinels outside colour range [0,9]); then
`Equal(Am,Bm) == (A==B) & rpos & cpos ==` the old `markgrid` exactly. Deletes `ingrid` + `markgrid`
(2×900B) and the trailing `And`; replaced by two length-30 masked profiles. 30×30 carrier count 6→4.
**After: mem 7773, params 24, pts 16.039 (+0.202), fail 0.**
**Verify:** `fresh_verify.py 202 "" 2000` → new graph fail=0 on 1711 fresh arc-gen; agent equivalence
bit-identical to live-exact baseline (cand!=inc=0 / 1283 fresh). LANDED to main + manifest.
⭐ TRANSFERABLE: to AND a broadcast in-grid mask into an `Equal(profileA, profileB)`, don't build the
mask plane — map each profile's out-of-grid entries to **distinct sentinels** so the Equal is
automatically False there. Kills the mask broadcast carrier for free.

## 2026-06-30 — plane-free output routing (ADOPTED, +0.17)

Prior semantic graph (15.669 @ mem 11253) carried a uint8 label cascade
`bandc -> hb -> mark -> cg0 -> cg -> Equal(levels)`.  Two structural cuts:

1. **mark via OUTER Equal of 1-D profiles** — `mark = (band(r,c) == band marked
   at orthogonal coord)` is `Equal(A[.,1], B[1,.])` with
   `A = Where(isvert, hbV, rowcolor)`, `B = Where(isvert, colcolor, hbH)`.  No
   broadcast `hb` (900) carrier.
2. **route into the FREE output** — `output = Where(mark & ingrid, onehot0,
   input)`: marked cells -> black one-hot, else the input one-hot.  Drops the
   `cg0`/`cg` (2×900) uint8 label planes and the final `Equal(levels)`.

**Result: 15.837 pts @ mem 9513, params 23 — 230/230.**

**Verification (generator absent locally, like task001/201):** rebuilt the prior
graph and compared OLD vs NEW on **4000 well-formed random instances → identical
(0 mismatch)**.  They diverge ONLY on degenerate width-1 bands (where black marks
corrupt the `isvert` orientation profile) — which the real generator never emits
(0 same-line colour-collisions in all 230 stored).  So exact in-domain.

⭐ **TRANSFERABLE (task001 family):** when the output is "input one-hot with some
cells recoloured to a FIXED colour by a data-dependent 1-channel mask", never
build a uint8 label plane + `Equal(levels)`; route `Where(mask, fixed_one_hot,
input)` straight into the free output.  Kills the label carrier + its Equal.
Pairs with: replace any `Equal(broadcast_a, broadcast_b)` over two profiles with
an OUTER `Equal(a[.,1], b[1,.])` to drop the broadcast carriers.

## 2026-06-29 semantic source rewrite

Replaced the generated exact-preserve b64 scaffold in `src/custom/task202.py`
with a source-owned semantic builder for the current compact live graph.

Result: **stored 230/230**, mem **11253**, params **24**, points
**15.669479467767712**; **fresh 1000/1000**.  Source/live reconcile remains
`mismatches: 0`.

Adopt decision: **adopt as ownership/parity cleanup, not as a score improvement**.
This preserves the current live score and makes the mechanism explicit:
distinct-colour band profiles identify orientation and band colour, black cells
are contracted per row/column profile, and the final uint8 label map is routed
through the free `Equal(..., levels)` bool output.

**Rule:** The grid is fully painted with stacked horizontal colored "strata"
bands (each band = `height` rows of one DISTINCT color, colors from
`random_colors` so never repeat). Sparse black(0) pixels sit inside the bands.
For every black pixel at column c within a band, the output paints the ENTIRE
vertical extent of that band at column c black. When `xpose=1` the whole grid
(input+output) is transposed, so bands run vertically and the fill is
horizontal. Because band colors are distinct, two rows are in the same band iff
they share a non-black color — no contiguity/flood-fill needed.

**Current:** 14.79 pts, custom:task202 (prior adopted), mem 27063, params 40
**Target tier:** A (separable per-band row/col contraction routed into a bool
mask + a single Where into the free output; no per-cell colour-index plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice-ch0 black + [10,30] colblk/rowblk contraction, all fp16 | A | 22863 | 33 | 14.96 | — | marginal (+0.17) |
| 2 | full-4D batched matmuls, drop `black` reshape + obb reshape | A | 20163 | 29 | 15.087 | — | +0.295 (just short) |
| 3 | rank-3 occupancy [1,10,30] + native-4D black, MatMul rank-broadcast (no reshapes) | A | **18963** | **25** | **15.148** | 500/500 | **beats +0.3** |
| 4 | uint8 `QLinearMatMul` band contractions + uint8 black/ob planes | A | **13563** | **25** | **15.483** | 1000/1000 | **ADOPTED** |

## Best achieved
15.483 @ mem 13563 params 25 — adopted as `custom:task202+qlinear+onnxsim`.
Beats prior live 15.148 by **+0.335**. Fresh 1000/1000.

## Irreducible-floor analysis
Memory now dominated by:
- 3600B fp32 `blkslice` — the channel-0 (black) read. This is the unavoidable
  single fp32 "entry plane" for reading a colour spatially (the 3600B rule).
- 4×1800B fp16 full planes: `black` (the fp16 black map), `obR`, `obC`
  (the two orientation candidates), `ob` (the orientation Where-select).
- 2×1200B fp32 `mrf`/`mcf` — the per-colour row/col occupancy entry reductions
  (ReduceMax on the free input outputs fp32; [1,10,30] not [1,10,30,30] so only
  1200B each).
The remaining cost is the TWO orientation candidates obR/obC (3600B together):
the non-xpose op multiplies black on the LEFT (`Rrow^T@Rrow@black`) and the
xpose op on the RIGHT (`black@Rcol^T@Rcol`), so they cannot be unified by one
matmul, and selecting operands earlier would require a full `black^T` plane
(+1800B for the transpose, +1800B for the Where) — strictly worse than just
computing both [30,30] candidates. So we are at the practical floor for this
two-orientation closed form.

2026-06-28 update: the fp16 floor was false for the contraction path. The values are
small integer counts and the final use is only `Greater(ob, 0)`, so `black`, `colblk`,
`rowblk`, `obR`, `obC`, and `ob` can all stay uint8 via `QLinearMatMul`
(scale=1, zero-point=0). This cuts each full candidate plane 1800→900 while preserving
exact counts. `blkslice` remains a 3600B fp32 entry plane because it reads from the fp32 input.

## OPEN ANGLES (re-attack backlog)
- Single-orientation: if a future op let you select the orientation BEFORE the
  black plane is read (e.g. an orientation-conditioned Slice axis), you'd drop
  obC + RcolT + rowblk (~3000B). No opset-11 op does a data-dependent axis pick
  without a Where on full planes, so currently blocked.
- The `ob` Where (1800B) + `mask` Greater (900B) could in principle fold into
  per-branch bool masks (maskR/maskC + Where), but that nets to the same 2700B.

## INSIGHT (transferable)
⭐ ONNX MatMul **rank-broadcasting** lets you keep occupancy vectors at their
native ReduceMax shape `[1,10,30]` (rank-3) and the colour slice at its native
`[1,1,30,30]` (rank-4) and contract them directly — the lower-rank operand is
left-padded with 1s, batch dims broadcast, and the matrix dims contract. This
removed FOUR explicit Reshape intermediates (each a full extra plane) vs forcing
both operands to a common rank. General lever: when chaining per-channel matvecs
through the free fp32 input, do NOT reshape operands to a uniform rank — let
MatMul broadcast, and only reshape the final bool mask into [1,1,30,30] for the
output Where (and even that is free if the last matmul already lands 4D).
⭐ "Distinct per-band colour" (from `random_colors`) collapses same-band testing
to same-colour testing → the band-similarity routes through a tiny [10,30]
band×col count, never a [30,30] similarity matrix or any flood-fill.

## S8 (2026-07-02) — code-plane deletion via Cauchy uniformity (+0.784) ADOPTED, div 0
Beyond the u8-recast brief (blocked: QLC needs u8 input = 9000B cast): code_f/iszero/black
(5400B) DELETED. Band colours = Div(Σk·cnt, max(Σcnt,1)) from free-input einsums (int÷int
exact fp32); orientation = Cauchy equality Σcnt² == (Σcnt)² per row (single 3-operand einsum,
input repeated); black-cell band colour = S_h − Σ rowcolor·occ contracted vs FREE input.
3253+34 vs 6997+200 → 16.119→16.902. numpy 20000/20000; div 0 vs deployed on 2500.
TRICK for registry: Cauchy uniformity test (Σx²==(Σx)² ⟺ ≤1 nonzero) as an einsum.
