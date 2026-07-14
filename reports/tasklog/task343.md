# task343 — d8c310e9

**Rule:** Input/output are 5x15 grids. A vertical stripe block of width
`len(lengths)` (3 or 4) is laid out column by column; with `flip=1` every other
block is column-reversed, so the drawn pattern is periodic with EFFECTIVE PERIOD
`EP = len(lengths)` (flip=0) or `EP = 2*len(lengths)` (flip=1), EP in {1,2,3,4,6,8}.
Only `visible` columns are drawn in the input; the OUTPUT tiles the same periodic
pattern across all 15 columns. The input always already holds a full effective
period in columns [0,EP), so `out(r,c) = input(r, c mod EP)` (verified 3000/3000
fresh). Off-grid columns 15..29 (and rows 5..29) are all-zero.
**Current:** 14.78 pts (prior), gen-based; **new** mem 1686, params 676
**Target tier:** A — the whole output is one column Gather of the input (FREE);
only period detection costs memory. (Not S: a fixed Conv can't route since EP is
data-dependent, so a small period-detection scaffold is required.)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Gather(input, c%EP, axis=3); EP from per-col sig + occ; fg-slice for occ | A | 37208 | 801 | 14.45 | — | fail-mem (8100B fg plane) |
| 2 | derive occ from colsig>0 (drop [1,9,30,30] fg slice) | A | 4508 | 797 | 16.42 | — | pass |
| 3 | fold rowW+chW into ONE Conv kernel [1,10,30,1] (no [1,10,1,30]) | A | 3308 | 1057 | 16.62 | — | pass |
| 4 | MAXP 15->8 (max real EP=8) | A | 2062 | 735 | 17.06 | — | pass |
| 5 | fp16 the bad-count Cast/ReduceSum plane | A | 1806 | 736 | 17.16 | — | pass |
| 6 | fold off-grid pad29 into periodtab (drop Where/keepmask) | A | 1686 | 676 | 17.23 | 500/500 | PASS |

## Best achieved
17.23 @ mem 1686 params 676 — adopted? N (build-only). Beats prior 14.78? Y (+2.45).

## Irreducible-floor analysis
Dominant intermediate is `sigAtMod` [8,15] fp32 = 480B — `Gather(colsigV, modmat)`
giving colsig[c mod p] for every candidate period p (1..8). It must stay fp32:
the column signature uses base-10 row weights up to 10000 (5 rows x colour 0-9) to
guarantee column injectivity, which exceeds the fp16 exact-integer range (2048), so
fp16 would risk silent column collisions -> wrong-period mis-detection. The Conv
kernel `sigK` [1,10,30,1] = 300 params is the largest init (height 30 to contract
all rows in one pass); slicing to 5 rows first would add a 6000B fp32 plane (worse).
Everything else is [8,15]/[30]/[15] vectors. The final Gather IS the output (free).

## OPEN ANGLES (re-attack backlog)
- Shrink `sigAtMod` 480B->240B by finding a fp16-exact (<=2048) yet collision-safe
  column signature. 5 rows x colour 0-9 can't be globally injective under 2048, but
  a weaker hash that is only injective among *columns actually present* might hold;
  needs a careful collision-bound argument (risky, could break genverify silently).
- `periodtab` 240 + `modmat` 120 params: a single [8,15] table reused for both the
  mod-gather and the source index (with a Concat to 30) could trim ~120 params.

## INSIGHT (transferable)
⭐ "Horizontal periodic tile/extend to full width" generalizes task231 to a
DATA-DEPENDENT period chosen by smallest-consistent-period search: test all p in
1..MAXP at once via `Gather(colsig, [c mod p][MAXP,W])`, AND mismatches with
occupancy, ReduceSum-per-row==0 => consistent, then `ArgMax(consistent * decreasing
ramp)` picks the SMALLEST consistent p with no Loop/NonZero. ⭐ Per-column
occupancy needs NO [1,9,30,30] fg-slice: with a base-weighted colour signature
`colsig[c] = Σ_{ch,r} chW[ch]·rowW[r]·input` (bg colour-index = 0), `colsig > 0`
IS the occupancy — one tiny [30] vector serves both period detection and extent.
⭐ Fold BOTH the channel and row contractions into ONE Conv kernel
[1,out=1,in=10,kh=30,kw=1] (no pad) to get the [1,1,1,30] signature directly,
eliminating the [1,10,1,30] MatMul intermediate (1200B -> 0). ⭐ Bake the off-grid
pad-column redirect (c>=W -> col 29) straight into the period table so the gathered
row is already the final source index — no separate Where/keep-mask plane.

## 2026-06-29 direct one-hot Gather adoption

Re-attacked the final output path.  The previous source built a compact label
grid, then paid for `Equal(output_ids, channel_ids)` over `[1,10,5,15]` and
`Pad` to the harness canvas.  But the generator guarantees the output is just
the input's visible period repeated horizontally:

`output[:, :, :, c] = input[:, :, :, c mod period]`

for columns `0..14`, while columns `15..29` are off-grid all-zero.  Therefore
the final graph can concatenate `chosen_cols` with fifteen copies of input
column `29`, then make the graph output directly:

`Gather(input, final_cols, axis=3) -> output`

This deletes the label-to-one-hot Equal/Pad path and lets the 10-channel result
be the free graph output.

Result:

- stored: `mem=1927, params=110, pts=17.380766` ->
  `mem=1147, params=75, pts=17.891756`;
- fresh: `1000/1000`, then `5000/5000`;
- adopted with `python -m src.adopt 343`.

Transferable rule: when the output is a pure crop/periodic remap of the original
one-hot input, route the original one-hot tensor to the graph output with Gather
instead of rebuilding labels and expanding them with Equal.  Off-grid columns can
point at an already-zero padded input column to avoid Pad entirely.
