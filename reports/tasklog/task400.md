# task400 — ff805c23

**Rule:** A 24×24 grid carries a pattern with full D4 dihedral symmetry (the
8-element orbit {(r,c),(c,r),(r,n-c),(n-c,r),(n-r,c),(c,n-r),(n-r,n-c),(n-c,n-r)},
n=23, all share one colour). A 5×5 block of blue (colour 1, excluded from the
pattern palette) is stamped over part of it, occluding the pattern; the generator
guarantees every occluded cell still has ≥1 non-blue orbit member. Output = the
5×5 region under the cutout, reconstructed from symmetry.
**Current:** 14.53 pts (untriaged pending pool)
**Target tier:** A (closed-form symmetry reindex + data-dependent crop; no flood/argmax-over-components)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 full 24×24 symmetrize (8 planes+7 max), pad, Equal | A | 51428 | 56 | 14.15 | 200/200 | below floor |
| 2 | fp16 planes | A | 29106 | 62 | 14.72 | 200/200 | +0.19 |
| 3 | crop 5×5 from 8 orbit images (no full-plane sym) | A | 16606 | 52 | 15.28 | 200/200 | +0.75 |
| 4 | blue mask = Equal(g,1) (drop blue slice); uint8 L30 pad | A | 14483 | 43 | 15.42 | 200/200 | +0.89 |
| 5 | locate via ReduceMin(val) (drop bluef plane) | A | 13475 | 44 | 15.49 | 200/200 | +0.96 |
| 6 | drop valT: transpose-group = transposed 5×5 gather of val | A | 12523 | 44 | 15.56 | 500/500 | +1.03 |

## Best achieved
15.56 @ mem 12523 params 44 — adopted? N (build agent only). Beats prior 14.53? YES (+1.03).

## Irreducible-floor analysis
Dominant: colf32 [1,1,30,30] fp32 = 3600B — the colour-index contraction
(Σ k·input_k) of the 10-ch input; this is the documented 3600B plane floor and
cannot be narrowed (fp16/uint8 cast ADDS a plane; the Conv on 30×30 input must
emit 30×30). Next: colf fp16 1800 + g fp16 1152 (cast→slice to the 24×24 active
grid — required to drop the planes to fp16 for the orbit gathers); val 1152
(the sentineled plane every orbit gather reads); L30 uint8 900 (padded label
plane feeding the FREE one-hot output Equal). All small working planes are fp16;
the 8 orbit crops are 5×24→5×5 tiny.

## OPEN ANGLES (re-attack backlog)
- Fuse colf cast: currently Conv→fp32(3600)→Cast fp16(1800)→Slice(1152). Slicing
  fp32 first then casting was measured WORSE (+504). No cheaper route found to
  fp16 g without paying both the fp32 contraction and one fp16 cast plane.
- Could the 24×24 `val` plane be cropped before gathering? No — orbit indices
  span the full 0..23 range (both [b..b+4] and [23-b..23-b-4]), so the whole
  24×24 is genuinely needed.

## INSIGHT (transferable)
⭐ A D4-symmetric "fill the occluded cutout" task is closed-form Tier-A, NOT a
correspondence/connectivity bail: the 8 D4 pullbacks of the value plane are
exactly {I, T} × {none, flipR, flipC, flipRC} (0-param Transpose + step −1
Slices); P = elementwise MAX over the 8 (with occluded cells set to a losing
sentinel −1). ⭐ For a small FIXED-SIZE data-dependent crop of a symmetrized
plane, DON'T materialize the full symmetrized plane — gather the K×K crop
directly from each orbit image (row/col index sets are ascending `b+[0..k]` or
descending `(n-b)−[0..k]`) and max the tiny K×K blocks; this turned 8 full
24×24 planes + 7 full maxes (~17KB) into 8 tiny gathers. ⭐ The transpose-group
images need no separate `valT` plane: gather `val` with swapped (col,row) index
vectors and transpose the resulting K×K block. ⭐ Locate a sentinel-marked
region with ZERO extra plane via ReduceMin of the already-materialized value
plane + Less(<0)+ArgMax, instead of casting a fresh mask plane.

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
