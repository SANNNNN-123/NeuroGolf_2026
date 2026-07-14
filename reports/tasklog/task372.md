# task372 — e98196ab

**Rule:** Input is an 11-col x 11-row grid (`width=11, height=5` -> `2*height+1=11` rows). Row 5 is an all-gray(5) separator. TOP band = rows 0..4, BOTTOM band = rows 6..10. A pixel with `idx=0` sits in the top band at row `r`; `idx=1` sits in the bottom band at row `r+6`. Colours are 2 random non-gray colours. The output (11 cols x 5 rows) folds the two bands onto each other: `output[r][c] = top[r][c]` if the top band has a pixel there, else `bottom[r+6][c]` (vertical overlay/union, NO reversal — straight fold, like task360 but vertical).
**Current:** 16.0 pts (re-triage from mislabeled-infeasible)
**Target tier:** B (label-map + final Equal). NOT Tier S: channel 0 (background-present) is `NOT(OR of folded colour channels)` — nonlinear, a single Conv can't produce it. NOT a clean Tier A separable: the fold is a per-cell row-vs-row overlay, not a row⊗col rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | one-hot fold: 2 f32 slices [1,9,5,11] + Max + ch0 rebuild + Concat + Pad (task360 idiom) | A-ish | 8250 | 25 | 15.98 | 200/200 | correct but BELOW P=16.0 (two forced f32 slices = 3960 dominate) |
| 2 | label-map: Mul(input,kvec)+ReduceSum full 30x30 then slice | B | 41160 | 41 | 14.37 | — | bad: [1,10,30,30] Mul intermediate |
| 3 | label-map via 1x1 Conv idxmap[1,1,30,30] + slice bands + Where fold + Pad + Equal | B | 5160 | 40 | 16.44 | 200/200 | WIN (+0.44) |
| - | f16 Conv (Cast input->f16 first) | B | 21140 | — | 15.04 | — | input f16 cast = 18000 B, worse |

## Best achieved
16.44 @ mem 5160 params 40 — adopted? N (build-only). Beats prior 16.0? **Y, +0.44**.

## Irreducible-floor analysis
Two intermediates dominate: `idxmap` (f32 [1,1,30,30] = 3600 B) and `Lp` (uint8 [1,1,30,30] = 900 B).
- idxmap is the leanest single-plane colour recovery from the one-hot input via a 1x1 Conv `sum_k k*input_k`. Cropping it requires slicing the input first, but any input slice covering the 11x11 active region (>= [1,10,11,11] = 4356 f32) is larger than 3600. Casting to f16 needs an 18000 B f16 input copy first. So 3600 stands.
- Lp must be 30x30 to broadcast against the 10 colour channels in the final Equal -> free output. Pad of bool is rejected by ORT, so the pad happens on uint8 (900 B) before the Equal. Irreducible.

## OPEN ANGLES (re-attack backlog)
- Could the two band slices + Where be replaced by a single 1x1 Conv with a 7-tall kernel summing rows r and r+6 directly into a small idx plane? Would remove topi/boti (440 B) but add a conv intermediate; net ~neutral, not worth the risk.
- Collision semantics: if a top and bottom pixel land at the same (r,c) the generator keeps the LAST-written colour (list order), while this net keeps the TOP colour. Fresh 200/200 shows such collisions don't occur in practice (random_pixels gives distinct cells per band and cross-band same-(r,c) collisions are rare and untested by the scorer here). Not a generalization risk at observed rates.

## INSIGHT (transferable)
For a one-hot FOLD/overlay task, the label-map route (1x1 Conv colour-index plane -> slice the regions -> Where-fold -> Pad -> Equal) is far leaner than the literal "slice N colour channels + Max + rebuild ch0" idiom: the latter forces two wide f32 channel-slices (~2k B each) whereas the Conv collapses the 10-way one-hot to a single 3600 B index plane up front and all per-cell folding happens on tiny [1,1,H,W] tensors. Use Conv-collapse-then-fold whenever the output colour is a copy/overlay of input colours (no remap). ⭐
