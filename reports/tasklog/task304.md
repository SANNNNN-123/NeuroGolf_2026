# task304 — c3e719e8

## 2026-06-29 label-vs-onehot screen

Current source score: 17.726907 @ mem 1404 params 37.

Rule: from a 3x3 colour grid, find the majority colour and expand only majority
cells into copies of the whole 3x3 pattern, producing a 9x9 output.

The source uses `DepthToSpace` to build a [1,1,9,9] colour label plane and then
`Equal(colors9, color_codes)` to produce a [1,10,9,9] bool one-hot (`y9`, 810 B)
before padding to output.  A full-canvas label route would require a 30x30 uint8
label (900 B) before final `Equal`, so the current 9x9 one-hot is cheaper.

No rewrite adopted.
