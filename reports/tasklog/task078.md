# task078 — 3906de3d

## 2026-06-29 compact-column-fill screen

Current source score: 17.835280 @ mem 1260 params 33.

Rule: blue top column heights and red bottom column heights are rearranged into
a top-blue / middle-red / background column fill in a 10x10 canvas.

The current graph slices only the relevant fixed colour channels, reduces each
column to top/bottom counts, builds three 10x10 bool masks (`background`, `blue`,
`red`), concatenates them as [1,3,10,10], and pads directly to the free output.

`conv_fit.py 78` failed for k=1/3/5.  No rewrite adopted.  The output height per
column is data-dependent; the compact three-channel pad is already cheaper than a
full-canvas label-map path.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** blue top-column heights and red bottom-column heights are rearranged into a per-column fill: top band blue, middle band red, rest background, on a 10×10 canvas.

**Cost (grader mem 1260, params 33):** ops Slice×2/ReduceL1×2/Less×2/Add/Xor/Not/Concat/Pad. Counted intermediates: `out3` [1,3,10,10] bool 300B (compact 3-channel emission), `red_bottom` [1,1,6,10] fp32 240B, `blue_top` [1,1,5,10] fp32 200B, several [1,1,10,10] bool masks 100B. Params: `pad_out` [8] int64 64B, `rows` [1,1,10,1] fp32 40B. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** interval-fill-band. Each column is filled to a data-dependent height with blue then red — a per-column vertical interval fill. The ReduceL1 column heights → Less thresholds → banded masks is the interval-fill signature. Output height per column is data-dependent so the 3-channel compact pad is already cheaper than a full-canvas label (log-confirmed).

**Lever:** `red_bottom`/`blue_top` are fp32 input slices (immovable dtype); the [1,1,10,10] band masks are bool already. fp16 recast of the fp32 slices is blocked (fp32-input invariant). No strong lever visible.
