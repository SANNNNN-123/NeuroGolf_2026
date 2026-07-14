# task189 — 7c008303 (cornercolor2)

**Rule:** Input is a 9x9 grid (size always 6). A 2x2 color "legend" sits in one
corner (cells (0,0),(0,1),(1,0),(1,1) = colors[0..3], sampled from {1,2,4,5,6,7,9},
i.e. 1..9 minus green/cyan). A cyan(8) cross (full row + full col at index 2)
separates the legend from a 6x6 green(3)-pixel region (rows/cols 3..8). Output is
6x6: each green pixel at (r,c) is recolored by its QUADRANT — legend cell
(r>=3, c>=3) → colors[2*(r>=3)+(c>=3)]. Then flip_horiz and/or flip_vert are
applied to BOTH grid and output identically. So in the *flipped frame*:
vflip ⟺ cyan at (6,0); hflip ⟺ cyan at (0,6); legend corner rows/cols = {0,1} or
{7,8}; green block = rows/cols {3..8} or {0..5}; out[R][C] = legend[R//3][C//3]
if green at the matching block cell else 0.
**Current:** 16.15 pts (label-map build, mem 6054 / params 952).
**Target tier:** B (per-cell label map). Tier S/A blocked: output colors are
arbitrary per-instance legend colors (a fixed Conv/separable mask can't route
them), so the value plane must be data-dependent.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior: 9x9 10-ch slice + cyan-row/col Convs | B | 6054 | 952 | 16.15 | — | baseline |
| 2 | flips read from kin9 (drop 2 cyan Convs) + drop ch0 from slice | B | 5562 | 357 | 16.31 | — | better |
| 3 | + fp16 downstream (kin/banks/MatMuls), Equal(==3) green | B | 5092 | 356 | 16.40 | — | better |
| 4 | split reads: ch3 green9 + 2-cell cyan + 4 corner legend slices | B | 3008 | 315 | 16.89 | — | better |
| 5 | shrink cyan to two single-cell slices (4B each) | B | 2756 | 317 | 16.97 | 500/500 | ADOPT |

## Best achieved
**16.97 @ mem 2756 params 317** — beats prior 16.15 by **+0.82**. Fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: L-pad uint8 [1,1,30,30]=900B (the 30x30 single-channel
label carrier feeding the final Equal → FREE bool output; irreducible — output is
30x30 and uint8 is the cheapest carrier dtype). Then green9 [1,1,9,9] fp32 324B +
fp16 cast 162B (the one channel-3 read; 9x9 is forced because the green block
sources rows/cols 0..8 across both flip orientations). Then the 4 corner legend
slices 4×144=576B (read all four 2x2 channel-1..9 corners because the legend
corner is flip-dependent; selected by index vflip*2+hflip). Reading the legend by
Gather-from-full-input is WORSE (the unselected 30-axis makes the intermediate
[1,10,2,30]=2400B), so the four fixed 144B corner slices win.

## Key levers used
- ⭐ NEVER form the 10-channel 9x9 plane (3240B). Split into single-purpose reads:
  one channel-slice per role (green=ch3, cyan=ch8) + tiny per-corner slices for
  the multi-color legend. Each role touches only the channels/region it needs.
- ⭐ Read flip/orientation bits from TWO single-cell channel slices (4B each) — no
  cyan-row/col sum-Convs (saved 600 params) and no 7x7 region.
- ⭐ One fp32 entry per slice, then Cast to fp16 and run ALL downstream
  (MatMul/Equal/Where) in fp16 — color indices ≤9 are fp16-exact (task377 lever).
- Data-dependent corner pick = stack 4 collapsed corners + Gather by a scalar
  index vflip*2+hflip (no per-cell selection plane).

## OPEN ANGLES (re-attack backlog)
- 4 corner slices (576B): if a contiguous flip-independent legend window existed it
  could be one slice, but the legend is at opposite diagonal corners under flips →
  4 fixed reads is the minimum without a runtime (symbolic-dim) slice.
- L-pad 900B and green9 486B are the two structural floors; both are single-channel
  reads/carriers that the B-tier label-map form requires.

## INSIGHT (transferable)
⭐ When an "all-channels per-cell" task only needs DIFFERENT channels for DIFFERENT
roles (here: legend=any color, mask=green, orientation=cyan), do NOT collapse one
shared 10-channel plane. Split into role-specific single-channel/small-region
slices — total bytes drop far below the 10-ch read because each role touches only
its own channel and minimal extent. Combined with reading orientation bits from
literal single-cell slices (4B) and going fp16 after each fp32 entry, a flip-
equivariant quadrant-recolor goes 16.15 → 16.97 (B-tier, +0.82).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.002)
**Mechanism (op-census diff):** Trimmed `h_idx`/`v_idx` [1,4]→[1,3] (−1 elem each). −2 params.
**Old→new:** mem 866→866, params 49→47.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task189_pre_s10.onnx`; source `public_candidates/bobmyers7186/task189.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.
