# task168 — 6e19193c

**Rule:** 2-3 single-colour "arrows" on a fixed 10x10 grid. Each arrow = a 2x2 block of
`color` with ONE corner left black (the tip); the missing corner picks a diagonal
direction d=(dr,dc) pointing OUTWARD. Output keeps the 3 coloured corners (missing
corner stays empty) and draws a diagonal ray of `color` from ONE step outside the
missing corner — cells (i+t·dr, j+t·dc), t=1,2,… — to the grid edge. Blocks are >=4
apart so 2x2 windows are clean. (Blank-note "confirmed-infeasible", NO reason — false.)
**Current:** 15.08 pts
**Target tier:** B — per-pixel diagonal-ray reconstruction, single colour routed into the FREE output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-input Mul colour + 19x19 ray convs | B | 45306 | 1518 | 14.25 | 20/20 | over-mem (36000B colour plane + big kernels) |
| 2 | per-channel-count colour + slice-9 occ + 10x10 ray convs | B | 9392 | 477 | 15.80 | 20/20 | beats P |
| 3 | occ from ch0 slice only (400B) | B | 5792 | 478 | 16.26 | 20/20 | better |
| 4 | batched grouped ray conv, centered 19x19 | B | 6193 | 1523 | 16.05 | 20/20 | worse (kernel params dominate) |
| 5 | per-dir 10x10 asym-pad kernels + Where(uint8) L | B | 5393 | 479 | 16.32 | 500/500 | BEST |

## Best achieved
16.32 @ mem 5393 params 479 — adopted? N (build-only). Beats prior 15.08? Y, by +1.24.

## Irreducible-floor analysis
Dominant = the 12 fp16 [1,1,10,10] planes (200B each) across 4 directions: detection
Conv resp + tip cast + ray Conv resp. The L-match detection (==3) is a genuine
nonlinearity that MUST precede the diagonal ray propagation, so detection and ray
cannot fuse into one conv. Conv I/O is float so fp16 (200B) is the per-plane floor;
the ray reach is corner-to-corner (10x10 kernel needed). Colour is a 40B scalar and
occupancy is a single 400B ch0 slice, so the geometry side is already minimal.

## OPEN ANGLES (re-attack backlog)
- Share each ray kernel across its opposite direction (TR↔BL anti, TL↔BR main) via a
  flip (negative-step Slice) like task037 — saves ~200 ray params but adds flip
  planes; net ~+0.04, likely a wash.
- Collapse opposite-direction ray pairs onto the batch axis with one main + one anti
  kernel reused by pad-side swap (task037 idiom) to cut conv-plane count.

## INSIGHT (transferable)
⭐ "Draw a diagonal ray from a detected feature" is closed-form tier-B, NOT a flood/
connectivity wall: (1) detect the seed with a small Conv whose kernel is +1 on the
required ON cells and a large negative on the forbidden cell, thresholded with Equal
(banded L-shape match); (2) propagate as a BOUNDED directional prefix-OR = ONE diagonal
Conv (10x10 kernel + asymmetric SAME pad, offsets t=1..9) thresholded >0 — the task037
diagonal-conv idiom. Detection's nonlinearity must come before propagation, so they
stay two convs. For a single-colour output, recover colour as a 40B scalar from
per-channel pixel counts (sum_k k·(cnt_k>0)) and route via Where(occ, c_u8, 0) +
Pad(255) + Equal(arange) into the FREE bool output — no full-resolution colour plane.
Centering a grouped multi-direction ray conv (one shared pad) forces a 19x19 kernel
whose params outweigh the saved planes — per-direction 10x10 asym-pad kernels win.
