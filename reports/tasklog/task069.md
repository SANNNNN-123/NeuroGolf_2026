# task069 — 321b1fc6

**Rule:** Grid is always 10x10. `num_boxes` (4) IDENTICAL 4-connected sprites (an arbitrary
connected subset of pixels inside a small box, width 2..4 height 2..3) are placed at random
non-overlapping positions (bounding boxes separated with margin 1). Exactly ONE sprite (first
drawn) is shown in its real per-pixel COLOURS; every other sprite is shown all-cyan (8). OUTPUT =
erase the coloured sprite to black, and redraw EVERY cyan sprite in the real colour pattern,
aligned to each sprite's own bounding-box top-left. Verified (500 fresh): output nonzero <=> input
== cyan(8); coloured box -> black; colour at a cyan cell = colourmap[(r-bbox_top, c-bbox_left)]
where the colourmap (offset -> colour) is revealed by the single coloured sprite, same for all.
**Current (stored):** 13.84 pts.
**Target tier:** B (label-map). NOT S (non-local: cell colour depends on its offset within its own
sprite + the revealed pattern). NOT A-separable (P is an arbitrary 2..4-colour pattern; the colour
table is irreducibly 2-D over offset). Per-cell offset IS recoverable locally -> label-map + Equal.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 8-conn min-prop (3x3 minpool) for bbox-top | B | 40896 | 84 | 14.38 | 200/200 | WRONG: merges corner-adjacent sprites (key 12+ OOB). Margin-1 bboxes can touch at a corner. |
| 2 | 4-conn PLUS-min (vert 3x1 + horiz 1x3 minpool, elementwise Min), 7 iters | B | 30098 | 104 | 14.68 | 200/200 | correct; min-prop in negated space (MaxPool, no per-step Neg) |
| 3 | histogram via dr/dc double-MatMul ([3,N]@[N,4] fp16) instead of [N,12] one-hot | B | 25950 | 99 | 14.83 | 200/200 | dropped 4800B keyoh plane |
| 4 | col_w = colf*(colf!=8) directly (drop occf/is_col) | B | **25150** | **99** | **14.86** | **500/500** | FINAL |

## Best achieved
14.86 pts @ mem 25150, params 99 — 264/264 stored, fresh 500/500. Adopted? **N** (orchestrator
gates). Beats stored 13.84 by **+1.02 (Y, generalizes)**.

## Irreducible-floor analysis
- **colf30 3600 B fp32 [1,1,30,30]** — the 1x1 colour-index Conv; output must be 30x30 fp32 (any
  linear combo of the FREE fp32 input is fp32). Entry-read floor (shared with task368/358).
- **bbox-anchor plus-min ~11 KB** — 7 iterations x 2 axes x ~4 small fp16 planes (vert minpool,
  horiz minpool, Max, re-mask Where) at 200 B each. Iterations: a 4-connected sprite <=3x4 has a
  longest plus-propagation path of 7; radius MUST be 1 (a wider minpool reaches across the margin-1
  gap and merges neighbour sprites). This is the dominant *non-entry* cost and is fundamental to
  recovering each cell's bbox top-left for ARBITRARY (holed) shapes under 4-connectivity.
- **L30 900 B uint8 [1,1,30,30]** — output label feeding the free final Equal; uint8 is cheapest,
  sentinel-99 pad off-grid. Irreducible for any per-cell colour rewrite.
- histogram ~2.7 KB (dr/dc one-hots [3,N]/[N,4] fp16+bool); N=100 since the coloured sprite can
  sit anywhere; table genuinely 3x4 (arbitrary pattern).

## OPEN ANGLES (re-attack backlog)
- The 7-iter plus-min is the binding non-entry cost. A non-iterative bbox-anchor would need either
  (a) a connectivity-respecting closed form for arbitrary holed shapes (none known cheaper than
  propagation), or (b) anchor-correlation: recover the coloured sprite as a small KxK stamp + find
  each cyan box's top-left by correlation, then stamp. Risky (bbox top-left corner can be an EMPTY
  cell for L-shapes, so the anchor isn't a pixel) — untried, maybe ~0.2 pt.
- colf30 3600 fp32 is the entry floor; no free-input fp16 colour-index path (shared wall).

## INSIGHT (transferable)
⭐ "Recolour every marker-coloured sprite from the one revealed sprite" generalises beyond SOLID
rectangles (task368) to ARBITRARY 4-connected shapes: the per-cell offset is `(r,c) - bbox_top_left`
of the cell's own sprite, recovered by propagating the MIN row/col index over the sprite via an
iterated PLUS-shaped min (NOT 3x3 — generator margin-1 bboxes touch at CORNERS, so 8-connectivity
merges distinct sprites; use 4-connectivity = elementwise Min of a vertical 3x1 and horizontal 1x3
min-pool). Do min-prop in NEGATED space so each step is a plain MaxPool with no per-step Neg.
Build the offset->colour table as a dr/dc double-MatMul ([3,N]fp16 @ [N,4]fp16, colour-weighted)
instead of a [N,12] key one-hot (saves ~4.8 KB). The whole thing lands as a label-map + final Equal.
⭐ Plus-min radius MUST be 1: a wider min-pool reaches across the inter-sprite gap and merges them.
