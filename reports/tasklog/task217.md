# task217 — 8f2ea7aa

## 2026-06-29 mechanism screen

Rule: use a 3x3 Conway-like motif as a sprite and render pairwise motif-position
convolutions in a 9x9 output area.

Current source score: 17.706982 @ mem 1429 params 41.  The graph is already close
to a semantic floor: reduce background/foreground colour, crop the motif, run one
dilated Conv for the pairwise placement mask, and route with `Where(mask, fg, bg)`.

No rewrite adopted.  The dominant cost is the small 9x9/10x10 crop and output
masking, not a removable full-canvas intermediate.
