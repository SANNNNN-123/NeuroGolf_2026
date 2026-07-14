# task323 — d06dbe63

**Rule:** 13x13 grid with one cyan(8) pixel at (row,col). Output keeps the cyan pixel and
draws a gray(5) staircase in two point-symmetric directions ((-1,+1) and (+1,-1)); each
branch alternates 2 vertical then 2 horizontal unit steps (a 45-degree staircase), clipped
at the grid edge. Relative to the marker the gray set is a FIXED 48-offset shape.
**Current:** 16.58 pts, conv-stamp (Concat 10-ch bool@13x13 -> Pad), mem 3887, params 645
**Target tier:** A (fixed-shape stamp at a marker = separable-free closed form; one conv).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 25x25 stamp Conv (fp32) + Pad fp32 30x30 + Where | A | 5852 | 648 | 16.22 | - | fp32 30x30 plane too heavy |
| 2 | same, fp16 pad 30x30 | A | 4390 | 648 | 16.48 | - | fp16 pad still 1800B |
| 3 | fp16 conv (cast marker->fp16) + fp16 pad + Where | A | 4052 | 648 | 16.54 | - | dropped fp32 resp plane |
| 4 | two 13x13 convs (params 361) + Add | A | 5742 | 361 | 16.28 | - | extra conv planes cost > param saving |
| 5 | direct 30x30 conv (asym pad) | A | 3714 | 648 | fail | - | mis-stamps off-grid margin (no clip) |
| 6 | **fp16 conv@13x13 -> Greater bool@13x13 -> opset-13 bool Pad -> Where** | A | 2421 | 656 | **16.97** | 200/200 | ADOPTED |

## Best achieved
16.968 @ mem 2421 params 656 — adopted? Y. Beats prior 16.58? Y (+0.39 >= +0.3).

## Irreducible-floor analysis
Two structural costs: the 625-elem 25x25 stamp kernel (a corner marker's staircase reaches
+-12, so SAME conv needs a 25x25 kernel — element count is the param floor, dtype is free)
and the single 900B bool [1,1,30,30] cond plane the Where requires. Everything upstream is
13x13 (m8_f 676 fp32, m8_h/stamp_h 338 fp16, condK 169 bool). The prior 16.58 net paid a
1690B 10-channel bool Concat to assemble the output at 13x13 then Pad; replacing that with
Where(bool-cond, gray_onehot, input) drops the 10-ch assembly to a single 900B cond.

## OPEN ANGLES (re-attack backlog)
- Kernel 625 is the largest single item. A point-symmetric kernel (K[i,j]==K[24-i,24-j])
  could store ~313 elems, but reconstructing the full 25x25 at runtime (flip+Concat) adds a
  ~625-elem intermediate that costs more mem than the param saving — net loss.
- Closed-form predicate (s=dr+dc, dr-parity, quadrant gates) eliminates the kernel but needs
  several 2D fp16 13x13 planes; predicate verified exact but plane count > kernel cost.

## INSIGHT (transferable)
⭐ A fixed-shape marker-stamp (single 1 in a channel) is one SAME Conv whose kernel encodes
the shape's offsets; clip to the active KxK grid with the Conv output size (do NOT widen pads
to reach 30x30 — that re-stamps the off-grid margin). The final-plane win: threshold to bool
at KxK (cheap) then route 10-ch via `Where(bool_cond, onehot, input)` so the ONLY full-canvas
plane is a 900B bool cond — never a fp16 30x30 (1800B) nor a 10-ch Concat (1690B). ⭐ Use
**opset 13** (not the builders' default opset 10) when you need Pad to accept bool / pads as
an input tensor; the scorer checks DOMAIN not VERSION, so an opset-13 graph scores fine.
