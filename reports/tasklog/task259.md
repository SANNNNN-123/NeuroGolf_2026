# task259 — a740d043

**Rule:** Input is a `width x height` (5..7 each) grid filled with BLUE (colour 1). A small
`hollow_conway` sprite (wide,tall in 2..3) is drawn on it at (rowoffset,coloffset); sprite
pixels take random non-blue colours (>=2). Output = the sprite cropped to its bounding box
(size wide x tall, <=3x3): every sprite pixel keeps its colour; every other cell inside the
bbox (blue background / hollow gaps) becomes BLACK (colour 0). Colours are random per
instance so the per-cell value must be carried. NOTE bg=BLUE(1), not 0; the bbox is the SPAN
(max-min+1) of occupied lines, NOT the count (sprite has hollow gaps).
**Current:** 16.169 pts, public crop+pad net (mem 6793, params 50)
**Target tier:** B (data-dependent crop window + arbitrary per-cell colour carry). NOT S/A:
the output colour per cell is an arbitrary per-instance value read from a data-dependent crop
window, not a fixed Conv/permute of the local one-hot and not a row⊗col separable rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 7x7-slice→Conv colf, bbox via occupied-COUNT, gather window, label+Pad+Equal | B | 3756 | 64 | 0.0 | — | BUG: bbox must be span not count (hollow gaps) |
| 2 | bbox = max-min+1 span (4 ReduceMin/Max) | B | 3772 | 66 | 16.747 | 200/200 | win |
| 3 | fold blue/gap→0 onto the tiny 3x3 window (drop 7x7 vmap+spritem planes) | B | **3572** | 66 | **16.801** | 200/200 | KEEP |

## Best achieved
16.801 @ mem 3572 params 66 — adopted? N (orchestrator gates). Beats prior 16.169 by **+0.63**.
GENERALIZES: isolated fresh 200/200.

## Irreducible-floor analysis
Dominant intermediates:
- **1960 B fp32 `xin` = Slice(input,[0:7,0:7])** — the 10-ch active region. The colour-index
  Conv needs the 10 channels spatially; slicing to the 7x7 active grid FIRST (escape (3),
  generator bounds width/height<=7) makes the Conv 196 B instead of 3600 B. Slicing fewer
  channels is impossible (sprite colours are random 2..9); Conv-on-full-input then slice colf
  = 3600 B (worse); casting input to fp16 = 18000 B. So 1960 B is the structural entry cost.
- **900 B uint8 `L` Pad [1,1,30,30]** — the final Equal must span the 30x30 output; Pad rejects
  bool so Equal-then-Pad is impossible; sentinel-10 Pad-then-Equal is the cheapest 30x30 route.
- Everything else <=200 B: colf [1,1,7,7] 196 B, the WORK(3) x 7 / 3x3 gather windows, and
  1-D occupancy/bbox scalars.

## OPEN ANGLES (re-attack backlog)
- Cut the 1960 B `xin`: needs an op that produces a 7x7 colour index from the 10-ch input
  without materialising the 10-ch 7x7 slice — no known cheaper primitive (Conv/Mul both pay it).
- Cut the 900 B Pad: only if a bool carrier could be padded; ORT Pad rejects bool — blocked.

## INSIGHT (transferable)
⭐ A "crop the sprite bbox + carry arbitrary colours" task on a SMALL generator-bounded canvas
(width/height<=7) beats the 3600 B colour-plane floor by SLICING the 10-ch input to the active
NxN FIRST (escape (3)) — the colour-index Conv then costs (active² fp32) not 3600 B. Two more
levers stack: (a) when bg is a NON-ZERO colour (blue=1), occupancy = `colf >= 2` not `>0`;
(b) fold the bg→black remap onto the tiny gathered KxK window (Greater+Where on 3x3), never on
the full active plane. Watch: with hollow/gappy sprites the bbox is `max-min+1` SPAN, never the
occupied-line COUNT (counting silently drops the gap columns/rows).
