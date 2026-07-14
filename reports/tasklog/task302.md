# task302 — c0f76784

**Rule:** 3 (sometimes 2) non-overlapping square gray(=5) box outlines of side L∈{3,4,5}
on a fixed 12×12 grid, each hollow (its (L-2)×(L-2) interior is background black=0). Output
keeps every gray frame and FILLS each hole solid with colour 5+(L-2)=3+L (L=3→6, L=4→7,
L=5→8). Every interior cell's fill = 5 + s where s = hole side (=L-2).
**Current:** 16.04 pts (prior public net)
**Target tier:** A (separable closed-form: per-cell colour-index plane routed into FREE bool output)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | run-length(hrun) + 4-dir gray-bound, ingrid AND, fp16 30×30 | A | 25164 | 125 | 14.86 | 266/266 | works, too big |
| 2 | drop ingrid (sentinel-99 Pad handles off-canvas) | A | 12888 | 117 | 15.53 | — | win |
| 3 | windowed-gray enclosure (no run-bound) + presence-count value | A | 11556 | 108 | 15.64 | — | win |
| 4 | LINEAR weighted-conv value (2·G@1+G@2 both sides) | A | 8388 | 87 | 15.96 | — | big win |
| 5 | pack gl/gr/gu/gd into ONE [4,1,7,7] conv + ReduceProd; uint8 L | A | 6084 | 240 | 16.25 | — | win |
| 6 | L = Where(interior, fill, 5·G) (fold fillterm/intf/add) | A | 5508 | 240 | 16.34 | 200/200 | win |
| 7 | interior = (G==0)&encG (drop black-channel slice) | A | 4932 | 234 | 16.45 | 500/500 | **adopted** |

## Best achieved
16.45 @ mem 4932 params 234 — adopted? Y. Beats prior 16.04? Y (+0.41).

## Irreducible-floor analysis
Dominant intermediates: the [1,4,12,12] fp16 enclosure conv stack (1152B, four within-3
directional gray sums; needs all 4 → can't collapse to a sum because adjacent boxes put 2
grays in one within-3 arm, so a sum==4 test misfires — ReduceProd>0 is required) and the
[1,1,30,30] uint8 padded colour-index L (900B, the one unavoidable index plane; uint8 via
Pad halves the fp16 cost). Everything else is 12×12 (=144 cells) so fp16 planes are only
288B — the small-active-canvas escape (generator size bound 12) is what makes this Tier-A.

## OPEN ANGLES (re-attack backlog)
- Could fold valsum into the enclosure conv as a 5th channel (−288B mem, +49 params) — net
  neutral, skipped.
- The [4,1,7,7] enclosure kernel is 196 of the 234 params; a reshape-to-batch trick to share
  one 1-D kernel across the 4 arms would cut params but adds planes — not worth at 16.45.

## INSIGHT (transferable)
⭐ "Read a hole's SIZE from the distance to its flanking frames" beats run-length: when a
generator guarantees objects are ≥1 cell apart, within a small window along one axis there is
AT MOST ONE marker, so the nearest-marker distance becomes a single LINEAR weighted Conv
(weights 2,1,… by proximity) — NO argmin, NO product run-length chain. Pair it with a
ReduceProd-over-channels enclosure test (pack the 4 directional within-k sums into one conv,
product>0 ⇔ marker present in every direction; a sum-threshold FAILS under adjacency). And
`interior = (plane==0) AND enclosed` removes a whole second channel slice when the in-grid
palette is just {bg, marker}.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1216/17.52; ch5 576B fp32 gray slice=min entry, features concat + ch5_u8 feed final QLinearConv (both needed). No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
