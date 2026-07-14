# task088 — 3de23699 (crop box interior, recolour sprite → corner colour)

**Rule:** A rectangular "box" is marked ONLY by 4 corner pixels in colour `colors[1]`
at the four cells just OUTSIDE the interior: (brow-1,bcol-1), (brow-1,bcol+wide),
(brow+tall,bcol-1), (brow+tall,bcol+wide). Inside the interior
[brow..brow+tall-1] x [bcol..bcol+wide-1] are `wide+tall+randint(-1,1)` sprite pixels
in colour `colors[0]`. Output = the tall×wide interior cropped to the top-left of a
fresh grid, with every sprite pixel painted the CORNER colour `colors[1]` and every
other interior cell background (0); outside tall×wide everything is off.
Colour ID (exact, verified 0/5000): corner colour = the non-bg channel with pixel
count == 4; sprite colour = the non-bg channel with MAX count (sprite ≥ 5 since
wide+tall ≥ 6). Box geometry from the corner colour's bbox: brow=rmin+1, bcol=cmin+1,
tall=rspan-1, wide=cspan-1.
**Current:** 13.846 pts, gen:biohack_new, mem 69698, params 120
**Target tier:** B (data-dependent crop + translate-to-origin). Not Tier A: output is a
data-dependent TRANSLATE of a recovered window to the origin (needs a Gather-shift, not
separable). Both the crop window AND the fill colour are data-dependent, but every
parameter collapses to a closed-form scalar (count==4 / argmax / 1-D bbox), so it lands
well below the detection floor. Same shape as task036 (crop+shift).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | counts→cornercol(==4)/spritecol(argmax); corner bbox from 1-D occupancy; Gather(input,spritecol,ch) full plane → 10×10 window shift; label map L→Equal | B | 12791 | 134 | **15.533** | 200/200 (+500/500) | WIN |

## Best achieved
**15.533 @ mem 12791 params 134 — adopted? N (orchestrator gates).**
Beats prior 13.846 by **+1.69** (≥+0.3 ✓). Generalizes: fresh isolated 200/200 AND a
separate 500/500; the colour-ID + bbox rule is 0/5000 exact in numpy.

## Irreducible-floor analysis
Dominant intermediates (same floors as task036, ~15.6):
- **3600 B fp32 `splane` = Gather(input, spritecol, axis=1)** — the full 30×30 sprite
  mask plane. Irreducible: the crop window position (brow,bcol) is data-dependent, so the
  full plane must exist before the windowed Gather (circular: window pos needs the corner
  bbox, plane needs spritecol). fp16 does NOT shrink it (ORT upcasts full planes to fp32).
- **two 1200 B fp32 occupancy profiles** `rowocc`[1,10,30,1] + `colocc`[1,10,1,30] from
  ReduceMax(input) — needed to recover the CORNER colour's bbox cheaply (avoids a SECOND
  full 3600 B corner plane). They stay fp32 (inherit input dtype).
- **900 B uint8 padded label map** L[1,1,30,30] — Pad rejects bool so the 30×30 label map
  must be uint8 before the final Equal→BOOL output.
Everything else (counts 40 B, scalars, fp16 ramps) is small.

## OPEN ANGLES (re-attack backlog)
- Avoid materialising splane: gather the 10-col / 10-row window straight from the 10-ch
  input then contract — tried mentally, [1,10,10,30]=12 KB is WORSE; collapse-to-1-channel
  first wins. No cheaper path while window pos is data-dependent (~3600 B is the floor).
- Fuse the two Gather(axis2)+Gather(axis3) window steps via one GatherND (≈ saves the
  300-elem `Vr` intermediate, ~+0.05, marginal).
- The two occupancy profiles could in principle be one ReduceMax if corner bbox came from a
  single combined scan, but row & col mins/maxes need separate axes — no clean fusion.

## INSIGHT (transferable)
⭐ "4 corner markers define a variable crop box" is the task036 crop+shift idiom with a
cleaner colour-ID: count-based discrimination (corner = count==4, sprite = argmax-count)
beats geometric span analysis when the marker colour has a FIXED small pixel count. Recover
the crop window from the marker colour's 1-D occupancy bbox (no second full plane), then
Gather-shift the OTHER colour's plane to origin and recolour via a label map. Two distinct
data-dependent scalars (window colour vs fill colour) cost only 2 cheap ArgMax/Equal over
the 40 B counts vector — no extra planes.
