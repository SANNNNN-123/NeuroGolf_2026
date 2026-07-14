# task280 — b527c5c6

**Rule:** Input has TWO solid green rectangles, each with one red dot on an edge. One box
is tall (H>=W) and emits a HORIZONTAL beam; the other is wide (H<W) and emits a VERTICAL
beam. A beam is an axis-aligned RECTANGLE: a green band (half-width = perpendicular box
extent through the dot minus 1, centred on the dot) running from the dot to the grid wall
on the side the box does NOT occupy (dot on left/top edge -> beam left/up, else right/down),
with the centre line red. Output = input + the two beams (red centre over green band; green
fills only previously-empty cells; clipped to the n x n in-grid region). No flip/xpose/defect
logic needed at inference — the dot's edge position alone fixes axis & direction.
**Current (prior):** 13.89 pts, gen:vyank6322, mem 65626, params 905 (heavy MaxPool/Where fill)
**Target tier:** A — beams are separable row(x)col rectangles routed into the free output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-dot separable rects, fp32 masks, 10-ch Equal | A | 180758 | 209 | 12.89 | ok | too heavy |
| 2 | bool masks + Concat output | A | 83786 | 1099 | 13.65 | ok | still heavy |
| 3 | slice->bool first, uint8 dots, gate-fold | A | 64946 | 1099 | 13.90 | ok | beats baseline |
| 4 | reduce 2-dot axis BEFORE outer-product | A | 54026 | 1099 | 14.08 | ok | |
| 5 | drop in-grid `<n` bound (ingrid gate suffices) | A | 50234 | 1096 | 14.15 | ok | |
| 6 | uint8 dot-finding flat (ArgMax/Scatter @900B) | A | 45734 | 1051 | 14.247 | 200/200 | ADOPT-CANDIDATE |

## Best achieved
14.247 @ mem 45734 params 1051 — beats prior 13.89 by +0.357 (>= +0.3). fresh 200/200
(isolated genverify on disk) + 800/800 numpy-spec + 300/300 ORT.

## Irreducible-floor analysis
Dominant remaining cost = three single-channel fp32 input Slices (R, G, bg) at 3600 B each
(10800 B). A single fp32 channel read is the hard 3600 B floor (Slice/Gather/ReduceMax all
cost the same). All per-dot beam geometry is collapsed to SMALL [2,30,1]/[2,1,30] vectors
that are OR-reduced over the 2-dot axis BEFORE any outer product, so no [2,30,30] or even a
[30,30] product exceeds 900 B. Dot finding is uint8 (900 B flat + uint8 ArgMax/Scatter).

## OPEN ANGLES (further squeeze, not required)
- Replace the 3 fp32 slices (10800 B) with ONE `ReduceMax(input,axis=1)` ingrid plane +
  a 2-channel red/green slice — same byte total, no clear win.
- The 900-param `falsech` (7x reused in the output Concat) could move to a runtime all-false
  plane (mem vs params wash). Not worth it; already +0.35.

## INSIGHT (transferable)
⭐ A "beam / ray / shoot-to-wall" task that LOOKS like a multi-directional variable-span fill
wall (prior net: 15 MaxPool + 32 Where at the ln-floor) can be FULLY CLOSED-FORM when the
beam is an axis-aligned RECTANGLE: each beam = separable row-mask (x) col-mask broadcast into
the free output. Key levers that took it from 180 KB -> 45.7 KB: (1) read scalar geometry
(dot coords via double-ArgMax-with-uint8-Scatter-mask; contiguous run length = distance to
nearest non-fg via ReduceMax/ReduceMin of a masked position ramp — NO loop/CumSum); (2) when
N candidate objects each contribute to a DIFFERENT axis (here: exactly one horizontal + one
vertical beam), GATE the small per-object broadcast vectors by the axis flag and OR-REDUCE
over the object axis BEFORE the outer product — kills every [K,30,30] plane; (3) uint8 ArgMax
+ uint8 ScatterElements both run under ORT_DISABLE_ALL (900 B vs fp32 3600 B for the flat);
(4) drop in-grid `<n` clamps when a final `AND ingrid` already suppresses off-grid fill.

## S8 (2026-07-02) — rect-recipe conversion ADOPTED, div 0
moment einsums (Σr,Σc,Σr²,Σc²,Σrc) + closed-form quadratic (Sqrt IEEE-exact on perfect squares) locate the 2 red cells; red_f + double-TopK dropped; 6039→4146, +0.376. Fresh: agent uncached 2500 div0 + my uncached 400 div0.
