# task369 — e8593010

**Rule:** A `size`x`size` grid (default 10, fixed at 10 in train/test) on a gray (5)
background holds non-overlapping, mutually-separated black blobs of size 1, 2 or 3
(single pixel / 2-cell domino / 3-cell line or L). The generator's overlap check
forbids two distinct blobs from touching (4-adjacency), so every 4-connected black
component IS exactly one blob. The output recolours each black cell by
`colour = 4 - component_size`: size1→3, size2→2, size3→1; gray stays 5; off-grid unset.

**Current (prior):** 16.10 pts, ext:galaxy_v1, mem 7200, params 129
**Target tier:** B (closed-form local detection — no flood-fill needed since blobs ≤3 cells & separated).

## Approach
Active grid is fixed 10x10 top-left → Slice channel-0 (black mask) to [1,1,10,10].
Closed-form size classification via two plus-shaped convs (all fp16, 200B planes):
- `deg` = #black 4-neighbours (plus-conv on black).
- `heavy` = black AND deg≥2  (the `black AND` is LOAD-BEARING: a gray cell can have
  deg≥2 by bordering two nearby blobs and would leak through neighbor_heavy onto an
  adjacent domino, mislabelling it size-3).
- `is3` = black AND (heavy OR a 4-neighbour is heavy) = black AND (heavy + conv(heavy) ≥ 1).
- `single` = black AND deg==0.
- Derived closed-form: **label = 5 − 3·black + single − is3** (gray→5, size1→3, size2→2, size3→1).
Cast label to uint8, Pad to 30x30 with off-grid sentinel 99 (uint8 Pad works), then
`Equal(label30, arange10 uint8)` routes the full 10-ch one-hot into the FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 10x10 planes, int32 label30 | B | 11600 | 43 | 15.64 | — | correct but worse than prior |
| 2 | fp16 planes + uint8 label30 (900B) | B | 5300 | 42 | — | 16.42 | beats +0.32 |
| 3 | algebraic label = 5−3·black+single−is3 | B | 4500 | 41 | 200/200 | 16.58 | ADOPTED, beats +0.48 |

## Best achieved
16.58 @ mem 4500 params 41 — beats prior 16.10 by +0.48. Isolated fresh 200/200.

## Irreducible-floor analysis
Dominant intermediates: `label30` 900B (30x30 uint8 — the output canvas is fixed 30x30,
uint8 is the dtype floor for a Pad-routed label plane) and `black32` 400B (the fp32
channel-0 Slice — entry plane, Slice preserves fp32). The remaining ~10 fp16 100-elem
planes (200B each) carry the deg/heavy/is3 logic. Not at a hard floor, but further cuts
(~400-600B) move pts only ~+0.08.

## OPEN ANGLES
- Fold the `heavy`/`is3` two-conv chain into a single shape-detecting conv (detect 3-runs
  directly) to drop 2-3 fp16 planes — would need to handle both line and L shapes in one kernel.
- Route the one-hot from the 10x10 label without a 30x30 carrier (avoid the 900B label30) —
  blocked because Pad rejects bool and Equal-on-10x10→[1,10,10,10] is 1000B (worse).

## INSIGHT (transferable)
⭐ "Colour each blob by its connected-component SIZE" is NOT a flood-fill wall when blob
sizes are GENERATOR-BOUNDED small (≤3) and blobs are mutually separated: size becomes a
LOCAL predicate from neighbour-degree (deg=plus-conv) plus a one-hop "heavy neighbour"
spread (second plus-conv on a BLACK-GATED deg≥2 mask). The black-gate on `heavy` before
the spread conv is essential — a gray cell with deg≥2 otherwise leaks size-3 onto adjacent
dominoes. Collapse the whole is1/is2/is3 select into ONE algebraic label
`5 − 3·black + single − is3`, run fp16, and use a uint8 (not int32) Pad-routed label plane.
