# task007 — ARC-AGI 05269061

**Rule:** 7x7 grid, 3 stripe colors. The TRUE color of cell (r,c) is `colors[(r+c)%3]`
(periodic along anti-diagonals, period 3). OUTPUT fills the whole grid with this rule.
INPUT shows that coloring only on a few complete anti-diagonals (`diags`), rest black;
`diags` picks exactly one diagonal per residue class mod 3, so every residue m in {0,1,2}
has ≥1 shown cell, all of color colors[m]. ⇒ recover color[m] = unique nonzero color on
cells with (r+c)%3==m, then closed-form periodic fill output[r][c]=color[(r+c)%3].
**Current:** 16.43 pts, closed-form separable-residue recover + Gather fill, mem 5079, params 173
**Target tier:** A (closed-form) — Tier-S copy is blocked because the SHOWN source diagonal
is data-dependent (varies per instance), so a constant Gather can't locate the color source;
the 3 colors must be reduced from the variable input → recover-then-rebuild.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | flatten 7x7 → MatMul Rmat counts → colidx → Rmat scatter → Equal(fp16 Lpad) | A | 6504 | 192 | 16.19 | — | fp32 planes |
| 2 | fp16 working planes | A | 6078 | 192 | 16.26 | — | dup grid planes |
| 3 | uint8 Lpad carrier (Equal on uint8) | A | 5227 | 192 | 16.40 | — | +0.01 short |
| 4 | separable col-then-row residue read (no [1,10,49] flatten) + Gather fill | A | 5079 | 173 | 16.43 | 200/200 | ADOPTED |

## Best achieved
16.43 @ mem 5079 params 173 — beats prior 16.11 by **+0.32** (Y). Fresh 200/200 (isolated).

## Irreducible-floor analysis
Two planes dominate: (1) the 7x7 input read `grid` [1,10,7,7] fp32 = 1960B — Slice
preserves fp32 and ALL 10 channels + the full 7x7 are needed (the colored diagonal may be
anywhere, residue colors can appear only on a far diagonal). (2) the output carrier `Lpad`
[1,1,30,30] uint8 = 900B — the color-index plane must be 30x30 to broadcast via Equal into
the FREE [1,10,30,30] output; uint8 (Equal+Pad both accept it) halves it vs fp16. Everything
between is ≤210-elem reductions. ~5079 mem is near the structural floor for this read+fill.

## OPEN ANGLES (re-attack backlog)
- The `grid` 1960B read could in principle drop if a Conv/MatMul could contract BOTH spatial
  axes in one op without materializing the slice — but two-axis contraction needs an
  intermediate carrying the un-contracted 30-axis (3600B) unless sliced first, so slice wins.
  Net: read floor ~1960B looks structural.
- Could fold the colvec→Gather into the residue read if a single matrix produced L directly,
  but that re-introduces a 49-wide scatter — current Gather (49 int params) is already minimal.

## INSIGHT (transferable)
⭐ Periodic anti-diagonal fill `out[r][c]=color[(r+c)%3]` is closed-form tier-A, NOT a fill/
detection wall: (a) `(r+c)%k` is SEPARABLE as `(r%k + c%k)%k`, so per-residue channel counts
come from TWO small MatMuls (col-residue [W,k] then a coupling [W·k,k]) — never materialize
the [1,10,W·W] flatten; (b) recover the per-residue color index as a length-k vector, then
`L = Gather(colvec, idxmap[r,c]=(r+c)%k)` fills the plane with ZERO scatter-MatMul; (c) carry
the final color-index plane in **uint8** (ORT Equal AND Pad both accept uint8) so the 30x30
output-carrier is 900B, half of fp16 — pad sentinel 99 keeps off-grid all-False in every channel.

## 2026-07-01 sequential deep pass

Current source has advanced beyond the older 5079B carrier solution:

- **memory 0, params 127, points 20.15581291354141**
- single `Einsum`: `ncyx,c,jef,ey,fx,jab,ar,bd->ncrd`
- initializers: `channel_mask [10]`, `rule [3,3,3]`, `period [3,30]`
- fresh recheck: **1000/1000 pass**

Rechecked smaller alternatives:

- Use a `[3,7]` period and pad a 7x7 result: fewer period params, but introduces
  a counted 10-channel 7x7 intermediate plus pad metadata, worse than 127.
- Remove `channel_mask` by slicing non-black channels: creates a counted 9x7x7
  float read, much worse than 10 params.
- Replace the `rule [3,3,3]` cyclic coupling with gathered coordinate tables:
  this trades 27 params for coordinate/table tensors or intermediates and does
  not beat the mem0 one-op graph.

Conclusion: no adoptable improvement.  The current graph is already in the
20-point high-score family: direct symbolic `Einsum`, no counted memory.
