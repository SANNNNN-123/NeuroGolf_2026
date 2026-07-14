# task200 — 8403a5d5

**Rule:** Input is a 10×10 grid (top-left of the 30×30 canvas) with a SINGLE coloured
pixel (colour 1..9, never gray) at `grid[9][col]`. The output draws a boustrophedon
"snake" starting there: fill the seed column, turn at the top with a GRAY(5) connector
cell, fill the next column downward, turn at the bottom with a gray connector, and so on,
moving right until off-grid. Closed form (offset k=c−col, in-grid r<10 & c<10): COLOUR
columns are `c>=col AND k even` (full r=0..9 = colour); GRAY connectors (k odd, disjoint
columns) are `k%4==1 -> gray at row 0` / `k%4==3 -> gray at row 9`. The padding region of
the output tensor is ALL-ZERO (not bg-channel-1).

**Current (deployed):** 16.912 pts, ext:galaxy_v1, mem 3192, params 63
**Stated P in build prompt:** 16.79  (bar to beat = 17.09, i.e. mem+params < 2723)
**Target tier:** A — separable row⊗col value plane; no copy of input cells (not S).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | sep. bool terms OR'd into output (ch0 missing) | A | — | — | — | 0/200 | wrong: ch0 unset |
| 2 | MatMul rowmat@colmat -> L f16 -> Equal(arange) | A | 8386 | 148 | 15.95 | 0/200 | wrong: lit padding ch0 |
| 3 | + out-of-grid sentinel (−100) in L | A | 8386 | 148 | 15.95 | 200/200 | correct |
| 4 | Conv k-weight col-recovery (no 10-ch interm.) | A | 5986 | 438 | 16.23 | 200/200 | correct |
| 5 | build matrices f16-native (drop f32 copies) | A | 4068 | 438 | 16.59 | 200/200 | correct |
| 6 | fold col-sentinel into v_color, f16 k-pipeline | A | 3770 | 408 | 16.66 | 200/200 | correct |
| 7 | rowmat via bool-Concat + single Cast | A | 3650 | 408 | 16.69 | 200/200 (+80/80 exhaustive) | BEST |

## Best achieved
16.69 @ mem 3650 params 408 — adopted? **N** (regression: below deployed 16.912 AND below
stated P 16.79). Beats P? **N**.

## Irreducible-floor analysis
Two hard floors stack:
1. **Value plane L = 1800B (fp16 [1,1,30,30]).** The 10-channel one-hot is routed into the
   FREE bool output via `Equal(L[1,1,30,30], arange[1,10,1,1])`; the broadcast REQUIRES L to
   be a full 30×30 plane. fp16 is the floor (uint8 would be 900B but MatMul/arithmetic can't
   produce uint8, and Cast-to-uint8 keeps the fp16 plane too → 2700B, worse). A single Where
   only handles 2 row-regimes; this rule has 4 (interior / row0 / row9 / out-of-grid-row), so
   nested Wheres cost ≥2 full planes. MatMul collapses all 4 regimes into ONE L — provably the
   leanest full-plane route here.
2. **Col-recovery Conv = 300 params.** Every cell is one-hot so a plain occupancy sum is 10 in
   every in-grid column (no signal); the seed column can only be isolated by excluding bg
   channel 0, which needs a channel-weighted reduction. A `Conv(input, W[1,10,30,1])` with
   W=k does channel-exclude + row-collapse in ONE op with NO big intermediate; every
   alternative (slice ch1:9, or ReduceMax over rows) forces a ≥1080–1200B intermediate, so the
   300-param kernel is cheaper on the mem+params metric.
L(1800)+Conv(300)+MatMul matrices(~480)+predicate vector swarm(~700)+ramps ≈ 3650. Even fully
squeezed this lands ~3500 → ~16.75, BELOW the deployed 16.912 and below P 16.79. The +0.3 bar
(total < 2723) is unreachable.

## OPEN ANGLES (re-attack backlog)
- Reverse-engineer galaxy_v1 (mem 3192, **params 63**): it hits 16.912 with almost NO kernel
  params — implying a col/colour recovery that needs no 300-elem Conv (maybe a tiny ArgMax on a
  cheaper bg-excluding signal I missed) and a ~3000B plane. If its col-recovery is param-light,
  grafting it onto this fp16-L (1800) MatMul body could drop total toward ~2100 → ~17.5. This is
  the only plausible path to the +0.3 bar.
- uint8 L: find a single op that yields a uint8 [1,1,30,30] from rank-1 vectors (would halve the
  plane to 900B). Blocked today by ORT (Add/Mul reject uint8; Where needs a full-plane cond).

## INSIGHT (transferable)
⭐ A boustrophedon / multi-row-regime value plane is a SUM OF RANK-1 OUTER PRODUCTS when the lit
regions are DISJOINT IN ONE AXIS (here columns: even-offset / %4==1 / %4==3). Build it as ONE
`MatMul(rowmat[1,1,30,J], colmat[1,1,J,30])` — the J row-regimes (incl. out-of-grid-row
sentinel) collapse into a SINGLE fp16 L (the only full plane), strictly leaner than nested
Where (≥2 planes). ⭐ The 30×30 OUTPUT padding is target-ALL-ZERO, not bg-channel-1: route
out-of-grid cells to a sentinel value OUTSIDE [0,9] (e.g. −100) inside L so `Equal(L,arange)`
lights no channel there — fold the column sentinel into the colour value vector and the row
sentinel as an extra MatMul rank-1 term (const −100 × og_row indicator). ⭐ Per-cell one-hot
input means a plain occupancy ReduceSum gives 10/in-grid-column with NO seed signal; isolating a
non-background pixel's position ALWAYS needs a channel-weighted (bg-zeroed) reduction.
