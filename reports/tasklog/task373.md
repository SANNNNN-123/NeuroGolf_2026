# task373 — e9afcf9a

**Rule:** Grid is ALWAYS 2 rows x 6 cols. Input row0 = colorA (solid across all 6 cols), row1 = colorB (solid). Output `output[(r+c)%2][c] = colors[r]`: even columns keep (A top, B bottom), odd columns swap (B top, A bottom). Outside the 2x6 region everything is background colour 0 (unchanged). Pure spatial permutation of a one-hot input.
**Current:** 18.76 pts, GridSample[1,2,6,2]+Pad (gs_out 480B fp32), mem 480, params 33
**Target tier:** S (spatial copy / fixed permutation of one-hot cells)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice col0 rows0-1 -> uint8 [A;B], reverse -> [B;A], Where by column-parity mask, Pad into free output | S | 240 | 25 | 19.42 | 200/200 | ADOPT-WORTHY |

## Best achieved
19.42 @ mem 240 params 25 — beats prior 18.76 by **+0.66** (>= +0.3 Y). fresh 200/200.

## Irreducible-floor analysis
The two colours are CONSTANT across the 6 columns, so one 2x1 slice of column 0
(`input[:,:,0:2,0:1]`, [1,10,2,1] = 80B fp32 entry) carries the whole instance.
Its axis-2 reverse is the swapped pair. The 2x6 block is one broadcast Where over
a fixed [1,1,1,6] column-parity mask, all in uint8 (one-hot is {0,1}, scored out>0),
Padded straight into the FREE output. Dominant tensors: the 120B uint8 Where block
(irreducible — a 10-channel 2x6 one-hot region must exist before Pad) and the 80B
fp32 entry slice (Slice preserves the fp32 input dtype; 2 rows x 1 col x 10 ch is the
minimum to carry both colours). 240B is essentially the structural floor for this
construction.

## OPEN ANGLES (re-attack backlog)
- Effectively at floor for the chosen tier. The only sub-240 idea would be emitting
  the block straight to the free output without a Pad-input tensor, but Pad needs a
  materialised 120B uint8 source; not reducible.

## INSIGHT (transferable)
⭐ For a "solid-row recolour/permute by column parity" task where each ROW is a
constant colour, the entire instance collapses to a single 2x1 (one-cell-per-row)
slice of column 0 — a [1,10,2,1] = 80B fp32 entry — and the whole spatial pattern
rebuilds by ONE broadcast Where against a fixed column-parity mask in uint8, then Pad
into the free output. Beats a GridSample-2x6 net (480B fp32 sampled plane) ~2x on
memory because GridSample forces fp32 on its full sampled block, whereas the
constant-per-row structure lets you carry only one column in fp32 then go uint8.
A small fp32 GridSample block is NOT at floor when the source is constant along the
sampled axis — slice the one informative column instead of sampling the whole region.
