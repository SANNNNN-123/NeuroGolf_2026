# task360 — e3497940

**Rule:** Input is a 10-row x 9-col grid with a gray(5) separator at col 4. The LEFT block
(cols 0..3) and RIGHT block (cols 5..8) each carry a colour or black per cell; whenever both
sides carry a colour at the same (mirrored) position they carry the SAME colour. The 10x4
output is the union of the left block and the horizontally-MIRRORED right block:
`output[r][c] = left[r][c] if left!=0 else reverse(right)[r][c]`. Verified 200/200 fresh.
**Current (prior):** 15.96 pts (public ext:biohack_new)
**Target tier:** S — pure index/reduce graph, ZERO params, no colour Conv needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | one-hot Max(left,reverse(right)), all 10ch | S | 4800 | 21 | 0 | — | FAIL: ch0 multi-hot where one side black + other coloured |
| 2 | Max colour ch 1..9, rebuild ch0=1-anyColour, Concat, Pad (f32) | S | 6240 | 28 | 16.26 | 200/200 | pass, marginal |
| 3 | same, cast slices->f16, f16 downstream, f16 output (Pad direct) | S | 6000 | 28 | 16.30 | 200/200 | ADOPTED |

## Best achieved
16.30 @ mem 6000 params 28 — beats prior 15.96 by +0.34 (Y). fresh 200/200.

## Irreducible-floor analysis
Dominant cost = the two f32 input Slices (left & mirrored-right colour channels), each
[1,9,10,4] = 1440B = 2880B total. Slice preserves the input's f32 dtype, and input is declared
f32 by the harness, so the read cannot be made f16 without a Cast that itself costs the same as
keeping fold in f32. Tried label-space (Conv-to-index) to shrink the 9-channel reads to 1-channel
labels, but Conv needs all 10 channels (10ch slice = 1600B each, worse) and a label->one-hot
expansion needs a non-free [1,1,30,30] Pad (3600B) before the final Equal. So 2880B of f32 reads
is the structural floor; f16 downstream (fold/anycol/ch0/full + f16 output via direct Pad) trims
the rest to ~3120B → 6000B total.

## OPEN ANGLES (re-attack backlog)
- If a future scorer ever let the input be declared/cast f16 cheaply, the two reads halve to
  1440B → ~16.9 pts. Not currently reachable (input dtype fixed f32).
- Add-instead-of-Max (left+right, rely on out>0) removes one Max but still needs both f32 reads
  and the ch0 rebuild — no byte payoff.

## INSIGHT (transferable)
⭐ One-hot union tasks ("fold/overlay two regions of the same colour"): per-cell colour union over
channels 1..9 is a plain element-wise Max of one-hot slices — NO colour Conv / label map. But the
BLACK channel 0 must NOT be Max'd (a cell black on one side + coloured on the other would become
multi-hot, breaking the `out>0` decode). Rebuild ch0 = 1 - ReduceMax(folded colour channels) and
Concat it back. Output can be declared FLOAT16 and the final Pad writes straight to it (free), so
the whole tail stays f16. The irreducible cost is the f32 input Slices, not the arithmetic.
