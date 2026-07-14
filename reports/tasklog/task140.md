# task140 — 6150a2bd

**Rule:** The whole grid is 3x3 (size=3), placed at the canvas top-left. Output is the 180-degree rotation of that 3x3 grid: `out[r][c] = in[2-r][2-c]`. Everything outside the top-left 3x3 is all-zero (the harness only sets the grid's 3 rows/cols — there is NO channel-0 background fill outside it, so Pad-with-0 is correct).
**Current:** 19.09 pts, Slice(neg-step)+Pad, mem 360, params 8
**Target tier:** A (spatial copy/permutation) — already there; this is the structural floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice[-1,-1]+Pad (current) | A | 360 | 8 | 19.09 | 200/200 | baseline = floor |
| 2 | Slice fp32 -> Cast uint8 -> Pad(opset13) | A | 450 | 17 | 18.85 | 265 ok | WORSE (fp32 slice still 360 + u8 90) |
| 3 | Cast input uint8 -> Slice u8 -> Pad | A | 9090 | 17 | 15.88 | 265 ok | WORSE (full cast plane 9000) |

## Best achieved
19.09 @ mem 360 params 8 — adopted? matches current. Beats prior 19.09? N (AT FLOOR).

## Irreducible-floor analysis
Dominant intermediate = cropped `[1,10,3,3]` fp32 = 10ch x 9 cells x 4B = 360B. Irreducible because:
- colours are arbitrary 0-9 → all 10 one-hot channels must pass through (can't drop channels);
- Slice/Gather preserve the fp32 input dtype; going uint8 needs a Cast, which either casts the full input first (9000B, measured) or adds a 2nd small plane after the fp32 slice (450B, measured) — both strictly WORSE;
- channel-collapse to a 1-ch index needs a full-canvas Conv/MatMul (≥3600B) so it can only run AFTER the slice and never shrinks the 360B entry;
- rot180 is a position-dependent permutation (NOT a convolution → no conv fusion), and fusing it into the FREE output with zero intermediate would need a full-canvas (≥3600B) gather/reshape.
Cutting all 8 params only reaches ~19.11 (negligible). So +0.3 is INFEASIBLE.

## OPEN ANGLES (re-attack backlog)
- None that beat 360B. The "uint8 whole-pipeline" lever fails here because the entry plane is a fp32 Slice (no uint8 source without a 9000B full cast). The "spatial-copy → mem 0" escape fails because rot180 needs a position-dependent permutation that any single-op formulation forces onto a full-canvas (≥3600B) intermediate.

## INSIGHT (transferable)
⭐ The uint8 whole-pipeline copy lever (task152) only pays off when the working plane is ALREADY small in a narrowable dtype at its creation. For a Slice/Gather pipeline the entry plane is fp32 (dtype inherited from the fp32 one-hot input); casting it to uint8 ADDS a plane rather than replacing one, and casting the full input first is catastrophic. A tiny fp32 crop (here 3x3x10 = 360B) of a 10-channel one-hot is already at floor for any per-cell permutation that keeps all channels — do not chase dtype tricks on it.
