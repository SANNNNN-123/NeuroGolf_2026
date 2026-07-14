# task113 — 496994bd

**Rule:** Fixed 10-row × W-col grid (W 2..10, height hard-coded 5 → 2*5=10 rows). Top rows hold a few solid-colour rows; bottom half + off-grid is background. Output row r = input row idx[r] with idx = [0,1,2,3,4, 4,3,2,1,0, 10..29]: rows 0-4 unchanged, rows 5-9 = vertical mirror of rows 0-4, rows 10-29 identity. Pure spatial row-copy.
**Current:** 21.60 pts, single Gather(axis=2, idx[30]), mem 0, params 30.
**Target tier:** S (spatial copy) — already realised by the public net; the question is only whether the param count can drop below 23.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 reverse Slice + Max OR | S | 36000 | 4 | 0 | — | wrong (mirrors whole canvas; OR breaks ch0 one-hot) |
| 2 | Slice top5 + reverse5 + tail20, Concat (fp32) | S | 36000 | 9 | 14.51 | pass | correct but tail-slice 24000B dominates |
| 3 | single Gather(axis=2, idx[30]) → output | S | 0 | 30 | 21.60 | 200/200 | matches public floor exactly |

## Best achieved
21.60 @ mem 0 params 30 — adopted? N (equals public, no gain). Beats prior 21.60? N (MARGINAL — cannot reach +0.3).

## Irreducible-floor analysis
Output height is fixed at 30, so a Gather(axis=2) index MUST have 30 elements → params=30 (params count ELEMENTS, dtype irrelevant). The remap [0..4,4..0,10..29] names 30 distinct source rows with no run a single Gather can compress. To beat 21.60 by +0.3 needs mem+params ≤ 22 (score 21.91), i.e. ≤22 params at mem 0. Any decomposition that shrinks the index (Gather a 10-row body + Concat 20 off-grid rows; Slice+reverse+Concat; runtime-built index via Range+Where) materialises a ≥6000B fp32 (the 20-row off-grid slice = 6000 elems × 4B = 24000B) or a few-hundred-byte index/arithmetic chain, collapsing the score to ~15-19 — all far below 21.60. mem=0 is the entire advantage and the length-30 index is irreducible. **At hard floor.**

## OPEN ANGLES (re-attack backlog)
- A single mem-0 op with <22 params that performs row-mirror+identity. None found: Gather index is dim-locked to 30; Slice(step −1) can only do a full-canvas reflection (0 params but wrong geometry); Conv/MatMul on the height axis cost ≥30 (banded) to 900 (dense) and the identity tail over 30 rows has no compact kernel. Considered exhausted.

## INSIGHT (transferable)
A fixed-canvas row/col REMAP (identity + a partial reflection) is a single mem-0 `Gather(axis=2, idx[H])` whose output IS the graph output — params = H (output extent), irreducible because the index is dim-locked. ⭐ When the public net is ALREADY this minimal Gather (mem 0, params = output dim), the task is at HARD floor: you cannot shrink a dim-length index, and any index-shortening decomposition pays a ≥6000B fp32 slice/concat that craters the score. BAIL MARGINAL fast. (Sibling lesson to the GridSample-1800 and mem-0-single-Conv floor rules.)
