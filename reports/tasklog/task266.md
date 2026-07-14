# task266 — a9f96cdd

**Rule:** Input is ALWAYS a 3x5 grid, all black except a single red(2) pixel at (row,col). Output stamps four FIXED colours at the diagonal neighbours of red (only if inside the 3x5 grid): green(3) at (row-1,col-1), pink(6) at (row-1,col+1), cyan(8) at (row+1,col-1), orange(7) at (row+1,col+1). Background black(0) elsewhere in the grid; everything OUTSIDE the 3x5 grid is all-zero (convert_to_numpy fills only the 3x5 cells). Pure shift-and-recolour of the single red channel.
**Current:** 18.186 pts, conv3x3+b (910 params dense [10,10,3,3]), mem 0
**Target tier:** B (colour-index plane routed to free output) — fixed local motif keyed on one input channel, no detection wall.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | small-canvas Conv [9,1,3,3] -> [1,9,3,5] fp32 + Pad | B | 600 | 108 | 18.44 | 200/200 | works, MARGINAL (+0.25) |
| 2 | colour-index conv [1,1,3,3] -> lab[1,1,3,5], Equal->one-hot[1,10,3,5], Cast u8, Pad | B | 420 | 37 | 18.875 | 500/500 | ADOPTED (+0.69) |

## Best achieved
18.875 @ mem 420 params 37 — adopted? Y. Beats prior 18.186? Y (+0.69, well past +0.3).

## Irreducible-floor analysis
The dense [10,10,3,3]=910-param public Conv had mem 0 but its 910 params capped it at 18.19; Conv params can't drop below 910 without slicing input (group convolution can't route red ch2 to scattered out-channels 3,6,7,8, and out_channels must reach index 8). Breaking it required the SMALL-ACTIVE-CANVAS escape: the grid is always 3x5, so slice red to [1,1,3,5] (60B), run a single colour-index conv [1,1,3,3]->lab[1,1,3,5] (60B) whose 4 diagonal taps carry the colour values {3,6,7,8} directly, Equal vs arange[1,10,1,1] -> one-hot[1,10,3,5] (bool 150B), Cast to uint8 (150B, because ORT Pad rejects bool — confirmed), Pad to [1,10,30,30] as the FREE output. The two [1,10,3,5] planes (bool Equal result + uint8 Pad input) = 300B dominate; both are inherent to "expand-then-pad" since Pad rejects bool so the Cast cannot be elided. Off-grid stamp dropping and off-grid all-zero target both fall out for free (we only build a 3x5 result and Pad zero-fills).

## OPEN ANGLES (re-attack backlog)
- Eliminate one of the two 150B [1,10,3,5] planes: if a uint8 one-hot could be produced WITHOUT the intermediate bool (e.g. an integer comparison that yields uint8 directly), mem would drop to ~270B (~19.25). ORT has no direct uint8-yielding equality; Equal always emits bool. Possibly a Mul/clip arithmetic one-hot in uint8 — but ORT Mul rejects uint8.
- Build the 4 colour channels as a [1,4,3,5] block + bg, skipping channel 9 expansion — but scattering to indices 0,3,6,7,8 needs Split+Concat, likely more intermediates than the single Equal+Pad.

## INSIGHT (transferable)
⭐ A "fixed local motif stamped around a marker" task whose generator grid is a FIXED SMALL size (here always 3x5) is NOT a 3x3-neighbourhood at-floor task — slice to the tiny active canvas, encode the motif as a single COLOUR-INDEX conv whose kernel taps carry the colour VALUES directly (out[R,C]=Σ colour_k·red[opposite-diagonal]), then Equal->one-hot->Pad into the free output. The 4 stamp positions being distinct guarantees no tap collision so the index plane is unambiguous, and the colour-index conv ([1,1,3,3], 9 params) replaces the dense [10,10,3,3] public Conv (910 params): 18.19 -> 18.875 (+0.69). Confirmed again: ORT opset-11 Pad rejects BOOL, so an Equal->one-hot must Cast to uint8 before Pad (costs one extra small plane).


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.118)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 311 (mem 120 + params 191), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: op-chain (Pad x5/BitwiseXor/Max, 11 nodes) -> 2-layer Conv+Relu MLP (3 nodes); algorithm redesign.
