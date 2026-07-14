# task214 — ARC-AGI 8e5a5113

**Rule:** size=3, grid 3x11. Block1 (cols 0-2) = S (identity). Gray separators
(channel 5) at output cols 3,7. Block2 (cols 4-6) = output[c][6-r]=S[r][c].
Block3 (cols 8-10) = output[2-r][10-c]=S[r][c]. Both are fixed rotations/reflections
of the 3x3 colour block S=input cols 0-2. Off the 3x11 region everything is all-zero
(no background channel set). Pure fixed-geometry one-hot copy/relabel, scored (out>0).
**Current:** 17.7594 pts, single GridSample(free fp32 input, grid[1,3,11,2]) -> gs_out
[1,10,3,11] fp32 (1320B) -> Pad. mem 1320, params 75.
**Target tier:** A (closed-form spatial copy) — non-separable 2-D remap (blocks 2,3
couple r&c), so it's a 2-D gather, not a row/col outer product, but still mem-light.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | uint8 flat-gather, gray-from-col3, 2 reshapes | A | 1380 | 55 | 17.73 | - | worse (reshapes double-count) |
| 2 | drop 2nd reshape (2-D idx [3,11] on rank-3 data -> rank-4 out) | A | 1050 | 51 | 18.00 | - | +0.24, near |
| 3 | slice cols 0-2 (360 vs 480) + append synthetic gray cell via Concat | A | 970 | 61 | 18.0617 | 200/200 | ADOPTED, +0.302 |

## Best achieved
18.0617 @ mem 970 params 61 — beats prior 17.7594 by +0.302 (clears +0.3). fresh 200/200.

## Irreducible-floor analysis
Dominant planes: Slice [1,10,3,3] fp32 = 360B (Slice preserves the fp32 input dtype;
all 10 channels needed since colours are random across 1-9; cols 0-2 is the minimal
coloured source) + final gather reg [1,10,3,11] uint8 = 330B (the true 3x11 output
region, already at the uint8 floor). The fp32 slice tax is why a single free-input
GridSample (1320B, params-cheap) was already decent: any dtype reduction of the
sampled plane requires a cast/slice plane. The win came from (a) uint8 everywhere
after the one mandatory fp32 slice, (b) the FREE output (Pad writes "output"), and
(c) shrinking the slice to cols 0-2 by appending ONE synthetic gray source cell
instead of widening the slice to cols 0-3 (saved 120B fp32).

## OPEN ANGLES (re-attack backlog)
- Eliminate the [1,10,9] reshape (90B) + Concat (100B) overhead (~190B) — would need
  a way to append the gray lane without a separate op; net gain ~190B -> ~18.25.
- The 360B fp32 slice is the hard floor for any cast-based approach; only a
  free-input gather avoids it but then can't go below fp32 3x11 = 1320B.

## INSIGHT (transferable)
⭐ For a small-output-block GridSample fed by free fp32 input that does a NON-separable
2-D one-hot remap: replace it with `Slice(minimal coloured corner, fp32) -> Cast uint8
-> Reshape flat -> ONE Gather with a 2-D index -> Pad(=output, FREE)`. A 2-D Gather
index [H,W] on rank-3 data [1,C,K] yields rank-4 [1,C,H,W] directly (out_rank =
data_rank-1 + idx_rank) — NO second reshape. And CONSTANT output cells (here gray
separators) can be a single appended synthetic source cell via Concat, so the fp32
slice only covers the data-dependent colours, not the constants.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 767 -> 525 (+0.379)
Mechanism: single-node Einsum + Gather/Pad.
Gate (fresh_verify, inc/cand fail on 1500-2000): 0/0 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 