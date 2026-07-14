# task136 — 5c0a986e (two 2x2 boxes, each emits a 45-degree ray)

**Rule:** Grid always 10x10 (top-left). Two solid 2x2 boxes, non-overlapping,
diagonals >=3 apart: box value 1 at top-left (R0,C0) (input ch1), box value 2 at
top-left (R1,C1) (input ch2). OUTPUT keeps both boxes and adds two 45-degree
diagonal rays sharing the boxes' top-left corners: value-1 ray = `r-c==R0-C0 AND
r<=R0` (up-left to the edge); value-2 ray = `r-c==R1-C1 AND r>=R1` (down-right to
the edge). The two colours never collide.
**Current:** 16.52 pts, ext:kojimar6275, mem 4700, params 136
**Target tier:** B (diagonal label-map + final Equal). Not A: a single 45-degree
diagonal `r-c==const` couples r&c, so it is not a row-cond x col-cond separable
product. Reconstruction is fully closed-form (no detection floor): box corners
recover as scalars from 1-D occupancy profiles.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 box slices -> scalar corners -> RmC diagonal masks OR box -> uint8 label -> Pad -> Equal | B | 3348 | 86 | 16.86 | 500/500 | WIN +0.34 |

## Best achieved
**16.86 @ mem 3348 params 86 — adopted? N (build-only).** Beats prior 16.52 by
**+0.34**. Fresh 500/500 isolated.

## Irreducible-floor analysis
Memory 3348 dominated by: 900B padded uint8 [1,1,30,30] label `L` feeding the
FREE Equal (irreducible — Equal writes the 30x30 carrier before broadcasting to
the free [1,10,30,30] bool output); two 400B fp32 [1,1,10,10] channel slices
(box1/box2) that serve double duty as both the box masks (Greater) and the
ReduceSum profile sources — fp32 because ORT ReduceSum needs float and Slice
preserves the fp32 input dtype (casting to fp16 would ADD a plane, no net win);
200B fp16 RmC diagonal plane (the one irreducibly-2-D coupling of r&c); the rest
are ~100B bool/fp16 10x10 working planes. Everything else (R0,C0,R1,C1, d1, d2)
is already scalar.

## OPEN ANGLES (re-attack backlog)
- The two 400B fp32 box slices (800B) are the only non-floor target. Deriving
  R0/C0/R1/C1 from a 1-D ReduceMax(input, axes=[1,3]) / axes=[1,2] profile
  (120B vecs) instead of a 10x10 slice could drop the slices, BUT the box MASKS
  themselves are still needed (the diagonal misses 2 of the 4 box cells), and a
  box mask rebuilt from scalar corners (row-band AND col-band) is separable and
  cheap — could shave ~600B toward ~16.9. Marginal; not pursued (already >+0.3).
- Tier-A is structurally blocked: a single 45-degree diagonal is not separable.

## INSIGHT (transferable)
A two-ray "find the boxes and draw their diagonals" task is closed-form tier-B,
NOT a detection floor: recover each 2x2 box's top-left corner as scalars (min
row/col from 1-D occupancy profiles), then each ray is a single Equal on
`RmC = rowramp - colramp` against the scalar `R-C`, half-plane-gated by a scalar
`row<=R0` / `row>=R1` (built from a 1-D ramp Sub + Greater, no 2-D gate plane).
Box OR diag -> disjoint uint8 label -> Pad-sentinel -> Equal into the free bool
output. The only 2-D plane materialized is the shared RmC diagonal field.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.016)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 1194 (mem 1159 + params 35), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
