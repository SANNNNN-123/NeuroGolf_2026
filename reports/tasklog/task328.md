# task328 — d22278a0

2026-06-28 status: installed/source-owned exact baseline = 16.159130 pts, mem
4971, params 1940.  `src/custom/task328.py` was re-synced from the live ONNX
because the previous semantic source scored only 14.939 / mem 23296 / params
113.

## Rule summary

The input is a square grid with 2-4 coloured anchor pixels at distinct corners.
For each in-grid cell, choose the uniquely nearest anchor by Manhattan distance;
if the winning anchor's Chebyshev distance is even, paint that anchor colour,
otherwise background.

## Current live mechanism

The installed graph uses a compact packed-template/lookup strategy rather than
the older explicit 18x18 distance-field semantic implementation.  It has lower
memory but high params from the packed template bank.

## Frontier status

Not immediately a 20+ candidate.  The old semantic source proves low params are
possible, but it pays many 18x18 distance/parity planes.  The live graph proves
lower memory is possible, but pays ~1940 params.  A real frontier breakthrough
would need a hybrid: keep the live packed-output route while replacing the
template bank with scalar/vector distance logic that does not materialize the
four 18x18 corner fields.  No safe rewrite has been identified yet.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.021)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 6634 (mem 4691 + params 1943), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).
