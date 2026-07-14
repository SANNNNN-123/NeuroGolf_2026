# task255 — WIN (fp16 Einsum-count recast)

- Before: mem=17423, params=293, pts=15.2178
- After:  mem=14663, params=293, pts=15.3871  (**+0.169**, −2760B mem)

## Mechanism
Geometry output = one Einsum `nchb,ncbw->nchw` over 8 rank-1 rectangles, then
`Greater(·, 0.5)`. The Einsum count is a sum of ≤8 products of {0,1} values → an
integer in [0,8], which fp16 represents EXACTLY (fp16 is exact for integers ≤2048).
Recast the two Einsum operand casts (`R_f` [1,1,30,8], `C_f` [1,1,8,30]) and the
`geom_cnt` [1,1,30,30] plane fp32→fp16, and point `Greater` at the existing fp16
`half_h` constant. Saves 480+480+1800 = 2760B.

## Gate (provably exact, verified)
- bundled 265/265 pass, fail=0.
- Equivalence vs original (git HEAD) build: **0 divergences on 265 bundled + 3000
  random in-domain inputs** (bit-identical, not a re-fit).
- Note: net has a pre-existing 76/2000 fresh generalization gap; candidate shares
  the SAME 76 (zero new divergence).

Landed: src/custom/task255.py + networks/task255.onnx rebuilt. Candidate at
reports/candidates/task255_cand.py.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 10809 -> 10379 (+0.041)
Mechanism: QLinearConv + Einsum vs FREE input.
Gate (fresh_verify, inc/cand fail on 1500-2000): 65/65 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. both fail 65/1500 equally; urad cheaper (likely a further-golfed descendant of our net).

## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.145)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 8978 (mem 8625 + params 353), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 57 | candidate fail = 57 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: 3x ConvTranspose+8 Reshape -> 3x QLinearConv; removes counted reshaped planes; mech (c).
