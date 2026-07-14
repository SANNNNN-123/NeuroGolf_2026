# task253 — a61ba2ce

**Rule:** A 13x13 input holds exactly four L-trominoes (3 cells, one solid colour each,
one colour each). Every instance uses all four distinct 2x2-missing-corner orientations.
The 4x4 output is a FIXED corner-L layout; only the colours vary. Each colour is placed in
the output corner DIAGONALLY OPPOSITE its missing corner: missing-BR→output TL, missing-BL→TR,
missing-TR→BL, missing-TL→BR (each output L mirrors the input L's shape). Verified by
brute-force over 300 fresh instances.
**Current:** 14.22 pts, public Conv+Gather+Where chain, mem 47830, params 439
**Target tier:** B — closed-form scalar recovery routed into the FREE bool output; the one
fp32 10→1 colour-index plane is the irreducible entry.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf30 + per-orient 2x2 occ-conv Equal==3 + cf/cf-shift readout | B | 14604 | 141 | 15.40 | — | pass 265 |
| 2 | fp16 Pad (drop int32 L30) | B | 12740 | 139 | 15.54 | — | pass 265 |
| 3 | 4-ch orient conv + cfsum/3 box-sum colour, compare 3*ramp | B | 12402 | 127 | 15.56 | — | pass 265 |
| 4 | W=14→13 active region | B | 11409 | 127 | 15.65 | — | pass 265 |
| 5 | Where(gate,cfsum,0) fuses cast+mul | B | 10257 | 128 | 15.75 | — | pass 265 |
| 6 | single corner-code conv (TL1/TR2/BL4/BR8) replaces 4-ch conv | B | 9393 | 119 | 15.84 | 200/200 | ADOPTED |

## Best achieved
15.84 @ mem 9393 params 119 — beats prior 14.22 by **+1.62** (>= +0.3 ✓). Fresh 200/200.

## Irreducible-floor analysis
- colf30 fp32 [1,1,30,30] = 3600 B: the 10→1 colour-index Conv entry (Conv output spatial =
  input spatial; slicing the 10-ch input first costs 6760 B, worse). Per FLOOR_RESEARCH this
  is the hard floor for a colour-index plane.
- L30 fp16 [1,1,30,30] = 1800 B: the label map MUST be 30x30 to broadcast against ramp[1,10,1,1]
  in the final Equal that writes the FREE bool output. Already fp16-halved; sentinel 30 (=3*10)
  keeps off-grid all-zero.
- prod fp16 [1,4,12,12] = 1152 B: the 4-way parallel orientation selection — recovering 4 colours
  at once needs a 4-channel fan-out somewhere; per-orientation serial selection costs the same total.

## OPEN ANGLES (re-attack backlog)
- Collapse the 4-channel `prod` (1152 B) into a MatMul: flatten cfsum[144,1] and a runtime
  selection matrix sel[4,144] → colors[4,1] without a [1,4,12,12] plane. Blocked today because
  the runtime Equal still materialises the [1,4,12,12] sel before any contraction; would need a
  way to build sel cheaply. Potential ~ -900 B → ~16.0.
- cf32 fp32 slice bridge (676 B): unavoidable today (cast-then-slice of full plane is 1800 B).

## INSIGHT (transferable)
⭐ A 2x2 (or k×k) occupancy Conv with **power-of-two corner weights** (TL=1,TR=2,BL=4,BR=8)
collapses "classify which of the 4 shape-orientations sits at each box" into ONE single-channel
plane: the box code = 15 − (missing-corner weight), uniquely tagging orientation without a
per-orientation channel stack. Pair it with a ones-kernel box-sum on the colour-index plane
(= N·colour at an N-cell shape, missing cell contributes 0) and read colour by comparing the
box-sum to **N·ramp** (no fp division — keeps the readout integer-exact). This turns an
apparent shape-correspondence task into closed-form tier-B.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 511 -> 500 (+0.022)
Mechanism: single Einsum + Gather. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].