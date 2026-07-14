# task338 — d5d6de2d

**Rule:** size x size grid (size=5*mult, mult in [2,5] -> size in {10,15,20,25}), background
black(0). Several non-overlapping (separated by >=1) solid red(2) rectangles; each box's INTERIOR
((tall-2)x(wide-2) inner block) is reset to black, so every box is a 1-cell-thick red ring around a
black hole. OUTPUT: interior holes -> green(3); red ring + outside background -> black(0).
**Current:** 14.285 pts, ext:kojimar6275, mem 43200, params 1816
**Target tier:** A — closed-form column ray-cast (parity of horizontal-wall crossings); no flood-fill.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-dir any-red prefix/suffix-OR enclosure | A | 25750 | 1289 | — | 191/200 | FAIL: gap cell surrounded by 4 separated boxes (global OR merges) |
| 2 | horizontal red-run-start parity (left scanline) | — | 27575 | 1301 | — | 86/200 | FAIL: solid top/bottom EDGE rows leak parity rightward |
| 3 | Hm=3-consec-red horizontal-wall, parity of Hm ABOVE (task204 ray-cast) | A | 15765 | 666 | 15.293 | 200/200 | works |
| 4 | cast red->fp16 once, Conv in fp16 (kill 2x fp32 2500B planes) | A | 14515 | 667 | 15.372 | 300/300 | adopted-as-best |
| 5 | uint8 QLinearMatMul for parity count (`Tl @ Hm8`) + uint8 Mod | A | **13890** | **667** | **15.414** | 500/500 | ADOPTED |

## Best achieved
15.414 @ mem 13890 params 667 — adopted. Beats prior live 15.372 by +0.042. Fresh 500/500.

## Irreducible-floor analysis
Dominant: the fp32 red slice red_f ([1,1,25,25]=2500B, Slice preserves input fp32 — the irreducible
entry plane) plus ~4 fp16 full planes (red, c1h(Conv), Hm, cnt(=Tl@Hm), par(=cnt mod 2) @ 1250B each).
The MatMul ray-cast (cnt) + Mod (par) are intrinsic to the column parity; W=25 (size<=25) is the active
canvas (can't crop tighter — boxes can sit anywhere). This sits just above task204's ~15.2 floor (its
W=20 makes its planes 36% smaller).

2026-06-28 update: the parity count itself does not need fp16. `Hm` is {0,1}; cast it to uint8 and run
`QLinearMatMul(Tl_u8, Hm8)` with scale=1/zp=0, then integer `Mod 2`. This halves `cnt` and `par`
from 1250B each to 625B each. `c1h/Hm` stay fp16 because the horizontal-wall detector is still a Conv.

## OPEN ANGLES (re-attack backlog)
- Fold notred into greenb without a separate Not plane (~625B, ~+0.04).
- The two ReduceMax in-grid profiles are tiny; gridb is a 625-bool full plane only for the off-grid
  sentinel — could be folded into the Pad sentinel if off-grid cols/rows were sliced exactly, but size
  is data-dependent (10/15/20/25) so a fixed W=25 slice can't auto-drop the off-grid background.
- A data-dependent W-slice (Gather by size scalar) would shrink every plane ~size^2/625 but trips the
  symbolic-dim "could not be measured" trap — not worth it.

## INSIGHT (transferable)
⭐ "Fill the hollow interior of each axis-aligned red box" is the SAME structure as task204 and is NOT a
flood-fill wall: the WINNING discriminator is the column ray-cast — Hm = a HORIZONTAL-WALL cell (red with
red BOTH left & right, via a 1x3 sum-Conv + bias -2 + Relu, so it fires ONLY on top/bottom edges, never
on 1-wide vertical side walls or isolated/gap reds), then interior = PARITY of Hm cells strictly ABOVE
(tril MatMul + Mod-2). Two WRONG approaches that look plausible but fail: (a) 4-direction any-red
prefix/suffix-OR enclosure merges separated boxes when a gap cell happens to have a box on each of its 4
sides; (b) horizontal red-run-start left-scanline parity leaks rightward on solid edge rows (one
unmatched crossing). The Hm restriction to genuine horizontal walls is exactly what makes the crossing
count local and edge-row-safe.

## 2026-06-29 — adopted uint8 red-run detector

Current source replaced the remaining fp16 horizontal-wall detector:
`Cast(red_f)->fp16; Conv([1,1,1], bias=-2); Relu; Cast->uint8`
with the exact integer form:
`Greater(red_f, 0.5); Cast->uint8; QLinearConv(sum3); Equal(3); Cast->uint8`.

Stored eval: **15.552139910765101 pts @ mem 12015 params 666**, 267/267.
Fresh side-by-side against previous live on 1000 eligible generator examples:
**divergence=0, base_fail=0, candidate_fail=0**. Adopted as
`custom:task338+qlinear-red3`.

Follow-up source cleanup reused the existing `three8` scalar for the green label
instead of carrying a duplicate `u3` initializer. Stored eval is now
**15.552218772008784 pts @ mem 12015 params 665**; fresh side-by-side against the
previous live on 500 eligible examples had divergence 0 and candidate_fail 0.

Reusable insight: thresholded fp16 Conv+bias+Relu run detectors can become
uint8 QLinearConv plus exact `Equal(count)` when the rule accepts one integer
count. This removes fp16 full-canvas Conv/Relu planes without changing the
semantic detector.

## S9 (2026-07-03) — RESHAPE-lens probe: FLOOR (rule is recolor, not reshape)
Output shape == input shape (fill hollow red-box interiors green) — Kronecker/crop levers
N/A. 18 counted planes all load-bearing at minimal dtype: red_f 2500 fp32 entry,
L30 900 one-hot carrier, 10×625 u8/bool chain, extent 1D 350, params 666 (Tl cumsum 625).
Measured rejections: u8-cond Where INVALID_GRAPH; CumSum no u8/fp16 (int32=2500B worse);
arithmetic mux ≥11 planes; free-input einsum loses (fp32 25×25 + red mask still needed);
signed-einsum N/A (variable #rects). DO NOT re-probe without new mechanism.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 10666 -> 9250 (+0.142)
Mechanism: value_info Slice crop + CumSum/Mod.
Gate (fresh_verify, inc/cand fail on 1500-2000): 0/0 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 