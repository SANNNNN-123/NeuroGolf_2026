# task246 — a2fd1cf0

**Rule:** `common.hpwl`. Red dot (2) at (r0,c0), green dot (3) at (r1,c1) on a
black grid (size 10–20). Output keeps both dots and draws a cyan (8) L-path:
horizontal along row r0 strictly between the two columns, then vertical along
col c1 from r0 (inclusive) to r1 (exclusive). Corner (r0,c1) is cyan.
**Current:** 16.29 pts, custom:task246 (`_hpwl` shared 10-ch double-MatMul), mem 5520, params 520
**Target tier:** A (separable bilinear → free bool output) — path is a union of two
rank-1 (row⊗col) regions plus two single-pixel dots, so a single colour-index
plane suffices; the 10→1 reduction routes into the FREE output, no per-cell colour army.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | single-ch colour plane L=R@M@C, 3 feats [a,b,c], Equal→bool | A | 4740 | 1219 | 16.31 | 0/200 | off-grid ch0 wrongly True (no in-grid mask) |
| 2 | +sentinel feats [a,b,c,in,ones], 99 off-grid | A | 5820 | 1295 | 16.13 | 200/200 | correct but 4 convs + f32 concat heavy |
| 3 | width-18 convs, fp16 per-feature concat | A | 4320 | 815 | 16.46 | 200/200 | downstream fp16 |
| 4 | 1 conv/axis (v∈{0,1,2}, w=clip, g=v−w); fp16 clip/maxpool | A | 4080 | 457 | 16.58 | 200/200 | −2 convs (−360 params), all ops fp16 |
| 5 | inclusive `b` both axes (strict cols via bc−wc folded into M) | A | 4020 | 457 | 16.59 | 200/200 | drops the col Sub; +0.30 boundary |
| 6 | feed `v` directly (a=2w−v, c=v−w in M), drop g Subs | A | 3900 | 457 | 16.62 | 500/500 | ADOPTED |

## Best achieved
16.6205 @ mem 3900 params 457 — adopted? Y. Beats prior 16.29384? Y (+0.327).
evaluate() ok pass 266/266; ISOLATED fresh 200/200 and 500/500.

## Irreducible-floor analysis
Dominant intermediate is the single colour-index plane L=[1,1,30,30] fp16 = 1800B
(the per-cell entry plane; cannot go below fp16 per FLOOR_RESEARCH). Everything
else is tiny: two [1,1,5,30] fp16 MatMuls/Concats (300B each) and the f32 floor of
4 input-reading ops (2 ReduceMax in-grid + 2 colour Convs, 120B each). The two
ReduceMax (in-grid 0/1 mask) MUST be f32 max-reductions (a Conv sum would give a
data-dependent COUNT that breaks the −99·in·in / +99·ones·ones sentinel
cancellation), so the 480B f32 floor is structural for this construction.

## OPEN ANGLES (re-attack backlog)
- Eliminate the +99/in/ones sentinel pair (2 of the 5 features, ~120B of MC/concat
  + the 2 ReduceMax 240B f32) if off-grid could be masked without an in-grid term —
  but in-grid-bg and off-grid both have all path-features 0, so a constant (ones)
  AND an in-grid product seem unavoidable for a single-plane Equal route.
- The 1800B L plane is the hard floor; only a Tier-S spatial-copy escape (N/A here,
  the path is synthesized not copied) would beat it.

## INSIGHT (transferable)
⭐ An L-shaped / Manhattan path between two marked points is a SEPARABLE bilinear:
collapse the whole thing to ONE single-channel colour-index plane L = R@M@C with
per-axis features [w=occupancy, v∈{0,1,2} endpoint value, b=between-inclusive band,
in=in-grid occ, ones], and route via Equal(L, arange)→bool — strictly cheaper than
a 10-channel double-MatMul (1800B plane vs 2400B R10). Two levers stacked here:
(1) feed the raw conv VALUE v (1=red,2=green) as a feature so endpoint identities
a=2w−v, c=v−w fall out of M's integer coefficients — no Sub planes;
(2) after the one mandatory f32 entry (Conv/ReduceMax read the f32 input), Cast to
fp16 and run ALL downstream vector ops (Clip/MaxPool/Mul) in fp16 — ORT_DISABLE_ALL
runs fp16 Clip/MaxPool/Mul/Sub fine (only the 10→input cast is forbidden, 18000B).
Off-grid masking for a single-plane Equal route needs a +K·(ones·ones − in·in)
sentinel (value ∉ {0..9}) so off-grid matches no channel.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 2753 -> 2637 (+0.043)
Mechanism: Einsum/value_info crop. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].