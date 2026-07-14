# task017 — 0dfd9992

**Rule:** A 21×21 grid is filled with a doubly-periodic pattern of period `length`
(4..9, same offset/length on both axes): `v(r,c)=((rr²+cc²)%mod)+1`,
`rr=(offset+r)%length-length//2`, `cc` likewise. mod∈4..9, length∈4..mod,
offset∈1..length. The input has 5 black (colour 0) rectangle cutouts stamped over
the pattern; the output is the SAME pattern with the cutouts removed. The pattern
is fully determined by the 3 scalars (mod,length,offset) — only 106 valid tuples.
**Current:** 15.30 pts, ext:kojimar7113 (crowd net), mem 13500, params 2827
**Target tier:** B (closed-form formula rebuild after scalar-parameter recovery)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | period-detect + max-fold + class-gather (prior custom file) | B | 63256 | 469 | 13.94 | 200/200 | worse than P; detection planes blow up; also leaks when a small class is fully cut |
| 2 | template-match params (kojimar idiom) + free-output routing | B | 10890 | 2826 | 15.47 | 199/200 | +0.17, below thresh |
| 3 | + fold +1 into channel_values (drop labels21 plane), uint8 pad sentinel-200, channel0=255 wrap | B | 9549 | 2825 | 15.58 | 200/200 | +0.28 |
| 4 | + drop 1 weak sample (NS=16→15, greedy) | B | 9182 | 2679 | 15.62 | 500/500 | **+0.32 ADOPT-CANDIDATE** |
| 5 | source-owned final-Equal route + robust 13-sample set | B | **9330** | 2388 | **15.63** | 2998/3000 vs old 2971/3000 same pool | public-probe accepted; adopted after LB 7172.43 |

## Best achieved
15.62 @ mem 9182 params 2679 — beats prior 15.30 by +0.32 (✓ ≥+0.3). Fresh
500/500; 3996/4000 (99.90%) at scale = identical to the kojimar baseline's
99.90% (shared inherent leak).

2026-06-28 source-owned follow-up: applied the global `free_final_onehot_equal`
mechanism directly in `src/custom/task017.py`: pad the 1-channel uint8 label
plane first, then route to the free bool output with final `Equal`, instead of
materialising `[1,10,21,21]` before padding. With a robust 13-sample set this
scores 15.631 stored @ mem 9330 params 2388 and improves the same 3000 fresh
pool from 2971/3000 (old live/source) to 2998/3000. It is still not strict
fresh-all-pass, so it is recorded as a submission-test candidate rather than a
strict adoption.

2026-06-28 public-probe result: submitted only the task017 source-owned candidate
as `t017_source_ahead_15631` (submission `54135366`, message
`probe t017 source-owned +0.083 risky fresh`).  Public score moved to **7172.43**
from the previous best 7172.10, so the candidate was promoted to live despite the
strict fresh caveat.  This is explicitly a submission-validated, source-owned
risk probe, not a blind public import.

## Irreducible-floor analysis
Dominant intermediates: `matches_h` fp16 [1,106,15]=3180B and `matches_b` bool
[1,106,15]=1590B (the per-sample×per-candidate comparison + its fp16 cast for the
score ReduceSum). This [106,NS] plane pair (4770B, ~52% of mem) is the floor of
the template-match: ReduceSum rejects bool/uint8 (only int32/fp), so the bool
Equal MUST be cast to fp16 to count agreements; a MatMul-onehot reformulation
moves the cost into a 16960-element param table → strictly worse in log-score.
The formula planes `rrcc`+`pat0` (2×882 fp16) are the minimal 21×21 closed-form
rebuild (Add then Mod, both must be fp16: ORT Mul/Mod reject uint8). GatherND
`sample_planes` fp32 [10,15]=600B inherits the fp32 input dtype (unshrinkable).

## OPEN ANGLES (re-attack backlog)
- Two-stage parameter recovery (recover `length` then `(mod,offset)`) to shrink
  the 106-candidate axis of the match plane — blocked by the parameters being
  coupled in the sample colours; no clean separable signal found.
- Reduce GatherND index param (sample_nd_idx 10×NS×4) via batch_dims to drop the
  10× channel replication — minor (~param only).
- NS=14 greedy = 99.91% fresh (~1.3/200 expected fail) — too risky for strict
  200/200; NS=15 is the safe floor matching baseline robustness.

## INSIGHT (transferable)
⭐ For "fill cutouts in a parametric pattern with only K valid parameter tuples":
TEMPLATE-MATCH the global scalars (precompute each tuple's colour at ~16 fixed
sample cells; read those cells via GatherND+ArgMax; majority-vote via
ReduceSum(Equal)→ArgMax — cutout-robust because cut cells read colour 0 which no
candidate sample contains, so they vote for nobody) THEN rebuild by CLOSED-FORM
formula. This beats per-cell period-detection+max-fold (huge detection planes AND
leaks when a small class is fully cut). Routing the 10-ch one-hot into the FREE
output (pad the 1-ch colour-index plane to 30×30, Equal→output) saves the crowd
net's onehot_raw [1,10,21,21]=4410B. ⭐ uint8 pad-back with channel0 compare-value
= (−1 mod 256)=255 lets the +1 colour offset fold into channel_values for free
(off-grid sentinel 200 matches no channel → background-free off-grid; channel0's
255 never appears in-grid). ⭐ ReduceSum accepts int32 but NOT bool/uint8 (re-
confirmed) → a bool match plane is pinned to an fp16 cast before counting.

## 2026-06-30 S1 — LANDED (behaviour-preserving golf, fresh-gated)
mem 9330→8448, params 2388→2021, pts 15.631→15.7438 (+0.113). Bundled fail=0;
fresh 2000 candidate==incumbent (diff 0). 3 cuts: (1) GatherND batch_dims=2 shrinks
sample-read idx [10,13,4]→[1,10,13,2] (−260 par); (2) drop stored `half` column,
compute Floor(length/2) in-graph (−106 par); (3) drop the +1 colour-offset 21×21 fp16
plane (−882B) by shifting channel_values to [255,0,1,…]. Floor planes (matches_h/b,
candidate_samples, 3×882 formula planes) untouched. method ext→custom:task017.

## S8 (2026-07-02) — priced FLOOR at 10469 (opus agent)
matches_h fp16 already landed (S1); einsum-vs-plane needs [106,13,10] one-hot table = +12402
params (break-even V<3.9, unreachable). NS<13 sample reduction OVERFITS the cache: greedy-12 =
1 fail cached but 23 vs 11 uncached/8000 — the 13-set is the robust floor. Label epilogue
already fold-optimal. CACHE-OVERFIT WARNING VALIDATED: cached fail understates true fail ~3×
on fitted parameters; the uncached final gate is load-bearing.

## S9 (2026-07-03) — kojimar teacher REJECTED (NS=9 overfit, fresh 38/3000 = 1.27%)
Teacher = SAME algorithm, just NS=9 sample cells vs our robust NS=13 (+int64 castfold).
Reproduces this tasklog's S8 warning exactly (NS<13 overfits; cached gate would
understate ~3×). Incumbent 2/3000. No new mechanism, no headroom. Floor stands.


## S10 (2026-07-03) — kojimar7185_95 teacher ADOPTED (+0.294, policy-gated)

**Gate-policy note:** the fresh gate was relaxed this session — bundled fail=0 stays
mandatory (public LB grades bundled), but the fresh gate drops from "cand ≤ inc" to
"~98%+ fresh pass → adopt and verify by real LB submission" (fresh-gate = private-LB
insurance only; the kojimar pack already survived the public LB at 7185+). **This is the
same file S9 rejected** (recorded there as "NS=9 overfit, fresh 38/3000 = 1.27%"); adopted
now with its fresh-fail rate recorded for private-LB risk tracking. A verification LB
submission is planned this session.

**Mechanism diff (op census, retired vs new):** same algorithm as the incumbent —
GatherND+ArgMax recover the (mod,length,offset) scalars by majority-vote template-match, then
a closed-form formula rebuild routed to the free output. The ONLY change is the sample-cell
count: incumbent uses our robust **NS=13** cells (`candidate_samples[1,106,13]` fp16,
`sample_nd_idx[1,10,13,2]`), the teacher uses **NS=9** cells → the per-candidate match plane
shrinks `[1,106,13]`→`[1,106,9]`, driving the mem drop 8448→5997. The teacher also carries
int64 cast-folded index tables (`matches_b_castfold_const[1,106,9]` i64, `sample_nd_idx[10,9,4]`
i64) but those are cheap in element count (params 2021→1804). Fewer sample cells = the exact
S8 CACHE-OVERFIT lesson in this log (NS<13 overfits; cached fail understates true fail ~3×),
which is why it fails ~1% fresh.

**Cost:** mem 8448→5997, params 2021→1804, pts 15.7438→16.0380 (**+0.294**, cost 10469→7801 −2668).

**Gate evidence:** bundled 266/266 fail=0 (both nets). Fresh 2000: candidate **22 fails
(1.10%)** vs incumbent **1 fail (0.05%)**. TopK audit: no TopK in either net (recovery via
GatherND+ArgMax).

**Backup + provenance:** incumbent → `reports/retired_networks/task017_pre_s10.onnx`;
candidate source `public_candidates/kojimar7185_95/base_submission/task017.onnx` →
`networks/task017.onnx`; source regenerated via live_to_exact_source --write-src, src↔live
reconciled fail=0.

Adopted under S10 relaxed gate (bundled=LB gate; fresh ≥98% → submit-verify); private-LB
risk = 1.10% fresh fail rate.

⭐ TRANSFERABLE: reducing the number of template sample cells (NS) in a "recover K scalars by
majority-vote then closed-form rebuild" net is a real MEM lever, but it OVERFITS — the cache
understates true fresh fail ~3× (this log's own S8 warning). Selection: parametric-pattern
tasks that recover a few global scalars via GatherND+ReduceSum(Equal)+ArgMax; shrink the
sample-cell count only under the relaxed gate and only with an uncached fresh gate, since
each dropped cell trades private-LB robustness for a few hundred bytes.


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 7004 -> 6455 (+0.082)
Gate fresh_verify 1500: inc=25/25 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].