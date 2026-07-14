# task044 (ARC 228f6490) — INFEASIBLE: shape-correspondence assignment wall

Deployed: kojimar 15.56 pts, 12444 B, 109 params, 180 nodes
(ArgMax×9 + GatherND + ScatterND + 2 Conv — heavy connected-component / assignment machinery).

## Rule (derived exactly)
Input: two hollow GRAY (5) boxes; two colored "sprites" (connected creatures) sitting
OUTSIDE their boxes; one "dust" color of scattered (mostly isolated) pixels.
Output: each sprite is ERASED from its location and STAMPED into a box interior at the
box's interior top-left corner `(brow+1, bcol+1)` (same relative shape). Dust + gray
unchanged. (Generator draws creature identically at sprite loc and box interior; the
in-box black marker is color-0 = invisible.)

## Why INFEASIBLE for a closed-form net
The transform needs: (1) connected-component creature extraction to tell sprites from
dust, (2) a per-sprite→box ASSIGNMENT, (3) per-sprite data-dependent 2-D translation of
an arbitrary connected shape into its box. (1)+(3) alone are already near the deployed
cost; (2) is the true wall:

- The assignment is NOT recoverable from any simple input feature. Measured over fresh
  instances: bbox-fit is ambiguous in **91%** (710/778) of cases; sprite-pixel-count→box-
  area rank is **wrong 27%** (133/485); largest-count→largest-box wrong 27%.
- The only local predicate that helps — "stamp creature at box-interior corner; valid iff
  it fits the interior AND lands only on background cells" — still yields a NON-UNIQUE
  matching in **195/3000 (6.5%)** of instances (dust can validly stamp by chance; two
  creatures can each validly stamp into either box).
- Dust-vs-sprite by connectivity (most-fragmented color) is **wrong 136/3000 (4.5%)**;
  count-2 sprite vs count-2 dust collide.
- No exact-fit disambiguator: the creature bbox equals its box interior in only
  2958/6000 (49%) of true pairs, so "tightest fit" cannot break the tie.
- These error sources don't even compose to an EXACT numpy reference (124/3000 fail with
  first-valid-permutation), so an EXACT closed-form ONNX net is impossible — the residual
  ~5% ambiguity is decided solely by the generator's hidden creature→box `idx` pairing,
  which leaves no signal in the input.

This is the documented multi-object shape-correspondence / ambiguous-template-recovery
wall (cf. task158, task279). Allowed ops (no Loop/NonZero/Unique/Compress) cannot do the
connected-component assignment + arbitrary-shape data-dependent translation exactly, and
even the deployed kojimar net (which can, with 180 nodes) only reaches 15.56. A from-
scratch closed-form net cannot reach EXACTNESS, let alone beat the score.

VERDICT: INFEASIBLE (assignment is input-underdetermined in ~5% of instances).

## S8 (2026-07-02) — priced FLOOR at 11119 (opus agent)
crop_f32 [1,10,10,10] 4000B = detection floor (per-colour matching needs the one-hot; fp16
copy already net-optimal). EPILOGUE FOLD MEASURED NEUTRAL here: 30×30 u8 index-pad (900+100)
== 10×10 one-hot bool (1000) exactly, +11 params → strictly worse. LESSON: the fold only wins
when it ELIMINATES a separate pre-output plane (task187's label+flood planes), not when the
index-pad swaps 1:1 with a small-grid one-hot.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k7(cost 11119): 1.26M 패치, 196k viols. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 9126 -> 5647 (+0.480)
Mechanism: rank-factored Einsum sandwich (initializer reuse counted once).
Gate (fresh_verify, inc/cand fail on 1500-2000): 2/2 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 

## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 5647 -> 4984 (+0.125)
Gate fresh_verify 1500: inc=1/1 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].