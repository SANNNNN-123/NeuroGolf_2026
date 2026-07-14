# task018 — 0e206a2e

**Rule:** 1–2 "creature" sprites (continuous_creature, 6–12 px in a wide×tall box, wide+tall=9), grid 12–24.
Each sprite is placed TWICE. Placement A ("original"): full sprite = a mode-colour body + exactly 3 uniquely-
coloured marker pixels (color_list[0,1,2]); drawn in both input and output. Placement B ("clone"): a RANDOMLY
ROTATED copy (rot∈{1,2,3,4}); in the INPUT only its 3 markers are shown (mode-colour body hidden), in the
OUTPUT the full rotated body+markers are shown. The output ERASES the original entirely and shows only the
reconstructed clone(s). To solve from input you must: group the 3 clone markers per sprite, find the rigid
transform (rotation+offset) mapping the original's markers to the clone's markers, then stamp the rotated full
body at the clone. mode = most-frequent colour (shared body colour). With 2 sprites both sprites share the SAME
3 marker colours, so colour does NOT identify which sprite — grouping is spatial.

**Current:** 13.34 pts, deployed net = 1395 nodes (Conv/ArgMax/TopK/ScatterND/Mod/Gather…), mem-heavy.
**Target tier:** detection/reconstruction — but see floor analysis: this is an INFO-BOTTLENECK WALL.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full numpy reference solver (CC-label originals, group clone markers, enumerate rot×offset, verify) | — | — | — | — | 297/300 | the 3 fails are GENUINELY AMBIGUOUS inputs (markers→multiple distinct valid outputs) |
| 2 | deterministic tiebreak (min-rot / max-rot) on ambiguous cases | — | — | — | — | n/a | rejected — true rot is uniform random, min-rot matches truth only 42% |
| 3 | bounded-iteration unrolling | — | — | — | — | n/a | N/A — blocker is not iteration depth; the rotation is simply not encoded in the input |

## Best achieved
No improvement attempted/adopted. Deployed net measured at fresh **496/500 = 99.2%** — already at the
information ceiling. Beats prior? N (no better net is possible).

## Irreducible-floor analysis
NOT a memory floor — an **information floor**. The generator chooses `rotates[i]` (and, for 2 sprites, the
clone marker grouping) UNIFORMLY AT RANDOM, and the input only shows the clone's 3 markers + the original's
full shape. Measured irreducible ambiguity: **1-sprite ≈0.7%, 2-sprite ≈2.0%** of fresh instances admit
≥2 valid (rotation, grouping) interpretations that produce DIFFERENT outputs. In ambiguous cases the true
rotation is split across rot∈{1,2,3} (18/9/4) with no deterministic tiebreak recovering it. Hence NO net —
exact or otherwise — can pass fresh 200/200; the achievable ceiling is ~98–99%, exactly where the deployed
net already sits (99.2%/500). The "4/200 failures" are the irreducible ambiguous cases, not a fixable bug.

## OPEN ANGLES
- None that change the verdict. (Confirmed: no marker-colour-order signal, no body-orientation signal, no
  tiebreak. The disambiguating bit was discarded by the generator.)

## INSIGHT (transferable)
⭐ task18 is a TRUE info-bottleneck WALL (same family as 219/255): the deployed 1395-node net already operates
AT the ~99% information ceiling, so the blank ledger note was a true wall, NOT a false-positive. Method lesson
for sprite-rotation-reconstruction tasks: before investing in an ONNX rebuild, run a Python *candidate
enumerator* over fresh instances and check whether the markers/visible-pixels UNIQUELY determine the answer.
When wide==tall or markers are rotation-symmetric, a 90°-rotation set leaves multiple valid placements ⇒
unsolvable. Don't chase fresh 200/200 on tasks whose generator picks an unobservable random rigid transform.

# (appended) S8 2026-07-02 — iteration collapse via graph surgery (+0.193) ADOPTED
Candidate replaced part of the 10-MaxPool iteration with einsum-based planes (graph surgery on
the incumbent module): mem 38441→31367, params 1018→1180 (total 39459→32547), 14.417→14.610.
Bit-identical on all 266 bundled vs deployed onnx; fresh 2500: cand 39 = inc 39, div 0
(the ~1.6% fail is the incumbent's own inherent rate). Adopted via ONNX materialization +
live_to_exact_source (candidate imported src.custom.task018). NOTE: agent report incomplete
(stuck in a monitor wait-loop) — mechanism details in the candidate file docstring at the
S8 scratchpad task018/cand.py if ever needed.

## S9 (2026-07-03) — fp16 walk-einsum chain recast (+0.049) ADOPTED
Sign-only consumption proof: cand_WA/WB [1,1,13,13] einsums + their ReduceMax reductions
feed only Greater(_,0); operands bounded nonneg → fp16 bit-identical. 6 Casts to=1→10,
2 inits recast, 12 value_info fixed. mem 31367→29807 (−1560), params 1180 unchanged.
Gates: stored fail=0; div 0/400 random (agent + orchestrator independent); fresh cached
2500: 55/55 inc=cand, uncached 600: 18/18, 0 divergence. Latency 0.9ms.
Adopted via ONNX materialization + live_to_exact_source (backup reports/retired_networks/
task018_pre_s9.onnx). Remaining floors priced: 3600 fp32 Conv read, 2×1800 fp16 TopK
ramps (values>255 so fp16 minimal legal), 14×900B+20×169B 1-byte planes, 0 CSE dups,
13a N/A (rotated-body ScatterND output). This was the last clean lever on this net.

## 2026-07-03 S12 — train-to-golf phase-2(Conv→ReLU→Conv, hidden-C) KILL
phase-1 단일 Conv에서는 k7-rate small로 최상위 EV 후보(+2.2)였으나, hidden-C 프로브 실패 — k7 C=4 NOT SOLVED, best_total_viol 213,990/1.50M 패치(60k steps, viol-boost oscillation, ~200k 아래로 못 내려감); C=1은 dead-ReLU collapse. C=4는 이미 economical capacity(C∈{1,2}) 초과라 경제적 경로 자체가 불가. 상세: reports/train_to_golf_report.md. 재탐사 금지 (단일노드/저-C 경로 모두 이 태스크 패치셋에서 선형분리 불가).


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 27616 -> 25445 (+0.082)
Mechanism: canvas-crop surgery (30x30 shrunk to provable NxN, re-inflate at terminal).
Gate (fresh_verify, inc/cand fail on 1500-2000): 31/31 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 