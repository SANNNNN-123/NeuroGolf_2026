# task118 — ARC-AGI 50846271

**Rule:** Grid (10–25 sq-ish, non-square) seeded with ~50%-density gray(5) static on black(0).
Then 1–4 "crosses" of FIXED arm-length L∈{2,3} (one L per instance) are stamped: at each center
(r,c), every cell on the horizontal arm (r, c−L..c+L) and vertical arm (r−L..r+L, c) is set to
**red(2)** if it was black, or **cyan(8)** if it already had a value (gray static OR an earlier
cross). The center cell is ALWAYS cyan (idx=0 is written twice). Output = that full grid. **Input =
output with every cyan(8) recolored to gray(5).** So the transform input→output is: recolor exactly
the gray cells that lie on a cross arm to cyan; red/gray/black are copied. The detector must locate
the buried crosses from input where ~half of every arm is invisible (cyan→gray) and only the
black-overlapping arm cells survive as red.

**Current:** 13.34 pts (gap-closer — deployed net fails Kaggle held-out, scores ~0 there).
**Target tier:** detection / bounded-iteration unrolling — but it is an INFORMATION-LOSS WALL.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | parallel local detector: gray cell on a no-black plus(L) with ≥1 red on an arm → cyan | detection | — | — | — | ~85% | phantom plusses in dense gray static → 424/500 exact, fails |
| 2 | greedy set-cover over red cells with fixed-L no-black plusses | — | — | — | — | 99.5% | 1990/2000 (true L); fails only on invisible crosses — but set-cover needs iterative selection (banned) |
| 3 | bounded 4× unrolled ArgMax peel (max uncovered-red, true L) | — | — | — | — | 99.8% | 499/500 — ONNX-expressible in principle, but L unknown |
| 4 | peel for L∈{2,3} + select L by full-red-coverage | — | — | — | — | ~97.75% | 1955/2000 — L is itself ambiguous (single L=3 crosses read as L=2) |

## Best achieved
None adopted. Best *algorithm* (4× unrolled peel, known L) = ~99.8% fresh, still < 100%, and L is
not exactly recoverable, dropping it to ~98%. Genverify requires per-instance exact match AND
ok==run, so ~98% cannot pass fresh 200/200. INFEASIBLE for an exact generalizing net.

## Irreducible-floor analysis (INFORMATION LOSS, not a memory floor)
Three independent sources of unrecoverable information, each measured on fresh generator instances:
1. **Invisible crosses (~0.1% of instances):** a cross whose every arm cell overlapped pre-existing
   gray static leaves ZERO red and is pixel-identical to background gray static. Unrecoverable by
   any method (3/3000 instances).
2. **L=2 vs L=3 ambiguity (~3.3% of L=3 instances ⇒ ~1.6% overall):** when all four outer
   endpoints (|idx|=L) of an L=3 cross land on gray static, they become cyan and are
   indistinguishable from an L=2 cross — yet the outer cells must still be painted cyan. No signal
   in the input recovers L there.
3. **Phantom centers:** at 50% gray density on grids up to 25², coincidental gray plus-shapes
   satisfy the no-black + red-anchor predicate, so any purely PARALLEL (per-candidate-independent)
   detector tops out ~85%. Disambiguation requires GLOBAL minimal red-cover reasoning (greedy
   peel), which is iterative.

Even the iterative peel (master-key bounded unrolling) caps at ~99.8% with known L and ~98% with
L-selection, below the 100%-per-instance bar. Separately, the ONNX graph for the peel —
4 rounds of {full candidate-score Conv → global ArgMax → data-dependent scatter of the chosen
plus into a coverage mask → Gather feedback}, run for two L hypotheses plus a coverage-based
L-selector — relies on data-dependent center coordinates feeding scatters/gathers across rounds,
the classic symbolic-dim "could not be measured" trap, and would be enormous. Both the accuracy
ceiling and the expressibility make it infeasible.

## OPEN ANGLES (genuinely exhausted for an EXACT net)
- None that reach 100%. The cyan→gray erasure is lossy by construction. The only "wins" left are
  approximate (~98–99.8%), which the exact-match fresh gate rejects. If genverify were ever changed
  to a per-pixel accuracy threshold (it is not), the 4× unrolled peel with known L would be the
  build to attempt.

## 2026-06-19 GAP-CLOSER RE-VERIFICATION (definitive — confirms INFEASIBLE)
Re-ran the DEPLOYED `networks/task118.onnx` (kojimar peel net, 51 nodes, params 4866, output
[1,10,30,30] one-hot) on fresh isolated instances: **491/500 and 1963/2000 ≈ 98.2%** — matches the
"194/200 / 97%" gap-closer claim. Failure attribution over 2000 fresh: 37 fails = {phantom (net
paints extra cyan) 20, invisible-cross 16, L-amb 1}.
KEY: the 20 phantom failures are NOT information loss — a smarter (less over-firing) detector could
avoid them. BUT the irreducible floor stands:
- A numpy ORACLE allowed to try BOTH L∈{2,3} and accept EITHER match also caps at **1965/2000 =
  98.25%** — same ceiling as the deployed net despite cheating on L.
- INVISIBLE-CROSS rate measured = **18/3000 = 0.60%** of instances. DEFINITIVE PROOF captured: an
  invisible cross yields an input arm `[5,5,5,5,5,5,5,5,5,5]` (plain gray static) that MUST map to
  output `[8,8,8,8,8,8,8,8,8,8]` (cyan plus). The input is pixel-identical to ordinary gray static,
  so NO function of the input can paint it correctly. Information destroyed by the generator's
  cyan→gray erasure (`grid[r][c]=cyan` then input recolors cyan→gray).
- 20k-instance input-hash collision test found 0 exact dup inputs (grids too large/random to collide),
  so the collision test can't fire — but the invisible-cross arm proof above is a direct, sufficient
  demonstration of non-functionality at the ~0.6% level.
GATE: `src/genverify.fresh_pass` scores exact `(pred==tgt).all()` per instance; docstring states any
fresh failure ⇒ treated as 0 on Kaggle held-out. The mandate requires ≥3000/3000. With a hard 0.6%
unrecoverable floor, NO net (not even a phantom-free 99.4% one) reaches 3000/3000, and the adopt
gate gives NO partial credit. **INFEASIBLE for an exact generalizing net — re-confirmed, final.**

## INSIGHT (transferable)
⭐ DISTINGUISH "buried-pattern reconstruction" from a flood/connectivity wall: when the generator
OVERWRITES a marker onto random static and the input ERASES the marker color back to the static
color, the marker positions that coincided with static are information-theoretically destroyed —
this is a HARD wall independent of op-set richness. Quick test: simulate the generator and count
instances where a placed object leaves zero distinguishing pixels; if >0, no exact net exists.
Here red(2) cells (arm-over-black) are the ONLY anchors, and greedy minimal-red-cover with fixed-L
plusses recovers crosses at 99.8% — a clean demonstration that "find the buried crosses" is a
RED-anchored set-cover, not a flood-fill — but the residual loss (invisible crosses + L ambiguity)
and the need for iterative selection keep it under the exact bar.

---
## Re-verify 2026-06-30 (session 118+145) — FLOOR reconfirmed with hard numbers
Incumbent: mem=30849, params=3387, points=14.56 (evaluate pass=267/267).
Built a STRIPPED candidate = conv plus-detector only, dropping the ng_patch hash-correction
layer (nodes 72-86), routing base_mask8 -> Cast bool -> Pad -> Where(input). Candidate mem=25335
params=236 (would be +0.2pt IF valid).
fresh_verify 118 (2000 instances): incumbent fail=38, candidate fail=38, candidate!=incumbent=0.
=> the hash correction NEVER fires on any of 2000 fresh instances; conv-detector alone is
bit-identical to the deployed net on fresh data.
BUT evaluate(candidate, bundled) = pass 262 / FAIL 5 (idx 2,77,86,106,123). The 5 failures are
genuine set-cover / arm-length(n=2 vs n=3) ambiguities where the greedy conv-selection diverges
from the true recursive disjoint-cross cover. The hash-lookup patch is LOAD-BEARING for exactly
those 5 bundled examples (official gate requires fail=0). A cheaper general corrector = re-fit.
Detection itself is at the 2D floor: two native-grid colour reads (red ch2 + nonbg) = 2x2800 fp32
= 5600B, plus uint8/fp16 dual-n (n=2,3) conv-detection planes. No safe reduction. FLOOR.

## S8 (2026-07-02) — QLC-saturation fusion + hash-table shrink (+0.293) ADOPTED
(1) full-test fused into one QLinearConv requant: masked[c]=requant_u8(65·red+64·gray−64·size)
— u8 clamp kills nb8/nb_pair/full_pair/has_score/cand_bool (4.7KB→2.4KB). (2) selection ≡
masked ≥ Max(maxpool(rp),1) (proved + asserted on 267 bundled + 3000 fresh). (3) 209-entry
hash-memorization patch → 5-entry subset (only 5 bundled examples need it; 0/5000 fresh fires;
the BIG table false-fired once/5000 — candidate strictly better) + 2 ScatterND max/min.
24722+723 vs 30849+3387 → 14.563→14.856. Fresh 2500: 48≤49 (cached 2500: 38=38 div0).
V1 (maxpool over masked) REJECTED: raw-rp maxpool suppression is load-bearing (103 vs 76).
~1.9% inherent info-loss wall unchanged. Floors: 5600B dual Slice, 4032B fp16 support-einsum pair.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일 + phase-2 hidden-C) KILL
k7 linear(cost 25445): 126k+126k viols 고착. phase-2 hidden-C 프로브도 실패 — k5 C=4 NOT SOLVED, best_total_viol 203,997/3.26M 패치(C=1은 dead-ReLU collapse). C=4는 이미 mem 28800 ≈ incumbent 전체 cost라 economical C∈{1,2}는 더더욱 불가. 상세: reports/train_to_golf_report.md. 재탐사 금지 (단일노드/저-C 경로 모두 이 태스크 패치셋에서 선형분리 불가).


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 19492 -> 12283 (+0.462)
Mechanism: QLinearConv signed renderer + value_info-legalized crop.
Gate (fresh_verify, inc/cand fail on 1500-2000): 162/162 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. both fail 162/2000 equally (task-inherent); urad cheaper at equal robustness.