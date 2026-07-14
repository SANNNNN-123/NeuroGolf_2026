# task278 — fixed-colour halo stamp

**Rule:** output = input with a colour-3 halo painted on the BACKGROUND cells of
a dilated box around each qualifying red(2) anchor (`green_u = bg AND
MaxPool(QLinearConv-anchor)`).  The input 2s and background are unchanged.

## 2026-06-30 — plane-free Where routing (ADOPTED, +0.11)

The golf agent marked this "floor-bound: 900B label + fp32 slices."  It missed
that the output is `input + fixed colour 3 on the green cells`, so it routes:
`output = Where(Pad(green_b), onehot[3], input)` straight into the free output.
This drops the colour-index label-combine chain `rg = red+green`, `rg3 = rg*thr`,
`cguK = c0u+rg3` and the final `Equal` — the input 2s/background no longer need
to be rebuilt into a label.  (Contrast task265, where the same idea is net-zero
because it has only ONE combine plane to drop; 278 has three.)

**Result: 16.391 pts @ mem 5436, params 47 — 265/265** (was 16.279 @ 6084).

**Verification (the careful part):** stored 265/265.  OLD-vs-NEW on 3000 random
2-marker grids first showed 719 mismatches — but ALL were grids >18×18 where the
incumbent only processes its fixed 18×18 active slice (treats rows ≥18 as
off-grid) while the routed version preserves the full input.  Real 278 grids are
**≤18×18** (max scored 18,18).  Re-running capped at ≤18×18: **0/4000 mismatches**.
So new ≡ old on the entire real domain (and is strictly more robust beyond it).
Source-owned, networks rebuilt, parity confirmed.

⭐ Lesson: a scary raw-mismatch count is not a veto — classify it.  Here it was an
out-of-domain grid-size artifact, confirmed by capping to the real ≤18×18 domain.

## 2026-06-30 — S6 WIN: route olive halo via Where onto FREE output
- Incumbent (ext:franksunp) rebuilt a red+green colour-index label and Equal-expanded it
  (3 label-combine planes, ~648B) before emitting the halo.
- Re-fit: `Where(green_mask, onehot[green], input)` routes the olive halo directly into
  the FREE output; detection unchanged (QLinearConv center-detect → MaxPool 3×3 dilate →
  Min(ch0,halo)). Deriving bg as 1−red was tried and is INCORRECT (off-grid sentinel +
  boundary-olive spill) → 5436 is the floor for this formulation.
- Verified vs the REAL incumbent (networks/task278.onnx): **0 divergence on 3000 fresh
  + 265 bundled, both fail=0.** Safe equivalence-preserving golf.
- **mem+params 6131→5483, pts 16.279→16.391 (+0.112). ADOPTED (custom:task278).**

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k5(cost 5483): 19.7k viols 고착. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.
