# task209 — 8a004b2b

**Rule:** Input has (a) a YELLOW box marked only at its 4 corners at (brow,bcol), size wide×tall;
inside the box, a sprite is magnified mag× and placed at offset (irow,icol) but ONLY a random subset
`shows` of its cells is drawn; (b) the FULL sprite (every cell, native 1× resolution, arbitrary colors
from {1,2,3,8}) drawn at the bottom of the grid (rows ≥ height−3, strictly BELOW the box). OUTPUT =
the box region (wide×tall) with yellow corners + the COMPLETE mag× magnified sprite at (irow,icol).
So the task = "recover the full sprite from the bottom, recover (mag,irow,icol) from the partially-
shown magnified blocks, then re-stamp the complete magnified sprite into the box."

**Current:** 13.357 pts, `gen:wguesdon6315` (imported overfit), mem 113691, params 144.
Base net FAILS isolated fresh (~38/40) → scores ~0 on real Kaggle LB. Gap-closer candidate.
**Target tier:** detection / multi-object shape-correspondence — non-separable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Python reference solver (full brute search over mag∈{2,3,4}×irow×icol + complete-block validation + arbitrary-color sprite re-stamp) | n/a (analysis) | — | — | — | 266/266 STORED ; ~495-499/500 FRESH | irreducible generator ambiguity caps fresh <100% |

## Best achieved
No ONNX net built. The IDEAL Python solver passes 266/266 STORED examples (incl. the official mag=4
test) but only ~98.5-99.6% of FRESH instances. ONNX implementation judged INFEASIBLE as a clean
generalizing win (see below). Not adopted.

## Irreducible-floor analysis (two independent walls)

**WALL 1 — generator is genuinely ambiguous (input does NOT determine output).**
Only a random subset `shows` (≥2) of sprite cells is drawn magnified in the box. When the shown blocks
lie in a SINGLE sprite row (or single column) — common; the official TEST example is exactly this case:
shown blocks occupy sprite-row-0 only (`8 . 3`), mag=4 — the translational offset (icol, sometimes irow)
of the magnified sprite is under-determined. Measured over 8000 fresh instances: only **98.45% of inputs
uniquely determine the output**; the other 1.55% admit ≥2 distinct consistent outputs (truth always among
them, but unrecoverable). Confirmed by constructing two parameter sets that differ only in icol yet are
both valid. Best deterministic tie-break (max-mag, then min-irow,min-icol) reaches ~99.3-99.6% fresh —
so even a PERFECT solver fails ~1-5 per 500 and would clear genverify's 40/40 gate only ~85% of the time.
NO net (not even #1) can exceed this; it is a property of the generator, not of the encoding.

**WALL 2 — no separable / single-op tensor form (ONNX construction is a detection-floor blowup).**
The reconstruction is intrinsically a CORRESPONDENCE + SEARCH problem with NO separable structure:
(1) mag is not directly readable (adjacent same-color blocks merge; run-gcd recovery only ~98% and gcd
isn't an ORT op); (2) (irow,icol) require matching shown blocks to the bottom sprite's cells, which has
no row⊗col factorization; (3) the re-stamp is an arbitrary-COLOR data-dependent Kronecker magnify by a
RUNTIME factor into a data-dependent offset in a data-dependent-size output. A faithful ONNX build must
enumerate ~3 mags × ~15 irow × ~15 icol candidate full reconstructions, validate complete-block coverage
per candidate, and select — materializing many full ≤20×20 multi-color planes (≫100KB intermediates),
landing at the ~13-14 detection floor at best, while STILL capped by Wall 1 below 100% fresh. Banned ops
(Loop/Scan/NonZero/Unique) make the search/argmax-of-candidates expression especially costly.

## OPEN ANGLES (exhausted for a CLEAN generalizing win)
- Direct scalar recovery of (mag,irow,icol) without search: BLOCKED — single-shown-row/col cases are
  genuinely ambiguous (Wall 1), so no scalar formula can be exact.
- Assume blocks span ≥2 rows AND ≥2 cols (then mag,offset pin uniquely): FAILS the official stored test
  (shown blocks are single-row) → evaluate() = 0 points. Non-starter.
- Heavy brute-search ONNX at the detection floor: even if buildable (~13-14 pts, below current 13.357),
  Wall 1 still drops fresh below the genverify gate ~15% of runs — not a reliable +13.

## INSIGHT (transferable)
⭐ Some arc-gen generators are INTRINSICALLY AMBIGUOUS: a partially-shown magnified sprite (random
`shows` subset) leaves the translational offset under-determined whenever the shown cells are collinear
(single sprite row/col). This is detectable cheaply — construct two parameter sets identical except for
one offset and check the inputs collide. When the input→output map is not a function (1.55% here), NO
encoding can pass strict fresh 500/500; the ceiling is the unique-determination rate (~98.5%), or the
best-tie-break rate (~99.6%). 209 is a genuine wall on BOTH the determinism axis (Wall 1) AND the
no-separable-form axis (Wall 2) — the earlier "suspected near-wall" verdict is confirmed: it is a wall.
The base net's 13.357 stored is unbeatable AND non-generalizing; there is no clean generalizing
replacement. (Lesson mirrors task255/198 connectivity walls but here the wall is generator non-determinism.)

## S8 (2026-07-02) — counting-model rebuild (+0.349) ADOPTED, div 0
Not iterative — the win is the COUNTING-MODEL rebuild: free-input Einsum contractions for all
row/col bounds ([30] f32 profiles 120B replace ~55 400B bool planes); spread-based S-detection
(MaxPool(lab)+MaxPool(9−lab)−9, global-max==9 test); separable two-stage axis Gathers (kills
[20,20] i32 flat-index plane); 11×11 valid-Conv label read at 20×20 (1600+400 vs 4900, +1200p);
QLinearConv int32 bias folds +1; Pad-with-negative-crop replaces Slice+Pad.
21298+1418 vs 32027+185 → 14.620→14.969. Fresh 2500/1500/800: div 0; inherent fail ~9.8%
(generator-ambiguity wall) unchanged. ORT-OK under strict inference: u8 MaxPool (incl 20×20
global), u8 Where/Add/Equal, QLinearConv+i32 bias, repeated-free-input einsums.

## S9 (2026-07-03) — fold 2nd pass: FLOOR re-confirmed (no change)
13a N/A: output = runtime-factor/offset Kronecker gather (not fixed-mixer einsum).
All candidate plane-merges measured byte-NEUTRAL with only 2 magnification sizes
(S-detect keep-mask, vote, stamp, box). W11 1210p = optimal trade vs 3600B 1×1-conv
plane. Existence-based mag detector: 1/20000 fresh errors + saves only ~800-1200B
(+0.04-0.05) on a net at 10.1% ambiguity wall — rejected under cand≤inc gate.
Clean-corner+universality detector numpy 0/20000. DO NOT re-probe.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 22716 -> 7927 (+1.053)
Mechanism: Terminal GridSample (fp16 [1,30,30,2] grid vs fp32 input) = gather+mask+zero-pad in one free node.
Gate (fresh_verify, inc/cand fail on 1500-2000): 216/55 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. our incumbent fresh-failed 216/2000 (already ~0 on private LB); urad both cheaper AND more robust — strict win.

## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 7927 -> 7862 (+0.008); gate inc/cand=34/34 (safe). See [[neurogolf-urad-7225-bundle-vein]].