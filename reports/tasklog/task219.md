# task219 — 90f3ed37

**Rule:** Fixed 15×10 grid. The instance defines ONE shared "legend" of three sub-patterns — A (tall×awide), B
(tall×bwide), C (tall×cwide) — all monochrome cyan, with tall∈{1,2,3}, awide/bwide/cwide∈{1,2}. The grid holds
2–6 horizontal "bands". Each band sits at a random top `row` (consecutive band tops spaced `randint(tall+1,tall+3)`
apart) and a random B-column `col∈{awide,2·awide}`. Per band the input draws: A tiled across cols `[0,col)` (period
awide), B at `[col, col+bwide)`, and — ONLY for the FIRST (top) band — C tiled across `[col+bwide, 10)` (period
cwide), all cyan. The OUTPUT is the input PLUS, for every band AFTER the first, the C-pattern tiled into that band's
own C-region `[col+bwide,10)` drawn in BLUE (color 1). Every changed cell is input-0 → output-blue (verified
3000/3000). So: copy input, and "replay" band-0's C-pattern, recolored blue, into each later band's empty right side.

**Current:** 14.605 pts, ext:kojimar7113, mem 18574, params 14113 — but scores only **49/500 ≈ 9.8% on isolated fresh**
(re-measured; matches handoff's 6–12%). Stored points are a non-generalizing memorization.
**Target tier:** GAP-CLOSER (any generalizing net beating ~10% adds ~its full score to REAL LB).

## Attempts (all numpy-oracle / recovery experiments; no ONNX written — see verdict)
| # | angle | result |
|---|---|---|
| 1 | closed-form per-cell | NO — output cell depends on band-0's far-away C-region (non-local copy) |
| 2 | C-pattern shift hypothesis (read band0 C, tile into later bands by Δcol) | EXACT 3000/3000 **with oracle params** |
| 3 | recover params from input: A-anchor alignment (first cyan row in cols[0,awide)) | anchor=top+min(arows) consistent 2000/2000 — robust vertical align without needing absolute top |
| 4 | recover cstart_i for later bands = 1+rightmost-cyan-in-band | EXACT 0 mismatch |
| 5 | recover band-0 anchor cstart0 (col0+bwide, col0∈{awide,2awide}) | **AMBIGUOUS** — B-pattern match leaves 2 candidates ~27%; A∪B reconstruction match leaves ~21% ambiguous |
| 6 | full global-reconstruction brute force over (tall,awide,bwide,cwide,cstart0) | ~72% fresh (band-detection edge cases + multi-accept votes; reachable ceiling is 99.95%, not 100%) |

## Irreducible-floor analysis — TWO independent walls

**WALL 1 — INFORMATION-THEORETIC (caps even a perfect oracle below the fresh bar).**
The C-pattern exists ONLY in band 0. Band 0's internal A|B|C column segmentation is NOT always determined by the
input: `col0∈{awide,2awide}` is an independent random draw, and when the A-pattern, B-pattern and C-pattern tiles
coincide at the A/B/C boundary, two different `cstart0` values reconstruct the SAME band-0 input but imply DIFFERENT
blue outputs. Direct collision scan: **same input → different output in ~1 case per 2174** (oracle ceiling
**99.954%**, measured over 100 000 fresh instances; 43 ambiguous inputs / 79 308 unique). The choice of cstart0
flips the output in 44% of cases, so it is load-bearing, not cosmetic. A perfect net therefore leaks ~1.4 per 3000.
The session's stated bar is ISOLATED fresh ≥3000/3000 (a prior 1/3000 leak caused an LB regression). 219 cannot
meet 3000/3000 — the residual is the theoretical floor, not a fixable bug.

**WALL 2 — ONNX EXPRESSIBILITY (the deeper blocker).**
Even accepting the 99.95% ceiling, the rule needs: detect a DATA-DEPENDENT NUMBER of bands (2–6) at data-dependent
rows; per band derive an A-anchor and a cstart (= 1 + rightmost cyan, a per-object reduction); identify band 0 as
the furthest-right band; disambiguate cstart0 via a 2-candidate GLOBAL self-reconstruction; then COPY band-0's
C-region tile into each later band with a per-band (Δrow, Δcol) shift, recolored. This is variable-count
multi-object correspondence with a data-dependent source→target copy — exactly the class the playbook lists as a
WALL (no Loop/Scan/NonZero; the band count and both source and target positions are all data-dependent, so the
copy cannot be unrolled into a fixed DAG the way task48's bounded flood was). The master-key bounded-iteration
unrolling does not apply: there is no bounded local propagation: the operation is a long-range gather of one band's
content into K others.

## Best achieved
No net written. Best understood-rule numpy recovery ≈72% (band-detection edge cases unsolved); even a flawless
recovery is capped at 99.954% by WALL 1. Not adoptable at the required 3000/3000.

## OPEN ANGLES (genuinely tried and rejected — left for the record)
- Per-cell closed form: rejected (non-local copy).
- Absolute-column C reading (skip cstart0): FAILS — cstart_i ≢ cstart0 (mod cwide) in ~11% (phase differs).
- Both-cstart0-candidates-agree shortcut: only 56% agree, so cannot dodge the disambiguation.
- A-anchor alignment + global reconstruction in numpy CAN approach 99.95% with more edge-case engineering, but
  (a) still below 3000/3000 by WALL 1, and (b) does NOT translate to ONNX by WALL 2.

## INFEASIBLE VERDICT
INFEASIBLE for a 3000/3000-generalizing ONNX net, on TWO independent grounds:
(1) information-theoretic: oracle ceiling 99.954% (~1 ambiguous input per 2174) — the input does not always
determine the output, so no model can reach the required exactness; and
(2) expressibility: variable-count (2–6) band correspondence with a data-dependent source→target region copy is a
no-Loop ONNX wall (long-range gather, not bounded local propagation — master key does not apply).
The handoff's "info-bottleneck / connectivity wall" label is CONFIRMED and now quantified: the bottleneck is band-0's
internally-ambiguous A|B|C segmentation, the sole carrier of the C-pattern.

## 2026-06-29 public/source parity recheck

Read-only parallel analysis rechecked the current live/source/public state.

- Current manifest/inventory entry is the URAD teacher overlay: `points=15.162385`, `memory=18633`, `params=92`.
- Public candidates under `boristown`, `lucifer`, `biohack_mix`, and `urad` have no useful structural delta from live/source: same computation, op histogram, initializers, and attributes aside from serialization/naming details.
- The older wall conclusion still stands. Public artifacts do not provide a new mechanism, and the task remains a poor source-owned rewrite target because the needed variable-count long-range band copy is not cleanly expressible in the available no-loop ONNX subset.

## INSIGHT (transferable) ⭐
- ⭐ When a "deterministic-looking" generator hides ONE template in a single object and that object's internal
  segmentation can coincide, RUN THE COLLISION SCAN (`dict[input.tobytes()] → set(output.tobytes())` over 50k–100k
  fresh) BEFORE building. A nonzero collision rate is a hard oracle ceiling; if it sits near the 1/3000 fresh bar,
  the task is INFEASIBLE regardless of recovery cleverness. (Here ~1/2174.)
- ⭐ A-ANCHOR ALIGNMENT for multi-band tasks: align repeated objects by the first cyan row of a shared sub-region
  (here the always-present A-region) — gives a per-object reference that is consistent across objects (= top +
  min(pattern-rows)) WITHOUT needing the true top, which can be empty. Useful for any "repeat this object N times"
  recovery. But it does NOT make the cross-object COPY ONNX-expressible when N and positions are data-dependent.

## S8 (2026-07-02) — batched-band + placement einsum (+0.922) ADOPTED; "18k floor" REFUTED
5 copy-pasted per-band blocks (~2KB each) → ONE K=6-batched block (also fixes the missing 7th
band); placement+accumulation → ONE fp16 einsum 'kjr,ks,jsc,k->rc' (placement one-hots ×
shift selectors × shifted patterns × exists flags → [15,10] mask, 7.8KB→1.4KB); shift variants
via init index-table Gather (params not memory); epilogue Max(8·cyan, mask) + Pad-255 + Equal.
7078+132 vs 18033+93 → 15.195→16.117. Fresh 20000: cand-only-fail 0, inc-only 13 (candidate
strictly ⊆); fresh_verify 2500 (1042≤1046) + 1500 (632≤633). Incumbent inherent fail ~42%
(public LB = bundled → still pointed). LOAD-BEARING QUIRKS preserved: 4-shift set {+2,+1,0,−1}
(Δ=−2 spuriously wins), k=2 block uses band-0's occ[1] validity (bands can have internal empty
rows). Latency 0.04ms.
