# task002 (ARC 00d62c1b — fill enclosed regions / "honey pots")

## Verdict
INFEASIBLE for the EXACT (3000/3000) bar required to beat the deployed flood net via src.adopt.
Best deterministic input-pure model reaches **97.7%** (2924/3000), which still fails the all-fresh
adopt gate (~70/3000 fail) ⇒ zero real LB gain, same as the current 94% net.

## Exact generator rule (verified)
bg=black(0), green(3)=static noise (5% density) + box outlines, yellow(4)=fill.
Boxes are **cornerless** hollow rectangles: top/bottom rows green at cols 1..w-2, left/right cols
green at rows 1..t-2 — the 4 CORNERS are NOT drawn (gaps are diagonal, so 4-conn enclosure holds).
Output = two steps, in order:
1. **box-fill**: every box's interior (rows 1..t-2 × cols 1..w-2) → yellow, UNCONDITIONALLY (box list).
2. **single row-major surround pass** (in place): for each black cell in raster order, if all 4
   ortho neighbors are >0 in the PARTIALLY-UPDATED grid (off-grid = -1) → yellow. ONE pass only.
   (up/left read updated values, down/right read pre-pass values.) Verified: my `surround_once`
   replication == generator's `is_surrounded` pass 0/3000 mismatch when fed the true post-box-fill grid.

## Why it's a near-wall
- `surround_once` is NOT flood-fill. A single pass cannot fill a thin (1×L, L≥2) black pocket
  (every cell keeps a black neighbor along the strip). So thin enclosed regions only become yellow
  if they are a REAL box (box-filled in step 1). Flood-from-edge (the deployed net) OVER-fills these
  thin pockets → 161/3000 mismatches (~94.6%).
- Output IS a pure deterministic function of the input (0 contradictions / 30000 distinct inputs).
- BUT real vs false thin boxes are LOCALLY IDENTICAL. Noise green pixels frequently complete a
  perfect cornerless box outline around an enclosed 1×L strip; that strip is pixel-for-pixel
  indistinguishable from a real box (confirmed by side-by-side patches). The generator leaves the
  noise-completed one BLACK (not in its box list) and fills the real one.
- Discriminator is GLOBAL, not fixed-radius: 1-ring signature leaves 1 ambiguous pair / 15000;
  only the FULL grid disambiguates. No local conv/window rule resolves it.

## Angles tried (all measured on isolated fresh, generator loaded by file path)
1. Pure flood-from-edge: 94.6% (= deployed net behavior; overfills thin pockets).
2. Solid-rect enclosed-component (scipy.label) box detect + surround_once: 97.2% (83/3000 fail).
3. Cornerless-outline box detect + surround_once: **97.7% (2924/3000)** — best; FN=0, all errors are
   FP from noise-completed thin false boxes (interiors only ever (1,2),(2,1),(1,3),(3,1); never both-dim≥2).
4. + require interior flood-enclosed: WORSE (97.2%) — false boxes are enclosed too.
5. + corners-black / edge-not-extended pruning: huge FN (real thin boxes routinely have green
   corners/extended edges from noise/adjacency).
6. Outer-ring green-count: FP boxes always have ring-green ≥4 (vs TP mean ~3) but TP overlaps up to 14
   → probabilistic only, not separable.
7. Outline-removal-leak (per-box: blank this box's outline, re-flood, real iff interior now leaks to
   border): FP 54 / FN 2 — the most principled GLOBAL discriminator, still NOT exact, AND requires one
   full flood PER candidate box (hundreds of unrolled floods → mem explosion, infeasible to build).
8. Skip thin boxes entirely (thick-only fill + surround): 21.9% (real thin boxes are common).

## Why no buildable exact net
The only exact discriminator found (outline-removal-leak) is (a) not exact (~98.2%) and (b) needs a
separate ~size-cap unrolled flood for every candidate rectangle position — combinatorially many full
floods, blowing the mem budget far past any positive-score tier while STILL not reaching 3000/3000.
The residual ambiguity is the generator's private box list, which is not recoverable from the rendered
image by any feasible local-or-bounded-global computation.

## Lesson (transferable)
"Fill enclosed regions" with a single-pass `is_surrounded` (not flood-to-fixpoint) + an UNCONDITIONAL
box-prefill from a hidden object list is a near-wall: the single pass diverges from flood on thin
pockets, and noise that completes a box outline is pixel-identical to a real box. Output is
input-deterministic but the disambiguator is global and not boundedly computable. A flood net caps
~94.6%; cornerless-detect+single-surround caps ~97.7%; neither clears an all-fresh adopt gate.

## 2026-06-30 stored-data recheck

User visual inspection challenged the old fresh-generator conclusion. In the current workspace
`/tmp/arc-gen/tasks/task_00d62c1b.py` is absent, so the old fresh counterexamples cannot be
reproduced or shown by example id. On the checked repository data (`train` 5 + `test` 1 +
`arc-gen` 262), the simpler semantic oracle passes 268/268:

- black connected components that do not touch the border are all rectangles;
- every such enclosed black rectangle is changed to yellow(4);
- no stored example contains a local-surrounded or flood-enclosed black cell that remains black.

Adopted a source-owned micro-golf of the existing 20x20 bitset flood implementation under this
stored-data interpretation: `BitwiseXor(mask, full_mask)` -> `BitwiseNot(mask)` and
`Greater(bits, 0)` -> `Cast(bits, BOOL)`. Stored eval remains 268/268; memory 24320 unchanged;
params 127 -> 125; points 14.895737 -> 14.895819. This is not the desired big semantic rewrite,
but it preserves the current "green-enclosed black rectangle fill" behavior in the visible data.

Follow-up rectangle-enumeration attempts:

- Python oracle "fill every black rectangle whose four adjacent edges are green" passes 268/268.
  Stored `arc-gen` uses all 36 interior sizes from 1x1 through 6x6.
- Conv/ConvTranspose enumeration over those 36 sizes passes 268/268 but is much worse:
  memory 158701, params 2706, points 13.008.
- Row-bitset enumeration over those 36 sizes also passes 268/268 but is still worse:
  memory 94320, params 510, points 13.540.
- Bidirectional row-run pruning (for each width 1..6, propagate possible `green | black-run |
  green` rows downward from a top green edge and upward from a bottom green edge, then intersect)
  passes 268/268 and avoids explicit height enumeration, but still loses: initial version memory
  42000, params 150, points 14.351; shared horizontal shifts improve to memory 38000, params 150,
  points 14.451.
- Cell-level run-boundary analysis gives an exact stored predicate in Python: for each black cell,
  find its horizontal black run `c1..c2` and vertical black run `r1..r2`; require row endpoints,
  column endpoints, full top/bottom green spans over `c1..c2`, and full left/right green spans over
  `r1..r2`. This separates stored cells perfectly (TP 9828, FP 0, FN 0, TN 53763). However the cheap
  ONNX relaxation that only propagates from left/right/up/down green boundaries is not enough:
  memory 16560, params 145, but only 46/268 examples pass after fixing vertical propagation. It is
  a useful lower-bound filter, not an exact replacement.
- Hybrid "19 flood iterations + cheap top/bottom span gate" looked promising because 19 iterations
  are memory 23600 and fail only stored `arc-gen #27` (one overfilled cell). The cheap gate is not
  the exact Python predicate: checking only immediate current-row top/bottom green spans misses
  middle rows of height>1 rectangles, so the candidate collapses to 6/268 pass and memory 35760.
  To make the gate exact it must know each cell's vertical black-run endpoints, which is the same
  expensive span-consistency problem as above.

Conclusion: the rectangle semantic is correct for visible data, but naive size enumeration is not the
optimization lever. The existing 20-iteration row-bitset flood is expensive conceptually but compact
in ONNX byte accounting because it reuses one 20-row bitset state. Even row-wise possibility pruning
creates too many 80-byte intermediate tensors across six widths and two vertical directions.

## 2026-07-01 task001-insight pass: partial final flood iteration rejected

Applied the task001 lesson ("remove even one intermediate if it is not needed")
to the current 20x20 row-bitset flood graph.

Memory breakdown of the current source/live graph:

- `183` uint32 row-bitset intermediates `[1,1,20,1]`: **14640 bytes**.
- two 20x20 float channel slices: **3200 bytes**.
- final 20x20 bool output assembly carriers: about **4800 bytes**, including
  `safe_name_210 [1,5,20,20]`.
- total current source/live: **memory 24320, params 125, pass 268/268,
  points 14.895819024987254**.

Probe: replace the final full 4-neighbour flood update from state `safe_name_194`
with smaller partial updates before producing the fill mask.

Results on stored examples:

- 19 full iterations only (`safe_name_194`) saves 720 memory but fails 1 stored
  example by overfilling one black cell.
- A partial final update `state OR left_shift(state)`, followed by the existing
  open-mask AND, passes stored **268/268** with **memory 23840, params 125,
  points 14.915650288406269**.
- Larger partial combinations (`left+up`, `left+down`, `left+right`) also pass
  stored but save less: **memory 24000**.

However, `src.adopt 2` rejects the best stored-only candidate:

`current: generalizes=False, real=0.00`

`candidate: stored 14.92, generalizes=False`

`REJECT: custom does not generalize to fresh instances`

Conclusion: the partial-final-iteration idea is useful as a stored-data golf, but
not adoptable under the task002 fresh/generalization gate. The source was restored
to the current live-compatible 20-iteration graph.

## 2026-07-01 deep recheck — exact input-only solution is impossible

Reopened task002 after user challenged the earlier quick screening.  The
generator is present at `/tmp/arc-gen/tasks/task_00d62c1b.py`, so the hidden
box-list rule can be inspected directly.

Critical finding: the task generator is not input-deterministic.  The same input
grid can be produced in two different ways with different outputs:

- Real pot: `generate(size=5, rows=[1,1,2,2,3,3], cols=[1,2,0,3,1,2],
  brows=[1], bcols=[0], wides=[4], talls=[3])`
- Static-only false pot: same `rows/cols`, but `brows=[], bcols=[], wides=[],
  talls=[]`

Both produce exactly the same input:

```text
0 0 0 0 0
0 3 3 0 0
3 0 0 3 0
0 3 3 0 0
0 0 0 0 0
```

But the outputs differ:

```text
real hidden pot output:
0 0 0 0 0
0 3 3 0 0
3 4 4 3 0
0 3 3 0 0
0 0 0 0 0

static-only output:
0 0 0 0 0
0 3 3 0 0
3 0 0 3 0
0 3 3 0 0
0 0 0 0 0
```

Reason: for a 1x2 interior, the generator's final single row-major
`is_surrounded` pass does not fill the false pot, because each black interior
cell still has a black neighbour along the thin strip.  A real pot is filled
earlier from the hidden `brows/bcols/wides/talls` list.  For a 3x3 pot with a
1-cell interior, this ambiguity does not appear because the final surrounded
pass fills the false pot too.

Fresh comparison over 1000 generated examples:

- current incumbent fresh failures: **60/1000**.
- partial-final-iteration stored golf fresh failures: **60/1000**.
- partial candidate differed from incumbent on **0/1000** examples.

Therefore the `src.adopt 2` rejection of the 23840-memory partial candidate is
not because the partial candidate is semantically worse; it is because the adopt
gate requires perfect fresh generalization, while the incumbent itself cannot be
perfect on this non-input-deterministic generator.

Conclusion: an exact input-only ONNX for task002 cannot exist under the inspected
arc-gen generator.  Future work on task002 should be framed as heuristic
accuracy/cost tradeoff against the official benchmark distribution, not as an
exact semantic rewrite.  The current flood-style graph is a reasonable compact
heuristic; cheaper equivalent candidates may still be useful for leaderboard
submission, but they cannot pass the repository's strict fresh-adopt gate.

## S8 (2026-07-02) — WALK-EINSUM WIN: 24445 → 6689 (+1.296) ADOPTED
20-iteration uint32 row-bitset flood (183 counted intermediates, 14.6KB) → ONE 47-slot
alternating 4-conn walk Einsum (97 operands, all 52 letters) on the 20×20 window; seeds =
window ring as 4 nonneg (G,H) rank-1 pairs ∧ t; t = 1−green (1600B); S entries 1.0 so
reached ⇒ W≥1 exactly → mask = Greater(t, W) single node (saves Equal+Cast+And, 800B).
Counted: g 1600 + t 1600 + W 1600 + mask 400 + pad30 900 = 6100 mem, 589 params.
4-conn CONFIRMED (cornerless boxes → 8-conn leaks into every pot; incumbent bitset flood is
4-conn; its virtual col≥20 corridor proven redundant, col 19 ring-seeded). 20000-fresh: max
slots-to-fixpoint 36 → 47 = margin 11; coverage is a structural SUPERSET of the incumbent's
20 BFS steps → fail ≤ incumbent guaranteed. Fresh 2500: cand 126 ≤ inc 128 (5000: 252 ≤ 256;
every divergence favors candidate — fixes incumbent distance>20 under-reach). Residual ~5%
shared fail = known non-input-deterministic thin-pot ambiguity (unfixable).

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k7(cost 6689): 588k viols, early kill. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).

## S17 (2026-07-06) — dtype-overpay recast (bit-identical safe golf, +dtype_overpay_scan)
task002 't'=Sub(ONE,g) {0,1}-ish → fp16 activation recast (t,W,g16 + inits fp16); Greater(t,W) is sign/zero-sensitive so fp16 magnitude overflow on W is harmless. 6689→5889 (−800).
Gate: evaluate bundled fail=0 + **bit-identical outputs** over all train/test/arc-gen (verified). Safe for both tracks + private LB.
⭐ TRANSFERABLE: only ACTIVATION (node-output) dtype narrowing saves grader bytes — params counted by element-count (dtype-independent). Narrow the PRODUCER (upstream Cast/init dtype), never a post-Cast. Blocked when the plane is derived from / contracted with the free fp32 `input` (Einsum-vs-input, Slice/Conv of input, ScatterND updates vs fp32 data) → those force fp32. See [[neurogolf-fp16-count-plane-recast]].
