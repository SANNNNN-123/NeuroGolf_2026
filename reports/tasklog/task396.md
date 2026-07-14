# task396 (ARC-AGI fcb5c309) — crop largest hollow box, recolour to static colour

WIN 2026-06-30: **15.551194 → 15.632656** (mem **12557 → 11567**, params **136 → 133**).
Leaner box-colour (L=J+1) detection on the GridSample live graph: the live graph spent
D(MaxPool 540B)+E(Slice 450B)+I(Equal 225B) = ~1215B as a black-exclusion guard
(`I = Equal(E,H)`, E = horizontal MaxPool-over-4 of the colour plane) feeding `J = And(I,G)`.
Replaced that guard with a single `Greater(H, 1.0)` nonzero-colour mask (+225B), keeping the
4×4 corner conv `F` and the 15×15 colour slice `H` untouched. Net −990B mem; dropped now-unused
slice inits e,f,g (−3 params). Everything from `M = Equal(C,L)` onward is byte-identical.
Robustness: the 4×4 corner stencil (7 cells of an L-shape uniform) is essentially impossible
for the 5%-sparse static colour to forge, so excluding only black (value>1) is as robust as the
incumbent's E==H guard. Gate: bundled 266/266 fail=0; fresh arc-gen **9000/9000** fail=0 AND
**0 divergence vs incumbent** (genuine equivalence, not a re-fit). Note: pure count-based J
(ReduceSum→ArgMax) was REJECTED — it disagrees with the generator in ~2/4000 static-dense cases
(static out-counts box borders), which the corner detector handles correctly.

Prior: installed/source-owned exact = 15.551194 (mem 12557, params 136).
2026-06-28: `src/custom/task396.py` was re-synced to the installed live graph.

Earlier P (ext:kojimar7113) = 14.72 → **15.29** (mem 16323, params 149), fresh
300/300.  That superseded an older k=2..7 run-conv source, but is itself now
superseded by the installed GridSample-style live graph.

## Rule
2-3 hollow rectangular boxes (1px outline colour c0, black interior) + scattered single-pixel
static (colour c1, some inside boxes). `wides`/`talls` sorted DESC ⇒ box 0 is the LARGEST
(max width AND max height). Output = tall0×wide0 crop of box 0 with every NON-BLACK cell
(outline + interior static) painted c1, black interior stays black.

## Encoding (single-direction run-length, plane-eliminated)
- colf30 = Σk·input_k (1×1 Conv, fp32 entry 3600B), Slice→18×18, Cast→fp16 colf.
- HORIZONTAL same-colour adjacency pairs eqh = (colf[:, :-1]==colf[:, 1:]) & colf>0 (uint8).
- **Run-length-ending-here via CUMSUM-RESET** (replaces the old k=2..7 conv army): pad a leading
  zero, cs=CumSum(eq, axis); reset=where(eq, -BIG, cs); rl = cs − prefixmax(reset) where
  prefixmax = one-sided full-length MaxPool (ZERO params). maxH = ReduceMax(rl) = wide0−1.
  CumSum needs fp32 (rejects fp16/uint8/int8; int32 same size) → pay ONE fp32 cumsum + one
  fp32 cast-up, everything else fp16.
- Position from the horizontal map only: bcol0 = (min col with per-col max-run==maxH) − maxH;
  brow0 = min row with per-row max-run==maxH (top edge). Both reduced to [1,1,1,18]/[1,1,18,1]
  BEFORE masking (no full-plane Where).
- **tall0 by a 1-D probe** down box-0's left-edge column (Gather col bcol0 from colf →
  [1,1,18,1]): tall0 = (first row ≥ brow0 where colvec != c0) − brow0. ⚠️ box may reach the
  grid bottom edge ⇒ the no-stop fallback MUST be A(=18), not BIG, else tall0 overshoots by 1.
  This **kills the entire vertical cumsum machinery (~5KB, 9 planes)** — box 0 has BOTH max
  width and max height (same sorted index 0), so one direction + a column probe suffices.
- c0 = colf at (brow0,bcol0); c1 = present non-bg colour ≠ c0 (ArgMax over masked chramp).
- Gather-shift colf to (brow0,bcol0), crop WORK×WORK, paint non-black→c1, **uint8 sentinel-99
  Pad to 30×30** (opset-11 Pad+Equal accept uint8 → 900B not fp16 1800B), Equal(L30, chan u8)
  → FREE BOOL one-hot output.

## Dominant intermediates (irreducible)
Conv entry 3600B (input is fp32 ⇒ Conv can't keep fp16); cumsum cast-up+cumsum 2×1296B
(fp32-only op); colff slice 1296B (transient fp32 18×18 before fp16 cast); output Pad 900B.

## 2026-06-28 high-score frontier check

Not a 20+ candidate by simple local-stencil collapse.  The rule is a
data-dependent crop: top-left, width, height, outline colour, and static colour
must be inferred from the scene.  The installed graph's GridSample-style path is
already compact for this kind of remap.  Since params are already 136, a 20+
model would have only about 12 bytes of memory budget left, which is unrealistic
without a new no-intermediate dynamic-crop primitive.

## Levers used / transferable
- ⭐ Single-axis run-length suffices when the target object maximises BOTH axes (sorted DESC):
  detect on one axis, recover the other dimension by a tiny 1-D edge-column run probe.
- ⭐ CUMSUM-RESET run-length (1 fp32 cumsum + one-sided MaxPool, 0 params) replaces a k-value
  conv army for max-contiguous-run; conv-sum overcounts across gaps, cumsum-reset does not.
- uint8 sentinel-99 Pad+Equal for the output one-hot (opset-11) → 900B vs fp16 1800B.
- Reduce a full-plane argmin to a per-row/col profile BEFORE the masking Where.
- Pitfall: a "first stop below" probe must fall back to the canvas EDGE, not a huge sentinel,
  when the object touches the grid boundary (+1 overshoot bug, caught at 263/266).

## S9 (2026-07-03) — single-tap valid-Conv in-op crop (+0.147) ADOPTED
Playbook #7: replaced 1×1 Conv 30×30 read (A 3600B fp32) + full-canvas Cast (Au 900B)
+ Slice with ONE 13×13 valid Conv (only tap at (0,0), out 18×18=1296B) — grids ≤18×18
per generator. mem 11046→7842 (−3204), params 135→1813 (+1678), total 11181→9655.
Gates: stored fail=0; bit-identical 0/3000 fresh + 0/400 random (orchestrator re-check);
uncached fresh 2000: inc 0 / cand 0 / div 0. Latency 0.345ms.
Floors: 1690-param 13×13×10 kernel irreducible (two-conv splits re-add ≥1444B plane;
sparse_initializer blocked by sanitize_model rename gap). Backup reports/retired_networks/
task396_pre_s9.onnx.
