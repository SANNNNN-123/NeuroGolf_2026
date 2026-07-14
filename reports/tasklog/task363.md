# task363 — e5062a87

**Rule:** size=10 grid (top-left of 30×30) holds a static gray(5)/black(0) background
plus one RED(2) reference sprite and several BLACK-carved copies of that same shape at
the OTHER placements. The OUTPUT paints EVERY placement RED. The sprite is a
diagonally-connected conway shape (bbox wide×tall, wide+tall=5 ⇒ ≤4×4). On FRESH
instances the generator legality guarantees the chosen placements are EXACTLY the
positions where the (red-derived) sprite fits onto black cells, so the rule = "correlate
the red template against the black plane; paint every shift where #matched == #red".
**Current:** 13.68 pts, gen:biohack_new, mem 81146, params 1620
**Target tier:** detection/A — closed-form template-match (red plane as Conv kernel, shift = placement offset).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | clean anchored-5×5-Conv match (no disambig) | A | ~5.5k | 72 | (16.4) | 200/200 | FAILS 2 fixed illegal trains → 0 pts |
| 2 | faithful rebuild of incumbent selection, cheap I/O | A | 21346 | 1634 | 14.96 | — | 265/265 fixed |
| 3 | +fg=1−black (drop colour-Conv 30×30 plane) | A | 17046 | 1620 | 15.17 | — | 265/265 |
| 4 | +shrink edge/top2 pads 28²→10² (pads=9) | A | 17046 | 252 | 15.24 | — | 265/265 |
| 5 | +ConvTranspose pads crop 28²→10² (paint/cover) | A | 14110 | 246 | **15.43** | 199/200 | **265/265 fixed, ADOPT** |

## Best achieved
**15.43** @ mem 14110 params 246 — beats prior 13.68 by **+1.75**. fresh 199/200 (≈99.1%,
identical to incumbent which is 1982/2000 fresh).

## Irreducible-floor analysis
The match lives in 19×19 SHIFT-SPACE (placement br,bc ∈[0..9] needs the full ±9 Conv pad;
sweep showed P<9 truncates real placements → fails). ~7 fp16 19×19 planes (722B) +
~14 bool 19×19 planes (361B) carry the correlations + the disambiguation logic. The two
1568B 28×28 ConvTranspose planes were cropped to 10×10 via pads=[9,9,9,9]. The 1800B
30×30 pad-back (uint8 Pad + bool) routes the small mask into the FREE Where output.
The incumbent's 81146 was 18000B (full-input fp16 cast) + 45000B (five [1,10,30,30] bool
output-assembly planes) — both ELIMINATED here by channel-slicing the input and using the
Where-into-free-output idiom.

## OPEN ANGLES
- The ~6500B disambiguation machinery (edge/top2/dense/overlap) exists ONLY to pass the 2
  hand-crafted ILLEGAL validate() trains. It is a positional FALSE-POSITIVE tradeoff: the
  dropped extras (e.g. train1 (1,3)) are STRUCTURALLY IDENTICAL to real legal top-row
  placements (verified: same top2/dense/n8/overlap features), so any rule passing the
  trains MUST drop ~0.9% of legal placements. Dropping the `top_false` branch gives
  2000/2000 fresh but fails train1 → 0 pts. No further reduction without sacrificing one.
- Shift-space could shrink if a data-dependent crop of the 19×19 to the valid bbox were
  free — but valid placements span the full 19×19 (rows 1–17, cols 0–18).

## INSIGHT (transferable)
⭐ A "fits-on-black template-match" task is closed-form: use the RED/template plane DIRECTLY
as the Conv weight (no anchoring needed) — the placement offset IS the correlation shift,
opset Conv does no flip. ConvTranspose `pads=[k,k,k,k]` crops the scatter output IN-OP
(28²→10²), avoiding a Slice plane. ⭐ When an incumbent net is at a low score with huge
mem, the waste is often (a) a full-input fp16 Cast (18000B) and (b) [1,10,30,30] bool
output-assembly planes (9000B each) — replace with channel-Slice + Where-into-free-output
for a big win at IDENTICAL behavior. ⭐ Some "confirmed" nets only generalize ~99% (the
hand-crafted validate() examples can be ILLEGAL w.r.t. the generator invariant, forcing a
positional heuristic that's a false-positive tradeoff); matching that profile at lower mem
is still a strict LB win even if fresh < 200/200.

## 2026-07-01 (S7 re-run) — LANDED delta-route refit
Output = input with painted black-cells→red = input + small delta. Replaced the
6-channel output assembly (out0/out2/zero_plane/active Concat + Pad) and the gray
channel slice/cast with ONE `Where(pad(paint_mask), red_onehot, FREE input)`.
Eliminated active(600B)+gray_f(400B)+gray_u8/black_u8/out0/out2/zero_plane(500B);
added only paint_mask30(900B bool)+paint2_b(100B). Net −500B.
Old: mem 4339, params 293, 16.559. New: mem 3839, params 296, **16.673 (+0.114)**.
Gate: bundled 265/265 fail=0. Side-by-side vs HEAD incumbent on 4000 fresh arc-gen:
old_wrong=12, new_wrong=12, **OLD-correct/NEW-wrong=0** → strict-equivalent refit at
lower mem (both share the 12 baked-in illegal-validate edge failures). networks/
task363.onnx rebuilt; manifest updated. Mechanism = the Einsum/delta-route-onto-FREE-
input lever.
