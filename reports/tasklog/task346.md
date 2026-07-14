# task346 — d9fac9be

**Rule:** A <=12x12 grid (top-left) is sprinkled with 0.2-density random pixels in two random
colours c0,c1. One special 3x3 block is stamped: the 8 ring cells are all colour c1, the centre
cell is colour c0. The output is a 1x1 grid holding `center` = c0 — i.e. the output tensor is 1.0
only at [0, c0, 0, 0], zero everywhere else. Task = output the CENTRE colour of the 3x3 mono block.
**Current:** 15.88 pts, gen:thbdh6332, mem 9069, params 23 (ledger class BAIL?/random_pixels,
retriaged FEASIBLE A ~17.0pt).
**Target tier:** detection (2-D 8-neighbour test) — a single full-res index plane is unavoidable so
the true ceiling is ~16.3, NOT the optimistic ~17.0 the retriage guessed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel idea / full f32 plane + f32 squares | det | 9990 | 45 | 15.79 | — | works, heavy |
| 2 | slice-input-first (10ch window) | det | 13880 | 47 | 15.46 | — | worse ([1,10,12,12]=5760) |
| 3 | conv-on-free-input -> V30, slice-then-cast f16, f32 squares | det | 9416 | 47 | 15.84 | — | ok |
| 4 | f16 squares (mono = 8*S2==S1^2) | det | 7688 | 46 | 16.05 | 2000/2000 | win |
| 5 | drop Cnt/O conv, replace w/ mono AND S1>0; Where colour-read; WORK=11 | det | 6197 | 46 | **16.26** | 5000/5000 | **best** |
| 6 | drop nz (S1>0) gate | det | 5955 | 46 | 0.0 | 3315/5000 | BROKE — isolated noise pixel (8 empty nbrs) is "mono", Where reads its own colour |

## Best achieved
**16.26** @ mem 6197 params 46 — adopted? N (orchestrator gates). Beats prior 15.88 by **+0.38**
(>+0.3 ⇒ a real win). Does NOT reach P=17.0 (see floor analysis). Generalizes: 5000/5000 fresh +
all hand-coded validate() train+test pass.

## Method (exact, no flood-fill)
V = sum_k k*input_k (1x1 conv on the FREE full input) → slice to 11x11 active window → f16.
8-RING conv (3x3 ones, centre 0): S1 = Σ neighbour values, S2 = Σ neighbour squares.
Centre ⇔ the 8 neighbours are all equal & non-empty ⇔ `8*S2 == S1^2` (Cauchy-Schwarz equality)
AND `S1 > 0`. At a true centre S1=8c1, both sides = 64·c1² (multiple of 64 ⇒ f16-exact ≤5184);
any non-equal 8-set has gap a·b·(c0−c1)² ≥ 7 ≫ the f16 step (4) at that magnitude ⇒ f16 Equal exact.
centre colour = ReduceMax(Where(centre_mask, V, 0)); output = Equal(colour, arange) padded to (0,0).

## Irreducible-floor analysis
`V30` [1,1,30,30] f32 = **3600 B** = 58% of the 6197. Any 2-D 8-neighbour test needs ONE
full-resolution plane. The channel reduction (sum_k k·input_k) can only land as f32 [1,1,30,30]
(Conv/ReduceMax output follows the f32 input dtype; there is no opset-10/11 op that emits f16/uint8
from f32 input except Cast-the-whole-input = 18000 B; pre-slicing the input keeps the 10-ch axis =
[1,10,11,11]=4840 B > 3600). So 3600 is the hard floor and the ceiling is ~25−ln(3600+~1500) ≈ 16.4.
The retriage "~17.0 / tier A" was optimistic — the output being a single cell is NOT a separable
row⊗col rule; the rule is an inherently 2-D 8-neighbour monochrome-ring detection.

## OPEN ANGLES (re-attack backlog)
- Eliminate V30: is there a channel-reduction that lands a 30x30 plane in <3600 B? (uint8/bool plane
  would need an op emitting non-f32 from f32 input — none found in opset 11). If one exists, ~16.4→~17+.
- Vf (484 f32 slice) is the f32→f16 bridge; only removable if V30 itself were f16 (it can't be).
- Remaining 11x11 f16 tensors (~1700 B) are near-minimal; no obvious further fold.

## INSIGHT (transferable)
- ⭐ A "single-cell output = the colour of <some special pixel>" task is detection-class, NOT tier A:
  the output cell count is irrelevant; what matters is whether locating that pixel is separable. An
  8-neighbour monochrome-ring centre is inherently 2-D ⇒ needs one full-res plane (3600 B f32 floor).
- "8 neighbours all equal" = `8·S2 == S1²` (Cauchy-Schwarz) via TWO 8-ring convs (S1 on V, S2 on V²),
  no per-channel expansion, no MaxPool-include-centre problem. f16-exact because the true-centre value
  is a multiple of 64 and the non-equal gap (≥7) dwarfs the f16 step (4) at magnitude ≤5184.
- The `S1>0` (ring-non-empty) gate is LOAD-BEARING, not cosmetic: an isolated noise pixel has 8 EMPTY
  neighbours which trivially satisfy "all 8 equal (==0)"; without the gate `Where(mono,V,0)` reads that
  pixel's OWN colour and ReduceMax can pick it over the real centre. (Verified: dropping it → 0 pts.)

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1224/17.87; CONST-OUTPUT false—output=centre colour of mono 3x3 ring (per-cell); fewer-pixel shortcut only 262/266. int8 QLinearConv ring test already minimal. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).
