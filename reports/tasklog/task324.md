# task324 — d07ae81c (diagonals through dots over a striped background)

**Rule:** H×W grid (both 10..20). Background = base colour bg0 everywhere + full
horizontal/vertical STRIPES in bg1 (rows in brows, cols in bcols, first stripe
index ≥2). 2..3 isolated DOTS, recoloured colors[0] if on base / colors[1] if on
stripe (generator guarantees ≥1 of each via len(seen)==2). OUTPUT: through every
dot draw BOTH 45° diagonals (r+c=const OR r−c=const); each diagonal cell becomes
colors[0] over base, colors[1] over stripe; non-diagonal cells unchanged. Dots are
fixed points of this rule.
**Current:** 15.904 pts, ext:kojimar7113 (crowd net), mem 7330, params 1586.
**Target tier:** B (label map + Equal). Genuinely 4-colour multi-step.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior-session draft (full 30×30 planes, 2-pass 21×21 convs) | B | 22216 | 976 | 14.95 | — | planes at 30×30 |
| 2 | dilated-conv crop colf→20×20, 4 diagonal convs | B | 18616 | 1000 | 15.12 | — | trimmed |
| 3 | single 39×39 X-conv (1 plane) | B | 15416 | 1639 | 15.26 | — | fewer planes, +params |
| 4 | fuse label inner=where(stripe,c1,c0); L=where(diag,inner,g) | B | 14216 | 1639 | 15.33 | 200/200 | best, still < 15.90 |
| 5 | fp32 packed col0/col1 recovery | B | 21500 | 1641 | 14.95 | — | fp32 packing blew up — reverted |

## Best achieved
**15.33 @ mem 14216 params 1639 — fresh 200/200, stored 266/266.** CORRECT net but
does NOT beat prior 15.904 (−0.58). Verdict: INFEASIBLE to beat by +0.3.

## Irreducible-floor analysis (why +0.3 is unreachable)
Beating 15.904 needs mem+params ≤ 6597. The diagonal X-conv mechanism alone is
forced: full ±19 reach on a ≤20×20 canvas ⇒ a 39×39 X-kernel = 1521 params (params
count ALL elements incl zeros; a smaller kernel can't reach corner-to-corner
diagonals; ORT has no diagonal pooling/MatMul shortcut; a 2-pass smaller-kernel
decomposition costs MORE planes net). Conv needs float ⇒ seed input plane (800B
fp16) + conv output plane (800B fp16) forced. The output label needs Equal over the
full canvas ⇒ colf entry fp32 1600B (Conv keeps fp32 input dtype) + 30×30 padded
uint8 label 900B. FORCED floor = 1521+800+800+1600+900 = 5621, leaving ~976B (~2
planes) for ALL of bg0/bg1 detect + 2-axis stripe detect + seed mask + 2-colour
recovery — which need ≥6 planes (~3000B+). kojimar 7330 packs these via TopK (4
colours one op) + banded ReduceMax; it is at/near the structural floor. My rebuild
lands 14216 and even maximal fusion only ties kojimar, never −2300B below it.

## OPEN ANGLES
- Param lever is the only hope: replace the 1521-param X-kernel with a ≤150-param
  full-diagonal mechanism — none exists in the banned-Loop op set.
- chrow/chcol consolidation could shave ~400-800B — far short of the ~7600B needed.

## 2026-07-05 — transfer probe follow-up

- Tried the task025-style capacity-shrink angle on the colour TopK (`k_four=4 ->
  3`).  The graph is structurally tied to four slots (`Split` into bg0/bg1/rare0/
  rare1), so the naive shrink is invalid and fails at runtime.  A real K=3
  variant would need a new colour-assignment mechanism, not just a narrower
  TopK initializer.
- Cost recheck still matches the old floor: counted bulk is `input_color_f32`
  1600B, `color_map` 900B, and eleven 20x20 bool/uint8 planes.  The 067/066
  axis-activity gate does not remove these because the expensive planes encode
  dynamic colour identities and diagonal spread, not a pure row/column activity
  predicate.

## INSIGHT (transferable)
⭐ A full-grid 45° diagonal SPREAD (dot → its whole X) is forced to a (2·side−1)²
Conv kernel = O(side²) PARAMS regardless of sparsity (params count zeros) + two
float full planes. On a 20-canvas ~1521 params + 1600B that no banned-Loop op can
undercut ⇒ such tasks have a hard ~5.6KB (mem+params) floor BEFORE any colour logic.
Distinct from task037/119 where the generator BOUNDS diagonal length (≤7) →
collapses to a tiny K×K conv; here length is the full grid so no collapse.
⭐ Reusable entry lever: a DILATED (dil=10) 2×2 Conv whose only nonzero tap is
[.,.,0,0]=arange yields the colour-index plane ALREADY cropped to 20×20
(30−(2−1)·10=20) in ONE op, so every downstream plane counts at 20×20 not 30×30.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-4 opus agent). Do not re-attempt without a new mechanism.
