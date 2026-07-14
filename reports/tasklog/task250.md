# task250 — a48eeaf7 ("pull each gray pixel onto the ring around the red box")

**Rule:** A 2x2 red(2) box sits at (boxrow,boxcol)..(boxrow+1,boxcol+1) on a 0
background; gray(5) pixels are scattered.  Output = red box copied unchanged, plus
every input gray at (r,c) snapped toward the box: R=clamp(r,boxrow-1,boxrow+2),
C=clamp(c,boxcol-1,boxcol+2) — i.e. each gray lands on the 4x4 ring
[boxrow-1..boxrow+2]x[boxcol-1..boxcol+2].  Multiple grays may collide; original
gray locations are dropped.  br=boxrow=min red row, bc=boxcol=min red col.  Grid is
always exactly 10x10; colours {0,2,5}.
**Current:** was 15.61 pts (public ext:biohack_new). Now 16.92 pts, mem 3164, params 61.
**Target tier:** B (data-dependent clamp = scatter-collapse). Box position is a
global aggregate (min red row/col) and each gray's output cell is an input-derived
clamp → not S (no fixed conv/permute window). NOT row x col separable: outgray[R,C]
requires the SAME pixel to map both coords, so a rowcond ⊗ colcond would create
cross-pixel false positives → Tier A out. B is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bg/red/gray slices + clamp-matrix double-MatMul + in-grid Or chain + Pad + Equal | B | 5564 | 93 | 16.36 | 200/200 | works |
| 2 | grid is always 10x10 → drop bg slice + entire in-grid mask; build CmatT pre-transposed (drop Transpose) | B | 4264 | 84 | 16.62 | 200/200 | trim |
| 3 | clamp matrices + gray + MatMuls cast to fp16 (values {0,1}, exact) | B | **3664** | **85** | **16.77** | **200/200** | FINAL |
| 4 | clamp matrices + gray + MatMuls as uint8 QLinearMatMul; onnxsim | B | **3164** | **61** | **16.92** | **500/500** | ADOPTED |

## Best achieved
**16.92 pts @ mem 3164, params 61 — stored exact, fresh 500/500.** Adopted as
`custom:task250+qlinear+onnxsim`. Beats prior live 16.77 by +0.144.

## Irreducible-floor analysis (after attempt 2)
The data-dependent clamp is realized as a boolean-semiring double MatMul:
`outgray = Rmat @ gray @ CmatT` where Rmat[R,r]=[clamp(r,br-1,br+2)==R] and
CmatT[c,C]=[clamp(c,bc-1,bc+2)==C], both built from the scalar br/bc via a clamped
arange + Equal, then summed (collisions harmless, thresholded `>0` at the end).
After the fp16 MatMul trim the dominant intermediate is the 900 B uint8 Pad (the
30x30 label feeding the FREE final Equal) — output spans 30x30, only the Pad makes
off-canvas cells all-channel-0; irreducible. Remaining cost:
- red + gray channel slices, [1,1,10,10] fp32 = 400 B each. red is load-bearing
  twice (ReduceMax min-index for br/bc; box mask) and must be fp32 (ReduceMax
  rejects uint8; Slice preserves fp32). gray is cast to fp16 then fed to MatMul.
- Rmat + CmatT clamp matrices + gray8 + rowmapped + colmapped, now uint8 [1,1,10,10] =
  100 B each. QLinearMatMul is exact here because values are {0,1}, sums are small, and
  the only downstream use is `Greater(count, 0)`.
- the 200 B bool clamp masks + scalar reductions + 2-level Where are ≤200 B.

## OPEN ANGLES (re-attack backlog)
- **Shrink the MatMul canvas.** Output gray + box occupy only the 4x4 ring + 2x2
  box (≤4x4 = 16 cells), but the box position varies (br,bc∈2..6) and grays read
  the full 10x10, so the input slice can't shrink. A data-dependent crop to the
  ring would be a Gather (its own ≥100 B), net neutral. Untried in detail.
- **Cast clamp matrices to fp16 for the MatMul.** DONE (attempt 3), then superseded by
  uint8 QLinearMatMul in attempt 4.
- **Avoid the red fp32 slice.** br/bc need ReduceMax (fp32); the box mask reuses the
  same slice. No cheaper min-index over uint8 (ReduceMax/Min reject uint8). Blocked.

## INSIGHT (transferable)
⭐ **A per-pixel data-dependent COORDINATE REMAP that is independent in row and col
(here a clamp toward a box) is NOT Tier-A separable — but it IS a boolean-semiring
double MatMul:** outgray = Rmat @ src @ CmatT, where each remap matrix Mat[out,in]
= Equal(remap_vector[in], out_arange) is built from the scalar parameters via a
clamped/shifted arange. Sum semantics are fine because collisions only over-count
and the final `>0` threshold flattens them. This generalizes any
"scatter each marked pixel to f_row(r), f_col(c)" rule (clamp, shift, fold,
modulo) without a Gather/Loop — two 400 B fp32 MatMuls land it solidly below the
Tier-B 16.8 ceiling.
⭐ 2026-06-28 dtype update: if the double MatMul only implements boolean-semiring OR
via `sum > 0`, use uint8 QLinearMatMul instead of fp16 MatMul. This halves every remap
matrix/intermediate and is exact with scale=1/zero-point=0.
⭐ **When the generator's grid is a FIXED full size (here always 10x10), the entire
canvas is in-grid → delete the bg slice and the whole in-grid Or chain; the 30x30
Pad sentinel alone produces the off-canvas all-zero cells.** (−1300 B, +0.26 pt
here.) Always check `input sizes` over 500 fresh instances before paying for an
in-grid mask.
