# task325 — d0f5fe59

**Rule:** Input (<=16x16, top-left) holds N separate cyan (colour 8) creatures — each a single
4-connected blob of 4..8 cells in a <=4x4 box, non-overlapping with gap>=1; N in 1..6.  Output is the
N x N grid that is cyan on its main diagonal and background elsewhere.  So the task is COUNT-CONNECTED-
COMPONENTS, then emit cyan*Identity(N).

**Current:** 14.33 pts, gen:thbdh6332, mem 41120, params 1902
**Target tier:** B — count-via-Euler is a fixed-cost scalar pipeline (no flood-fill); the N x N
identity output is a tiny K x K label padded to 30 (task184 idiom).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Euler+holes on full 30x30, 30x30 label, Where output | - | - | - | - | - | Where(bool) NOT_IMPLEMENTED under ORT_DISABLE_ALL |
| 2 | switch to Equal(L,chan); full-30x30 planes | B | 41664 | 102 | 14.36 | 266 ok | correct but heavier than prior |
| 3 | crop fg to 16x16, output as 6x6 label Pad->30 | B | 9064 | 63 | **15.88** | 500/500 | adopt-ready (+1.55) |

## Best achieved
15.88 @ mem 9064 params 63 — adopted? N (build agent does not adopt). Beats prior 14.33? **YES (+1.55)**.

## Method (exact)
b32 = Slice(input, ch8, 16x16) fp32 (512B), Cast f16.  Cubical Euler on b (f16, {0,1} products exact):
V=ReduceSum(b); Eh=Σ b[:,:-1]·b[:,1:]; Ev=Σ b[:-1]·b[1:]; F=Σ (hp_up·hp_down) reusing hp=b·b_right;
euler=V-Eh-Ev+F = (#components - #holes).  holes = Σ (empty cell with 8 filled neighbours) via ONE 3x3
ring Conv (center weight 0) == 8 AND b==0 — the ONLY hole shape this generator emits is the 3x3 ring.
N = euler + holes (scalar f16, exact for these small integer counts).  Output: 6x6 label L = 8 on the
in-grid diagonal (ondiag AND r<N), 0 on in-grid off-diagonal, 99 outside (so it matches no channel);
Pad 6x6 -> 30x30 with 99; output = Equal(L, channel-arange) BOOL.

## Irreducible-floor analysis
Dominant: the f16 16x16 working planes (b 512B, neigh-conv 512B, the four shifted-product planes
~450-480B each, plus their bool partners is8/empty/hole_b).  The fp32 entry slice is only 512B thanks to
the 16x16 geometry bound (vs the usual 3600B 30x30 colour plane).  Everything downstream of N is a tiny
6x6 (<=72B) plane.  ~9KB is essentially the cost of materialising the shifted products needed for the
five Euler/hole scalars — they cannot collapse to convs because they are PRODUCTS of shifted planes, not
sums.

## OPEN ANGLES (re-attack backlog)
- The four shifted-product planes (hp, vp, fp, neigh) are the bulk.  Eh/Ev could be folded: Eh+Ev =
  Σ b·(b_right + b_down), one Mul against a single Conv(b, [[0,1],[1,0]]-style) sum — saves ~1 plane
  (~0.05-0.1 pts).  F still needs the 2x2 product separately.
- Crop tighter: active grid is <=16x16 but often smaller; a data-dependent crop would trip the symbolic-
  dim trap, so 16 is the safe fixed bound.

## INSIGHT (transferable)
⭐ CONNECTED-COMPONENT COUNT is NOT always a flood-fill wall: when components are simply-connected (or have
a SINGLE enumerable hole shape), #components = cubical Euler V - Eh - Ev + F (cells - h/v adjacent pairs +
full 2x2 blocks) PLUS a local hole correction (here: empty cell whose 8 neighbours are all filled, one
ring-Conv).  All five terms are SCALAR reductions of shifted {0,1} products — fixed cost, no Scan/NonZero.
Verify the hole-shape set is finite by enumerating euler!=1 components from the generator first.
⭐ Geometry bound is the big lever here (generator grid <=16, output N<=6): slice the fg to 16x16 (fp32
entry 512B not 3600B) and build the variable N x N output as a fixed 6x6 label Pad->30 — turned a 41.7KB
full-30x30 build into 9KB (14.36 -> 15.88).
