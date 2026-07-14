# task051 — 25d487eb

**Rule:** "Laser beam from an arrowhead." A solid downward-narrowing TRIANGLE
(arrowhead) of colour c0 is drawn — widest at its base (width 2*depth-1),
narrowing to a single apex cell (depth in {3,4}) — with a single TIP pixel of
colour c1 at the centre of the base. A BEAM of colour c1 fires from the apex out
to the grid edge along the triangle's axis of symmetry. apply_gravity then
rotates/flips the whole figure into one of 4 cardinal orientations (arrowhead may
point up/down/left/right). INPUT = triangle + tip; OUTPUT additionally paints the
beam. Recovery rule (0 errors / 3000 fresh): tip colour = channel with pixel
count == 1 (triangle count > 1, exclude bg ch0); beam AXIS = the shorter triangle
span (base is the wide edge, so cspan>rspan => vertical/up-down, else
horizontal); beam DIRECTION = toward the apex = toward the side of the tip where
the triangle centroid lies (vertical: up iff centroid-row < tip-row; horizontal:
left iff centroid-col < tip-col). Beam fills the axis-line through the tip, in the
apex half-plane, on in-grid background cells.
**Current:** 15.00 pts (public gen:vyank6322) -> custom:task051 **15.90** pts, mem 8822, params 144
**Target tier:** **A (closed-form)** — NOT detection. The beam is a separable
axis-aligned ray whose colour COPIES the apex's; the whole output rides in the
FREE `Where(beam, apex_onehot, input)`. No colour-index/label plane is needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map L=V+tipcol*beam; tip/tri masks as [1,10,30,30] selects; fp32 planes | det | 156655 | 104 | 13.04 | (stored ok) | too big |
| 2 | derive tip/tri masks from V==colour scalars (kill [1,10,30,30]) | det | 108099 | 104 | 13.41 | ok | better |
| 3 | all per-cell planes fp16; fuse beam into 1 outer product (linerow x linecol) | det | 55149 | 106 | 13.78 | ok | |
| 4 | channel-space ROW/COL PROFILES -> tip/tri stats are 1-D, no 2-D mask plane | det | 39923 | 106 | 14.40 | ok | big cut |
| 5 | in-grid-bg via ch0 conv; fuse sentinel chain | det | 37383 | 116 | 14.47 | ok | |
| 6 | 20x20 working canvas (grids <=20x20, top-left); Pad L back to 30x30 | det | 29163 | 129 | 14.71 | ok | big cut |
| 7 | drop ch0 conv: in-grid = rowany(outer)colany from profiles | det | 24163 | 119 | 14.90 | ok | |
| 8 | fold tipcol into 1-D linecol; Where(ingridB,Lf0,15) sentinel; Where bgin | det | 21323 | 120 | 15.03 | 200/200 | prev best |
| 9 | REWRITE: count-1 apex chan; ar/ac via ArgMax; dir=sign(centroid-apex); s,q two fp16 30x30 planes; Where(beam,apex,input)->FREE fp32 out | A | 19862 | 137 | 15.10 | 265 ok | fp16 profiles |
| 10 | SEPARABLE beam: factor half-line into [1,1,30,1]/[1,1,1,30] A,B vectors selected by `vert` scalar; ONE And -> beam (kills s,q) | A | 13808 | 138 | 15.46 | 265 ok | -3.6k |
| 11 | s>=depth (depth=sqrt(tricnt+1)) skips triangle -> drop the 3600 fp32 ch0/bg0 plane; ingrid=rowany⊗colany folded into A,B | A | **8822** | **144** | **15.90** | **200/200** | **BEST** |

## Best achieved
**15.90** @ mem 8822 params 144 — adopted? **N** (build-only per task scope).
Beats prior 15.00? **Y (+0.90)**. ISOLATED FRESH **200/200** (separate spawn
process, genverify.fresh_pass reading networks/task051.onnx from disk).

## Irreducible-floor analysis (attempt 11, mem 8822)
The whole task is NOT detection — it is closed-form Tier A.  No colour-index plane
is ever built; the 10-channel output expansion rides in the FREE fp32 graph output
via `Where(beamcell[1,1,30,30], apex_onehot[1,10,1,1], input)`.  Remaining mem:
**two [1,10,30,1]/[1,10,1,30] fp32 reductions (1200 B each = 2400)** — these are
`ReduceSum(input, axis3/2)` and MUST be fp32 because the input is fp32 (the one
fp32 "entry" the prompt allows); everything downstream of them is fp16/bool.  Then
one **[1,1,30,30] bool beamcell (900 B)** — the single full-canvas plane, the final
And of the separable row/col masks.  The rest are <=600 B fp16 masked profiles and
tiny scalars.  This is at/near the closed-form floor for this rule.

## OPEN ANGLES (re-attack backlog)
- The two 1200 B fp32 reductions could perhaps be fused into one no-pad Conv
  emitting both row and col profiles, but each is fp32-input-bound; expected
  payoff small (<=1200 B -> pts ~+0.05).
- The 900 B bool beamcell is the lone 30x30 plane; a uint8/bool outer-And is
  already minimal. Could route the row/col vectors straight into the Where cond
  if ORT broadcast a [1,1,30,1]&[1,1,1,30] inside Where (it does not for bool).

## INSIGHT (transferable)
⭐ **Collapse 2-D detection to 1-D PROFILES in channel space before selecting a
colour.** Instead of building a [1,1,H,W] tip/triangle mask (and the 4-5 derived
planes per mask), reduce the input to row/col profiles `ReduceSum(input,axis3)` /
`(...,axis2)` = [1,10,H,1] / [1,10,1,W] (tiny), then select the tip/triangle
channel in profile space. tip-row/col, triangle centroid, row-span/col-span, AND
the in-grid rectangle (`rowany (outer) colany`, valid because the grid is a solid
H×W block anchored top-left) all fall out of 1-D vectors — zero 2-D mask planes.
⭐ **A separable axis-aligned ray = combine the two 1-D vectors BEFORE the outer
product.** vertical beam = vhalf(rows) ⊗ oncol(cols); horizontal = onrow ⊗ hhalf.
Pick the right pair with scalars (`linerow = vert*vhalf + (1-vert)*onrow`,
`linecol = vert*oncol + (1-vert)*hhalf`) so a SINGLE [1,1,H,W] Mul yields the beam
— and fold the output colour into the 1-D `linecol` so the same product is the
colour contribution. ⭐ **`Where(ingridB, value, sentinel)` beats additive
sentinels** (one op, reuses the boolean in-grid mask, no offgrid/sentadd planes),
and `Where(ingridB, notpres, 0)` builds in-grid-background with no extra Mul.

⭐⭐ **THIS TASK IS TIER A, NOT DETECTION** (the v8 log mis-tiered it). The whole
+0.87 jump (15.03 -> 15.90) came from three closed-form moves that the
"label-map" framing hid:
1. **Output = `Where(mask, apex_onehot[1,10,1,1], input)` straight into the FREE
   fp32 output.** Since output = input EXCEPT beam cells flip to the apex colour,
   pass the raw fp32 `input` as the Where false-branch — the entire 10-channel
   plane is then the (free) graph output, never a built copy. No colour-index
   plane, no Pad, no sentinel. (Beats casting input to uint8, which costs 9000 B.)
2. **A separable axis-aligned half-line: build the ray as `A[1,1,30,1] AND
   B[1,1,1,30]`** with the two 1-D conditions selected by a scalar `vert` via
   And/Or (ORT has NO bool-typed Where, so use And/Or not Where for bool branch
   selection). This removes BOTH 30x30 `s`/`q` parallel/perpendicular planes
   (3600 B) — only the final outer-And is a full plane.
3. **Skip the obstacle (triangle) with a COUNTED threshold, not a per-cell mask.**
   The beam starts at `s >= depth` where `depth = sqrt(tri_pixel_count + 1)`
   (apex overwrote one triangle cell, so colour-0 count = depth^2 - 1). Folding
   `s>=depth` into the 1-D `along_r/along_c >= depth16` ELIMINATES the 3600 B
   fp32 ch0/background plane that was only there to avoid painting over the
   triangle. Combined with `ingrid = rowany⊗colany` (folded into A,B), the lone
   full-canvas tensor left is the 900 B bool beamcell.
⭐ **Direction from a symmetric figure's centroid:** dir = sign(centroid(all
nonzero) - apex). A figure symmetric about its apex axis has its perpendicular
centroid component EXACTLY 0, so sign() yields a clean axis unit vector — no
neighbour inspection, no per-cell direction logic. (Verified 8000/0 fresh.)
