# task145 — 6455b5f5

**Rule:** A grid (<=20x20, top-left anchored on the 30x30 canvas) is recursively
guillotine-bisected into axis-aligned rectangular leaf regions; every cut line is painted
red(2). The INPUT shows ONLY the red lines (all other in-grid cells = bg 0; off-grid cells
have all channels 0). The OUTPUT keeps the red lines, fills the leaf region(s) of MINIMUM
area cyan(8) and the region(s) of MAXIMUM area blue(1), everything else 0. Because every
region is a SOLID RECTANGLE bounded by red/border, area = (horizontal free-run length) *
(vertical free-run length) per cell — NO flood-fill / labeling / global-argmax-loop needed.
Run length = (nearest wall right) - (nearest wall left) - 1; a "wall" is red, off-grid, or
the grid border. Then amin/amax are two scalar reductions and selection is two Equal masks.

**Current:** 15.51 pts (DEPLOYED = adopted **ext:kojimar7113**, mem 13104 / params 111).
This crowd net SUPERSEDES all our prior log-doubling attempts below (best 14.21 / mem 48424).
It uses the directional MaxPool prefix-carry (task350/367 lever) — the same insight our open
angles were groping toward — at near-zero param cost. RE-GOLF VERDICT (2026-06-19): INFEASIBLE
to beat by +0.3. Need mem+params <= 9799; the directional-MaxPool design floors at ~13104.
Every plane is irreducible (see analysis at bottom). Our src/custom/task145.py is the OLD
14.21 net — do NOT adopt it (it would REGRESS 15.51 -> 14.21). Keep kojimar7113.
**Target tier:** B/detection — connectivity+global-argmax form, but the rectangle structure
collapses it to closed-form per-cell area + two reductions (well above the ~13.4 floor).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 cummax wall-dist, Where→[1,10,30,30] route | det | 209464 | 317 | 12.75 | - | correct but huge |
| 2 | +Conv colf, single colorindex→Equal output, crop W=20 | det | 79944 | 305 | 13.71 | - | better |
| 3 | uint8 cummax/min via Where(Greater/Less), uint8 tail | det | 56424 | 306 | 14.05 | - | |
| 4 | slice ch0/ch2 (input only has colours 0,2) — drop full-30 Conv/ReduceSum | det | 49624 | 304 | 14.18 | - | |
| 5 | drop area_for_max (walls=area1<max), drop blue free-And | det | 48424 | 303 | 14.21 | 200/200 | ADOPT-CANDIDATE |

## Best achieved
14.206 @ mem 48424 params 303 — beats prior 13.90 by +0.306. ISOLATED fresh 200/200.

## Irreducible-floor analysis
Dominant: the four uint8 log-doubling cumulative-max/min chains (nearest-wall positions)
= ~48 uint8 [1,1,20,20] (400B) planes + ~30 bool comparators (400B). 4 chains x 5 doubling
steps (W=20 -> ceil(log2 20)=5) x (slice+pad+Greater/Less+Where). This is the genuine 2-D
segmented-scan cost; uint8 (1B) is the dtype floor (ORT has no uint8 Max/Min, so
Where(Greater,...) is used; uint8 Sub/Mul are unsupported so the rw*rh area is fp16).

## OPEN ANGLES (re-attack backlog)
- Replace Slice+Pad shift with a single Gather over a once-augmented sentinel array to drop
  the per-step `sliced` plane (~5KB). I prototyped this but the per-step re-augment Concat
  ate the savings; a cleaner version that Gathers from a fixed augmented `cur` could net
  ~3-4KB (-> ~14.3).
- Pack the two horizontal chains (LM cummax / RM cummin) onto the batch axis so one set of
  shifts serves both (different ops though — max vs min — so only the shift is shared).
- Fold the two `-1` subtractions and 4 casts into fewer fp16 planes (~1.6KB).

## INSIGHT (transferable)
⭐ "Segment into components + global argmax + variable crop" is NOT the ~13.4 floor when the
components are RECTANGLES from a guillotine partition: per-cell component AREA = (row free-run)
x (col free-run), turning the global argmin/argmax into TWO scalar ReduceMin/Max + two Equal
masks — no flood-fill, no labeling, no NonZero. Nearest-wall distance via uint8 log-doubling
cumulative max/min: ORT has no uint8 Max/Min but `Where(Greater(a,b),a,b)` / `Where(Less,...)`
work and run at 400B (W=20) vs 800B fp16. Bake the grid BORDER into the scan as the cummax
init (0 => pos -1) and cummin pad-fill (W+1 => pos W). Wall cells get area 1, so amax =
ReduceMax(area) needs no free-mask (generator guarantees a region of area>1), but amin still
must mask walls. Big mem wins came from: input only has colours {0,2} -> slice ch0/ch2 on the
WxW active region instead of a full-30 Conv/ReduceSum; and a single uint8 colour-index plane
-> Pad(uint8) -> Equal(arange) BOOL output (no [1,10,30,30] intermediate ever materialised).

## RE-GOLF of kojimar7113 (2026-06-19) — at directional-MaxPool floor
Deployed net: Slice ch0 -> z_f32 [1,1,20,20] fp32 (1600B entry, FORCED — Slice inherits
fp32 input dtype). 4 directional MaxPool prefix-carries find nearest-red position per dir
(zero-param 1x20/20x1 kernels w/ one-sided pad): Where(z_bool,0,pos)->src (4x400) ->
MaxPool->marker (4x400) -> Add lr/ud (2x400) -> DequantizeLinear(zp=20) folds u8 sum ->
fp16 neg_width/neg_height in ONE op (2x800) -> Mul area (800) -> ReduceMax + Where-mask
area_for_min (800) + ReduceMin -> Equal max/min masks (2x400) -> 4-Where color chain
(0/2/offgrid10/blue1/cyan8, 4x400) -> Pad uint8 color30 (900) -> Equal(arange) BOOL output.
Total 13104B, every plane distinct-purpose.

Per-plane irreducibility:
- z_f32 1600 fp32: Slice of fp32 input; a 1x1 Conv->fp16 entry costs 1800(30x30)+800(slice)=worse.
- 4 src + 4 markers (3200): MaxPool needs a placed-pos plane per direction; left/right use
  opposite pos arrays + opposite pad sides (no min-pool in ORT) so cannot share. 8 planes inherent.
- u8->fp16 width/height: DequantizeLinear(zp=20) already fuses Add-offset+cast in one op; the
  uint8-width alternative needs extra Casts (Mul rejects uint8) -> strictly worse.
- area_for_min 800: wall/red cells have area=1 (pollute ReduceMin); the Where-mask is the
  single cheapest mask. ReduceMax needs no mask (walls<max) but ReduceMin does.
- inside chain ~600: off-grid (all-zero one-hot) vs in-grid-red BOTH have z_bool=False, so an
  in-grid mask is mandatory to send off-grid -> sentinel 10 (else off-grid prints red).
- color30 900: the Equal-to-output feeder; a pre-Equal expansion would be [1,10,20,20]=4000B.
- Width/height are genuinely 2-D (per-cell): guillotine partition makes each row band have a
  different vertical segmentation, so row/col PROFILES cannot replace the area plane.

To reach +0.3 needs cutting ~3.4KB (~8.5 of the 400B planes) — no such slack exists.
VERDICT: INFEASIBLE. Keep ext:kojimar7113.

## INDEPENDENT RE-CONFIRMATION (2026-06-21) — same floor, MARGINAL/INFEASIBLE upheld
Rebuilt the directional-MaxPool design from scratch in src/custom/task145.py (the old 14.21
log-doubling net was overwritten — it would have regressed, so the file now holds the GOOD
design even though it still doesn't beat deployed). Net: 15.49 pts, mem 13344, params 124,
**fresh 3000/3000 EXACT** (verified with isolated load_gen). One structural change vs the deployed
net: replaced its two 20x20 'inside'-MaxPools with **1-D occupancy profiles** —
ReduceSum(input, axes=[1,2])->[1,1,1,30] and axes=[1,3]->[1,1,30,1], >0, And — a clean ~2KB
saving on inside-detection. But that saving is exactly absorbed by the unavoidable area machinery,
landing 240B ABOVE deployed (13344 vs 13104), i.e. a tie at ~15.5. Confirms the prior verdict:
the 4 fp16 area planes (neg_w, neg_h, area, area_for_min = 3200B) + fp32 bg slice (1600B) + uint8
pad-back (900B) are an irreducible ~13K floor. area_for_min is MANDATORY: measured 743/2000
instances have a 1x1 bg room (area 1) coexisting with non-bg cells (also area 1), so the global
min cannot be taken over the raw plane. uint8 area is impossible (max room area ~324 > 255).
⭐ Reusable: the 1-D occupancy-profile inside-mask (ReduceSum over channel + one spatial axis ->
tiny vector -> Greater -> And) is the cheapest "in-grid rectangle from a top-left-anchored grid"
detector and cleanly replaces inside-MaxPools, but on an already-tight net the win is absorbed.
opset note: this design needs opset>=14 (uint8 Add for the marker sums); deployed uses opset 19.

---
## Re-verify 2026-06-30 (session 118+145) — FLOOR reconfirmed
Incumbent: mem=13104, params=111, points=15.51 (evaluate pass=267/267).
fresh_verify 145 (1500 instances): incumbent fail=0 — genuinely correct, not a re-fit.
Rule decoded: colour-2 walls partition grid into rooms; largest room's cells -> 1, smallest -> 8.
Memory breakdown confirmed: fp32 bg read z_f32=1600B (2D detection floor at 20x20), 4 fp16 area
planes (neg_w/neg_h/area/area_for_min = 3200B; area>255 so uint8 impossible; area_for_min
mandatory to exclude walls from the global min), 8 uint8 directional src/marker planes (3200B) for
per-cell room width/height via MaxPool wall-distance, uint8 colour-index chain + 900B pad-back
output carrier. Tried: int16 area (same 3200), fewer directional planes (need 2/axis for
both-side wall distance, can't fuse on pinwheel layouts), channel-wise Where output routing
(makes a 9000B [1,10,30,30] intermediate — worse than the 900B colour-index+Equal carrier).
No safe strictly-lower variant. FLOOR.

# (appended) S8 2026-07-02 — WALK-EINSUM WIN (+0.230) ADOPTED — old FLOOR verdict REFUTED
Rooms are rectangles ⇒ area = h-run × v-run. 8-plane directional-MaxPool machinery (~8.4KB) →
TWO 41-operand fp16 einsums (800B each): EXACT run-length via 3-phase monotone walk
{Stay,Right,Left} (Φ transition + shift operands) — each cell reached by exactly ONE walk ⇒
einsum value = run length (NO multiplicity; self-loop walks only give >0 reachability, and
(I+A)(I+A²) factored powers double-count). width einsum seeds the area einsum (height never
materialized). fp16 exact ≤2048 (max area 360). 9244+1248 vs 13104+111 → 15.511→15.742.
Fresh 2500+1500 fail 0 div 0; 5000 vs deployed onnx div 0. NEW TEMPLATE VARIANT for the
registry: multiplicity-free exact-count walks (use for any run-length/area/size computation).
