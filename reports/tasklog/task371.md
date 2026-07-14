# task371 — e9614598

**Rule:** Input has exactly two blue(1) dots separated by 2*space along ONE axis
(xpose=0: same row, cols col±space; xpose=1: same col, rows row±space). Output =
input PLUS a green(3) plus-shape (centre + its 4 orthogonal neighbours = the L1
ball of radius 1) centred at the MIDPOINT of the two blue dots. Grids fit in 14×14
(width∈{10,12,14}, height∈{6..14}, xpose only swaps).
**Current (prior public):** 15.51 pts.
**Target tier:** A/B — closed-form midpoint + separable L1-ball mask routed into the FREE Where output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | on-disk draft: fp32 centroid + dist<1.5 + Where | B | 15800 | 81 | 15.33 | 200 | below P |
| 1 | fp16 all working planes (no crop) | B | 8588 | 84 | 15.93 | 200 | +0.42 |
| 2 | crop blue→14×14, fp16 dist 14, Pad→30 | B | 5320 | 53 | 16.41 | 200 | +0.90 |
| 3 | crop blue→14×14 for profiles; fp16 dist over FULL 30 ramps; Less→cond; no Pad | B | 3968 | 112 | 16.69 | 500 | ADOPT +1.18 |

## Best achieved
16.69 @ mem 3968 params 112 — beats prior 15.51 by **+1.18**. fresh 500/500.

## Key reformulation
Plus centre = (mean blue row, mean blue col), integer & orientation-AGNOSTIC: the
two dots share one coordinate and straddle the other by 2*space, so the centroid is
the centre in BOTH xpose values — no orientation branch. Plus = (|r−mr|+|c−mc|)<1.5
(the radius-1 L1 ball). Build separably: dr=|rowidx−mr| [1,1,30,1], dc=|colidx−mc|
[1,1,1,30], dist=dr+dc broadcasts to the single 30×30 fp16 plane; Less→bool cond;
output=Where(cond, green_onehot[1,10,1,1], input). Blue dots are at L1 distance
space≥3, never inside the plus → survive the Where unchanged; green is a constant
one-hot (input has no green).

## Irreducible-floor analysis
Dominant intermediates: dist 30×30 fp16 (1800B) + cond bool (900B) = 2700B. The
Where cond MUST be a 30×30 bool (900B), and producing it via Less needs one 30×30
float operand (the L1 sum). Separable bool decompositions (square AND-NOT-corners,
or term1 OR term2) all need 3×900B bool planes = 2700B — no better than one fp16
sum (1800) + cond (900). The 14×14 fp32 channel-1 slice for the profiles adds 784B;
a MatMul contraction off the free input ([1,10,30,1]=1200B) is larger, so the slice
stays. So ~2700+784 ≈ floor for this style.

## OPEN ANGLES (re-attack backlog)
- Eliminate the 784B blue slice by reading mr,mc as scalars via a single chained
  MatMul/Conv that picks channel 1 and weights by coord without a [1,1,14,14] plane
  (every attempt so far produced a ≥1200B intermediate — net loss).
- Avoid the 30×30 fp16 dist by emitting cond from a uint8/bool Pad of a 14×14 mask
  (blocked: opset-10 Pad rejects bool/uint8; fp16 Pad gives the same 1800B).

## INSIGHT (transferable)
"Place a fixed local stamp at the midpoint of two markers" is closed-form &
orientation-free: centre = channel centroid (Σ coord·profile / count), and a PLUS =
the radius-1 L1 ball `(|r−mr|+|c−mc|)<1.5` built as ONE fp16 outer-sum plane routed
into the FREE Where output — no per-cell colour plane, no xpose branch. Crop the
input slice to the generator's true active region (here 14×14) to keep the entry
plane cheap while the cond stays full 30×30.
