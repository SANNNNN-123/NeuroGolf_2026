# task263 — a87f7484

**Rule:** A 3x3 grid is replicated into K (3..5) side-by-side stamps, laid out HORIZONTALLY
as a 3x(3K) grid (or VERTICALLY as a (3K)x3 grid when xpose=1). K-1 stamps share one "basic"
Conway-sprite shape; exactly ONE ("weird") stamp has a DIFFERENT shape. The generator
GUARANTEES `len(basicpixels) != len(weirdpixels)`, so the weird stamp is the unique-pixel-count
stamp (counts observed 3..7). Each stamp is painted in its own colour `colors[idx]` (1..9). The
OUTPUT is a 3x3 grid = the weird stamp's shape in colour `colors[weird]`, i.e. the weird stamp's
own 3x3 block moved to the top-left corner. Everything transposed iff xpose=1.

**Current:** 16.11 pts (public net), target B/closed-form
**Target tier:** B (closed-form select + corner-move; not S — output colours copy arbitrary
input colours so the 10-ch routing needs a colour-index plane, and the 30x30 output needs a
carrier).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-index 30x30 + 15x15 fp32 slice + dual-orientation count/select | B | 19109 | 59 | 15.14 | — | works, far over |
| 2 | fp16 downstream of 15x15 slice | B | 10000 | — | 15.78 | — | better |
| 3 | merge slices, drop occ full-plane, derive orientation from cnt | B | 7494 | 62 | 16.07 | — | better |
| 4 | slice colf30 -> two STRIPS (3x15,15x3) directly, multi-axis Slice | B | 6279 | 62 | 16.25 | — | big win (no 15x15) |
| 5 | unify orientations to one [5,3,3]/[5,9] path; count via vsum/colour; MatMul block | B | 5758 | 57 | **16.33** | 200/200 | **best** |

Rejected: full-fp32 (6411, fp16 halves the colf-region tensors), Gather-from-flat-colf30
(9029 — Reshape colf30->[900] DUPLICATES the 3600B plane), probe-slice hflag (5879, the 3x12
fp32 probe costs more than reusing a value-sum).

## Best achieved
16.33 @ mem 5758 params 57 — beats prior 16.11 by **+0.22** → **MARGINAL** (< +0.3). NOT adopted.
Fresh isolated 200/200 (generalises). Stored evaluate() ok, 267/267 examples pass.

## Irreducible-floor analysis
Two fixed costs dominate and cannot be removed:
- **colf30 = 3600B** — the 10->1 colour-index reduction (Conv `sum_k k*input_k`) over the
  30x30 input is the documented fp32 plane floor. Slicing the INPUT to the two strip regions
  first (in_h [1,10,3,15]=1800B + in_v [1,10,15,3]=1800B + two convs) gives the IDENTICAL sum
  (1800+1800 = 3600), so there is no escape: the per-cell colour value needs all 10 channels.
- **L = 900B** — the output is BOOL [1,10,30,30]; the 10-ch expansion is routed into the FREE
  output via `Equal(L_uint8_30x30, chan)`, and L must be a 30x30 uint8 carrier (the Pad of the
  3x3 corner block with off-grid sentinel 99). uint8 is the 1-byte floor; Pad rejects bool, so
  no narrower carrier exists; building [1,10,3,3] and padding gives a 9000B intermediate.

3600 + 900 = 4500 hard floor. The remaining ~1260B is the dual-orientation working set: two fp32
strip slices (360B) + the fp16 reshape/transpose/Where chain that NORMALISES the two layouts to
a common [5,9] (the horizontal layout is column-interleaved `r*15+3s+c`, so it REQUIRES one
transpose to group by stamp; vertical reshapes to [5,9] for free). To reach 16.41 (+0.30) the
working set would have to drop from ~1260 to ~890; every restructure (block-level Where vs
unified path, [5,9] vs [5,3,3], fp32 vs fp16) lands within ~120B of 5758 — the transpose tensor
(~90B) and the dual fp32 strips (~360B) are the binding constraints and resist removal.

## OPEN ANGLES (re-attack backlog)
- Eliminate the horizontal transpose: reorder the column-interleaved horizontal strip via a
  CONSTANT [45] Gather index instead of reshape->transpose->reshape. Tried conceptually — still
  3 tensors (flatten + gather + reshape = ~270B), no better than the transpose chain. Worth a
  concrete ORT measurement in case Gather fuses.
- Avoid materialising BOTH orientation strips: a data-dependent conditional Slice trips the
  symbolic-dim "could not be measured" trap, so this is blocked unless a 0-param Transpose of a
  SMALL (<=15x3) tensor can normalise orientation before the count — but the output-transpose
  equivariance (xpose flips the output too) makes a single-orientation pipeline emit the wrong
  (transposed) 3x3 for half the instances.
- The 900B output carrier and 3600B colour plane are both at documented floors; only the ~370B
  working-set gap stands between MARGINAL and adopt.

## INSIGHT (transferable)
⭐ "Odd-stamp-out among K side-by-side mono-colour sprites where the generator guarantees a
DIFFERENT PIXEL COUNT" is closed-form tier-B, NOT a shape-correspondence wall: the weird stamp =
the UNIQUE-count stamp, found with a tiny [5,5] self-match-count matrix (`matchcount==1`, no
sort/argmax). Pixel count of a mono-colour stamp = `value_sum / colour` (ReduceSum/ReduceMax over
the stamp, colour clamped >=1) — avoids a separate occupancy Greater+Cast plane. The xpose
orientation is resolved by NORMALISING both layouts to a common [5,9] stamp tensor and `Where`-ing
once on a scalar flag (#nonzero horizontal stamps >= 2) — but the column-interleaved horizontal
layout costs one unavoidable Transpose. The whole select+corner-move is blocked from tier-A only
by the irreducible 3600B colour-index plane + 900B 30x30 output carrier (sum 4500); the residual
dual-orientation working set (~1260B) keeps it at 16.33 (+0.22, MARGINAL).

## S16 (2026-07-06) — public bit-identical golf (llccqq624) ADOPTED
Engine public-mine loop. fresh_verify 1500 = 0/0/0 (bit-identical to incumbent). Minor cost drop
(dead-initializer / redundant-node removal), private-LB safe. Manifest updated. Backup in scratchpad.
