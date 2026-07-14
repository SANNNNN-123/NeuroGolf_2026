# task212 — 8d510a79

**Rule:** Gray horizon row (color 5) spans the whole 10x10 grid at row `horizon`∈[3,6].
Source pixels are blue (1, idx=0) or red (2, idx=1) at scattered (r,c); per column at
most ONE source above the horizon and one below. Each source emits a vertical ray painting
its color from the source cell outward, stopping at a grid edge or an already-painted cell.
Direction `dr=-1 if blue else +1; dr=dr if r<horizon else -dr` ⇒ BLUE always travels AWAY
from the horizon (to the near edge), RED always travels TOWARD the horizon (stops one cell
before the gray row). Horizon row is preserved gray in the output.
**Current:** 16.32 pts, separable triangular column-OR MatMuls → colour-index L → Equal, mem 5260, params 647
**Target tier:** B (per-cell colour-index plane) — output colour is a per-cell deterministic
function but NOT row⊗col separable (sources at arbitrary positions), so A is blocked; the
10→1 colour collapse forces one fp32 entry and a 30×30 carrier → B floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-region triangular OR + Greater/Cast threshold chain | B | 7720 | 449 | 15.99 | 200/200 | MARGINAL (+0.02) |
| 2 | drop ALL thresholds (acc already {0,1}), source-mask blue / result-mask red | B | 5660 | 448 | 16.28 | 200/200 | win |
| 3 | fold ×2 into red tri-matrices, single Sum for L | B | 5260 | 647 | 16.32 | 500/500 | **adopted-candidate** |

## Best achieved
17.28 @ mem 2068 params 181 — merged MaxPool-ray bundle (2026-07-09). Prior source-owned
MatMul path was 16.32 @ mem 5260 params 647. Fresh stored eval 265/265 on current graph.

## 2026-07-09 semantic re-own
Replaced live_exact blob with readable `src/custom/task212.py` (documented slices, MaxPool rays,
ConvInteger LUT). Probed params/mem cuts (`ConstantOfShape` bg, `Expand` bg, `Where` priority,
drop-bg-channel, early-cast): none beat incumbent; task appears at local merged-bundle floor.
`source_live_reconcile --task 212`: mismatches 0. Rebuilt `networks/task212.onnx`, packed zip.

## Irreducible-floor analysis
Dominant intermediates: the 30×30 uint8 colour-index carrier (Pad, 900B) + three fp32
single-channel slices (ch1/ch2/ch5, 400B each = 1200B) + their fp16 casts (600B) + ~13 fp16
[1,1,10,10] working planes (~2600B). The 900B carrier is the B-tier floor — Equal needs L at
the full 30×30 output shape; building L at 30×30 from the start (slice→conv) costs ≥3600B, so
the small-canvas-then-pad path is optimal. The 1200B fp32 slices are the colour-extraction
entry; a 1×1 colour Conv would need the [1,10,10,10] all-channel active slice (4000B) → worse.

## KEY LEVER USED (⭐ no-threshold triangular OR)
Each prefix/suffix-OR accumulator is EXACTLY {0,1} because the generator guarantees at most
one source per (column,region,colour) — so the triangular MatMul sum over a single 1 is 0/1
and NO Greater/Cast threshold is needed. Dropping the four fill thresholds AND the two mask
thresholds cut mem 7720→5660 (+0.29 pts) in one move. Colour weights (×2 red, ×5 gray)
fold into the tri-matrix values / a final variadic Sum, removing per-colour scale planes.

## OPEN ANGLES (re-attack backlog)
- Collapse the 3 fp32 slices: blue+red are contiguous channels — one [1,2,10,10] slice (800B)
  + one fp16 cast (400B) then split, vs 3×(400+200); marginal, splitting re-adds slices.
- Eliminate the 900B carrier by routing 4 colour channels (0,1,2,5) as separate padded bool
  planes — but bg/in-grid still needs a per-cell complement, likely no net win.
- A direction-separable single pair of triangular MatMuls (merge blue+red per direction) is
  blocked: thresholding to bool loses colour identity, and blue/red use opposite matrices.

## INSIGHT (transferable)
⭐ When a generator guarantees AT MOST ONE source per scan-line/region, a prefix/suffix-OR
implemented as a triangular MatMul is already exactly {0,1} — skip every Greater/Cast threshold
(and the mask thresholds derived the same way). This is the vertical-ray analogue of task037's
diagonal fill, but cheaper because the at-most-one guarantee removes the whole boolean-cast army.
Direction = colour×side-of-barrier; a single gray barrier row splits the column into two regions
whose prefix/suffix ORs can't leak once each is masked to its own region (blue: source-mask,
red: result-mask).
