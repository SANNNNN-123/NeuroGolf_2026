# task248 — a3df8b1e

**Rule:** Input is a 10 x W grid (W in 2..10), all black (ch0) except one blue
pixel (ch1) at bottom-left (row 9, col 0); only W varies (height fixed = 10).
OUTPUT is the bounce path of a ball starting at (9,0) going up one row per step,
column = triangle wave reflecting off col 0 and col W-1: with s = 9-r,
p = 2*(W-1), m = s mod p, c_path(r) = min(m, p-m). Blue on the path, black on the
rest of the grid, zero off-grid.
**Current (prior):** 16.22 pts
**Target tier:** A (closed-form per-row column + OneHot stamp routed into FREE output)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Equal(cramp,pcol)+Sum, fp32 planes | A | 14412 | 76 | 15.42 | - | working draft |
| 2 | fp16 downstream planes (3 full) | A | 9012 | 76 | 15.89 | - | better |
| 3 | sentinel-fold rows + band-route Sum (3 full: pathB+pathF+L) | A | 5202 | 133 | 16.42 | - | +0.20 short |
| 4 | fp16 pcol chain (Min-free) | A | 5022 | 132 | 16.45 | - | +0.23 short |
| 5 | **OneHot blue plane (drops bool + cast), 2 full fp16 planes** | A | 4602 | 105 | **16.54** | 200/200 | **PASS +0.32** |

## Best achieved
16.54 @ mem 4602 params 105 — adopted? N (build-only). Beats prior 16.22? Y (+0.32).
Fresh 200/200 (+ explicit width-sweep 2..10, 360/360).

## Irreducible-floor analysis
Two full [1,1,30,30] fp16 planes dominate (1800 each = 3600): `blueval` (the
OneHot path stamp) and `L` (the band-routed colour-index plane). Everything else
is <=240B (int64 OneHot index vectors) or tiny scalar/vector arithmetic. The two
planes are structural: the bounce path is a genuine 2-D pattern (one cell per
row at column pcol[r]) so the blue stamp is one full plane; the black-vs-off-grid
distinction needs the in-grid mask folded in (colin[1,1,1,30] + rowConst[1,1,30,1]
broadcast in a Sum) which yields the second full plane. The OneHot trick removed
the THIRD plane: a comparison `Equal(cramp,pcolG)` + `Cast` to numeric cost a
bool (900) + fp16 (1800) plane; `OneHot(pcolIdx, 30, [0,2])` produces the numeric
stamp directly in ONE fp16 plane, no bool, no cast.

## OPEN ANGLES (re-attack backlog)
- Crop the working canvas: path/black only live in r<10, c<10. But the OUTPUT is
  fixed 30x30 and L must broadcast to it, so a [1,1,10,10] plane cannot broadcast
  to [1,1,30,30] (dim 10 vs 30) without a Pad (which re-materializes a full
  plane). Likely no gain.
- Drop one of the two full planes by stamping the band value 4 directly via OneHot
  AND folding the black plane elsewhere — but black is a 2-D rect (rowConst x colin)
  that must be summed in, forcing the 2nd plane. Believed at floor for this rule.

## INSIGHT (transferable)
⭐ A "ball bounce / zigzag path" is closed-form tier-A, NOT a scan/flood wall: the
column per row is a TRIANGLE WAVE `min(s mod p, p - s mod p)` with s = (rows from
the start) and p = 2*(W-1); W comes from per-column occupancy. ⭐ Stamp a per-row
single-column pattern with **OneHot** instead of Equal-compare-then-cast: `OneHot(
colIdx[1,1,H], depth=W, values=[0,on])` materializes the numeric stamp in ONE fp16
plane with NO bool plane and NO Cast — saves a full ~900B(bool)+1800B(fp16-cast)
vs `Cast(Equal(cramp,pcol))`. ORT's OneHot kernel REQUIRES int64 indices (int32
fails "kernel not supported") but DOES accept fp16 values at opset 11 under
ORT_DISABLE_ALL, so the output plane is fp16 (half size). Out-of-range indices
(set off-grid rows to a >=depth sentinel) emit an all-off row for free.
⭐ Band-route the 10-channel one-hot through the FREE bool output as `Equal(L,
band[1,10,1,1])` where L = 2*stamp + colin + rowConst (values 4=blue->ch1,
2=black->ch0, 0/1=off->no channel); rowis a CONSTANT here (height fixed) so it is
a free init, not a reduction.
