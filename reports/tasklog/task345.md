# task345 — d9f24cd1

**Rule:** size-10 grid. Gray pixels (5) at (rows,cols). Red rays (2) start at the
bottom row at column `start` and rise; at cell (r,c) if the cell directly above
(r-1,c) is GRAY the ray steps RIGHT (c+=1), else UP (r-=1); every visited cell
turns red and gray is never overwritten. Input carries the gray dots plus the
red start pixels at row 9; output adds the full ray paths. Rays do not block
each other (only gray deflects).
**Current:** 15.631 pts, ext:kojimar6275, mem 9860, params 1856
**Target tier:** A (closed-form flow on a fixed 10x10 canvas — no 30x30 plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | generic 2-D recurrence, 16 linear bool iters (MatMul shifts + Greater) | A | 23700 | 339 | 14.91 | — | passes, too many planes |
| 2 | same, pure fp16 (Max=OR, Mul=AND), no per-iter threshold | A | 25500 | 340 | 14.84 | — | worse (more fp16 planes) |
| 3 | max-1-jog closed form: 2 doubling vfills + jog, MatMul shifts | A | 11100 | 640 | 15.63 | — | only matches baseline |
| 3b | same with Slice+Pad shifts (param-free) | A | 12980 | 101 | 15.52 | — | extra planes cost > param saving |
| 4 | single vfill: triangular MatMul derives jog seed, ONE doubling fill | A | **7820** | **740** | **15.945** | 500/500 | **adopted** |

## Best achieved
15.945 @ mem 7820 params 740 — adopted? Y. Beats prior 15.631? Y (+0.314).

## Irreducible-floor analysis
All intermediates are fp16 {0,1} planes on the 10x10 active corner (200B each);
there is NO 30x30 colour-index plane (the one-hot is sliced to two single
channels at 10x10). params (740) is dominated by seven 10x10 fp16 matrices
(4 doubling up-shifts + gray-down + right-shift + lower-triangular suffix-sum),
each 100 elements — irreducible because the contraction axis is the 10-wide
grid. mem ~7.8KB is ~30 fp16 10x10 planes from the doubling fill + jog
machinery.

## OPEN ANGLES (re-attack backlog)
- Replace the lower-triangular suffix-sum MatMul (T, 100 params) and the
  Sdn/Sright single-shifts with Slice+Pad to shave params, but attempt 3b shows
  the extra sliced+padded planes cost more mem than the param saving — only a
  net win if combined with dropping the corresponding plane elsewhere.
- A fully closed-form (no doubling loop) vfill via a single segmented-scan
  matrix per gray-free run would remove ~12 planes, but the segment boundaries
  are data-dependent (gray positions) so it needs a per-column gather.

## INSIGHT (transferable)
⭐ A "ray bounces off obstacles" sim is NOT a connectivity/Scan wall when the
generator SPACES the obstacles so each ray deflects a BOUNDED number of times
(here max-1-jog, measured over 5000): the whole flow collapses to
{vertical run} -> {one lateral step} -> {vertical run}, and the LAST vertical
fill (seeded by starts U jog-cells) reproduces the earlier runs for free, so
only ONE fill is needed. A gated vertical "fill-until-blocked" is a per-column
segmented OR-scan R[r]=seed OR (notgray[r] & R[r+1]) solved by a Hillis-Steele
DOUBLING scan (ceil(log2 H) rounds) with fixed up-shift MatMuls; fp16 {0,1}
keeps Max==OR / Mul==AND exact. The jog seed is recoverable WITHOUT a first
full fill via one lower-triangular suffix-sum MatMul (sclr = no-gray-above) AND
the gray-directly-above shift AND the start-column indicator.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 10 (fixed size). Flagged `red_mask` [30,30] bool 900B is a Where cond; a 10→30 broadcast is impossible. A task187-style rebuild prices 1535 vs 1495 (−0.026); a 10×10 one-hot + Pad = 1000B > 900B. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
