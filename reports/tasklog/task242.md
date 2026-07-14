# task242 — 9ecd008a

**Rule:** Input is a 2*size×2*size (size=8 → 16×16) grid built with full D2 symmetry: a
symmetric `bitmap` is replicated into all four quadrants via horizontal AND vertical
reflections, so `grid[r][c] == grid[15-r][c] == grid[r][15-c]`. Every cell is a colour 1..9
(`random_color` never returns 0). A contiguous minisize×minisize (=3×3) block at (row,col) is
blacked out (set to 0). Output = the ORIGINAL 3×3 values that were blacked out, placed at the
output's top-left corner. Recovery: the hole is the only all-zero 3×3 block in the 16×16 grid;
its values come from a mirror. The vertical mirror overlaps the hole when it straddles the row
centre (rows 7/8) and the horizontal mirror when it straddles the col centre, but never both at
the same cell, so `output = max(vflip(grid), hflip(grid))` over the hole window (verified 0/3000).
**Current (prior):** 15.09 pts.
**Target tier:** B — data-dependent crop+translate of a recovered 3×3 window to the origin
(needs a Gather-shift; not separable/single-conv). The hole locate + mirror recovery are all
closed-form (no flood-fill / no global argmax), so it lands well above the detection floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | vflip-only window Gather | B | 9423 | 75 | — | — | 1 stored fail (hole straddles row-centre → vflip reads black) |
| 2 | max(vflip, hflip) window | B | 10157 | 75 | **15.77** | 200/200 | win |

## Best achieved
15.77 @ mem 10157 params 75 — adopted? N (orchestrator gates). Beats prior 15.09 by **+0.68** (≥0.3 ✓).
GENERALIZES: stored 266/266, fresh 200/200, dual-mirror recovery 0/3000 exact.

## Irreducible-floor analysis
Dominant intermediate: the `colf` colour-index plane. Entry must be fp32 [1,1,30,30] = 3600 B
(the 10→1 reduction can't go below fp32 per FLOOR_RESEARCH), but it is immediately Cast to fp16
and sliced to 16×16, so the fp32 plane is the single largest tensor. All downstream full-grid work
(vflip/hflip Slices, hole mask, ramp Where/ReduceMin) is on fp16 16×16 (=512 B each) or tiny 3×3
windows. The fp32 entry plane is the floor; the 3×3 dual-mirror Max + Pad + Equal are negligible.

## OPEN ANGLES (re-attack backlog)
- Skip the full 30×30 colf: Conv only needs the 16×16 active region. Slicing the 10-ch INPUT to
  [1,10,16,16] before the Conv would drop the colf plane to 16×16 fp32 (1024 B vs 3600 B) — a
  pre-Conv input Slice is free (Slice preserves dtype, input tensor is free). Est. ~0.4–0.6 pt.
  (Not pursued because already comfortably past +0.3; cheap to retry.)
- The two ramp-ReduceMin scalars (r0,c0) could be derived from a single CumSum-free band trick,
  but they are already fp16 1-D (16 B) — no payoff.

## INSIGHT (transferable)
⭐ "Recover a blacked-out K×K hole in a D2-symmetric grid" is closed-form Tier-B, NOT a fill/
connectivity bail: the hole is the unique all-zero K×K block (locate via ramp ReduceMin on a
hole-mask), and its values come from a mirror window. KEY GOTCHA: when the hole straddles a
symmetry centre its own mirror overlaps it and reads black — take the ELEMENTWISE MAX of the two
axis mirrors (vflip and hflip) over the K×K window; a hole never makes both read black at the same
cell, so max() recovers every cell. fp16 Max crashes under ORT_DISABLE_ALL on full planes but is
fine cast to fp32 on the tiny K×K window. Reuses the task036 recover-(r0,c0)-then-Gather-shift
crop-to-origin idiom with a sentinel-10 Pad + final Equal into the free BOOL output.
