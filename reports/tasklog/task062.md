# task062 — 2bcee788

**Rule:** Size is ALWAYS 10x10. A connected 3x3 sprite (always with a column-0
cell, `unshown>0` so it has a c>0 cell too) is drawn in `color` (color in
{1,4,5,6,7,8,9}) at offset (row 1..6, col 4..6). Each sprite cell (r,c) paints
`grid[row+r][col+c]=color`; the MIRROR cell `grid[row+r][col-c-1]` gets a red(2)
marker only for c==0 (c>0 mirror cells stay background=0 in the input). The
OUTPUT paints BOTH the sprite and its full mirror across the axis at col-0.5 in
`color`, on a green(3) background. A random flip_horiz / transpose is then
applied to grid and output identically, so the reflection axis is vertical or
horizontal. Solve: C=colored mask, R=red mask; d = C-centroid − R-centroid;
VERTICAL iff |dcol|>|drow|; axis-sum s = 2·r_active + sign(d_active); reflect C
across that axis (x→s−x); M = C ∪ refl; colour = the one non-{0,2,3} channel;
output = green in-grid except M-cells = colour, off-grid all-zero.
**Current:** 14.95 pts, custom:task062 (prior version), mem 13538, params 9686
**Target tier:** A (separable reflect + closed-form scalar axis; no flood-fill,
no global argmax). Not S because the colour copies an arbitrary input colour and
the reflection genuinely needs a 2-D occupancy plane (one Gather).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop work to 10x10, BGSEL 9000-param const, Where output | A | 10238 | 10270 | 15.07 | — | passes, +0.12 (under thr) |
| 2 | derive colour profiles from C10, single combined Slice, drop 2 convs | A | 8798 | 9656 | 15.18 | — | +0.23 (under thr) |
| 3 | ⭐ Pad(incolor10[1,10,10,10]) AS the free output → kills 9000-param BG + 30x30 mask pad | A | 9898 | 666 | 15.73 | — | +0.78 |
| 4 | incolor10 + colour-onehot fp16, output declared fp16 | A | 7918 | 666 | 15.94 | 200/200 | adopt |

## Best achieved
15.94 @ mem 7918 params 666 — adopt-recommend **Y**. Beats prior 14.95 by **+0.99**.

## Irreducible-floor analysis
Dominant intermediate: `C30` = the colored-occupancy plane, [1,1,30,30] fp32 =
3600B. A 1×1 channel-contracting Conv MUST output H×W=30×30 (no spatial crop in
Conv), and the alternatives are strictly worse: slicing the 10-channel input
spatially first is [1,10,10,10] fp32 = 4000B, and casting the full input to fp16
is 18000B. So 3600B is the floor for obtaining the 2-D occupancy needed by the
reflection Gather. Second: `incolor10` [1,10,10,10] fp16 = 2000B — the genuine
10-channel × 10×10 colour expansion, already fp16 and routed into the FREE output
via Pad. Everything else is ≤200B (all reflection work is on the 10×10 active
canvas in fp16, all axis/centroid params are scalars).

## OPEN ANGLES (re-attack backlog)
- Eliminate C30 (3600B): would need a channel-contraction that emits a 10×10
  (not 30×30) plane — no opset-10 op does this without a ≥3600B intermediate.
  If a future ORT allowed a strided/cropping channel reduce, this drops to ~16.4.
- The two red-profile Convs (`Rcp30`/`Rrp30`, 120B each + 40B slices) could fold
  into the C30 reduction if red were summed alongside colour into banded planes;
  marginal (~200B → ~0.05 pts).

## INSIGHT (transferable)
⭐ **`Pad(small_plane) AS the graph output beats a full-canvas BG constant.** When
the active grid is a fixed small region (here 10×10) sitting in the top-left of
the 30×30 canvas and off-grid output cells are all-zero, build the entire
coloured result at 10×10 (`Where(M10[1,1,10,10], colvec[1,10,1,1],
bgvec[1,10,1,1])` → [1,10,10,10]) and make the **final op** `Pad(...,
value=0)` to 30×30 — the 30×30 expansion lands in the FREE `output` and the
off-grid zero-fill is exact. This removed a 9000-param background constant AND
the 30×30 mask planes (−9000 params, −2700B mem in one move: 15.18→15.73).
⭐ Combined with output declared **fp16** (colour one-hot is {0,1}, exact; harness
reads `out>0`), the colour expansion halves (15.73→15.94). General lever for any
"small-active-grid recolour on a fixed bg" task.
⭐ The harness leaves **off-grid output cells ALL-ZERO** (no channel set), NOT
channel-0=1 — verified the hard way; a BG plane that sets ch0 off-grid fails.
