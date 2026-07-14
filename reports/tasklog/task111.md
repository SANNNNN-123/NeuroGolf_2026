# task111 — 48d8fb45

**Rule:** Four conway sprites are stamped on a size=10 grid, all in one colour
`color` (gray excluded). A gray (=5) marker is placed at (minirows[0]-1,
minicols[0]+1). Sprite 0 occupies the 3×3 block at top-left (minirows[0],
minicols[0]); the output is that 3×3 block (`color` where the sprite has a
pixel, background elsewhere). So gray at (gr,gc) ⇒ block top-left (gr+1, gc-1);
output = the 3×3 crop of the input at rows [gr+1..gr+3), cols [gc-1..gc+2). The
gray marker sits one row ABOVE the block, never inside it, so the crop equals
the output one-hot exactly.

**Current (public):** 18.32 pts, ArgMax-locate + data-dependent Slice→Pad, mem 772, params 25
**Target tier:** B (data-dependent crop-copy) — output is a literal copy of a 3×3×10 input region; no cheaper than a single two-axis Slice.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | tighter gray bbox slice [1,1,7,8] + direct Slice→Pad crop | B | 732 | 25 | 18.37 | 200/200 | works |
| 2 | + trim index overhead (ends=starts+3, drop r1/c1) | B | 716 | 26 | 18.39 | 200/200 | best |

## Best achieved
18.39 @ mem 716 params 26 — adopted? N (only +0.07, below +0.3). Beats prior 18.32? MARGINAL.

## Irreducible-floor analysis
Two planes dominate and are both structurally required:
- **crop [1,10,3,3] fp32 = 360B.** The output IS this region (just Pad it).
  Cropping a single channel instead would need a [1,1,30,30]=3600B full-channel
  plane first (ch0/occupancy reduce or channel Slice), or a per-axis Gather that
  pays a [1,10,3,30]=3600B intermediate. The two-axis Slice to [1,10,3,3] is the
  cheapest path. A uint8 cast only ADDS a 90B plane (output is free anyway), fp16
  cast adds 180B — neither replaces the fp32 360B.
- **gray slice [1,1,7,8] fp32 = 224B.** Gray row∈[0,6], col∈[1,8] (mr0∈[1,7],
  mc0∈[0,7]); the 7×8 bbox is the minimal 2-D region carrying both coords, and
  Slice preserves fp32. Replacing it with two channel-5-picking Convs costs
  ~240B mem + 600 params (worse). Reducing the full channel-5 plane costs 1200B
  (all-channel ReduceSum) or 3600B (channel Slice) first.

Floor = 360 + 224 = 584; even with ~0 other overhead the max reachable is
≈ 25−ln(660) ≈ 18.5, below the 18.62 needed for +0.3. The task is at floor.

## OPEN ANGLES (re-attack backlog)
- None promising. The crop-copy + marker-locate structure has no separability
  (conway sprite is arbitrary), no count-collapse (full shape must be copied),
  and the marker is outside the block (can't be derived from the crop). Both big
  planes are the minimal fp32 footprints of "read a 2-D region of the input".

## INSIGHT (transferable)
⭐ A "locate-a-marker + copy a fixed-size K×K×10 block to the output" task floors
at (block fp32 bytes = 10·K·K·4) + (marker-bbox fp32 slice bytes). A two-axis
data-dependent Slice (declare the output value_info shape so static memory is
measurable) is strictly cheaper than per-axis Gather (which pays a [1,10,K,30]
intermediate) and than any single-channel extraction (which needs the 3600B full
channel plane first). uint8/fp16 casts only ADD planes here because the crop
itself is the free output. For K=3 the floor is ~700 ⇒ ~18.4-18.5 pts; if the
public net is already doing slice-locate→two-axis-Slice→Pad it is AT FLOOR — BAIL
MARGINAL fast rather than chasing dtype tricks on a region that is the output.
