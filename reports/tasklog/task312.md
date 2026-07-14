# task312 — c9f8e694

**Rule:** Grid is fixed size=12 (12x12, top-left of the 30x30 canvas). Column 0 of every row r
holds a per-row pattern colour pattern[r] (2-3 random NON-gray colours; with start=1 row 0 is
background 0). The grid also holds 3-4 axis-aligned GRAY (colour 5) rectangular boxes at columns
col = randint(2, size-wide) >= 2, so boxes NEVER touch column 0. OUTPUT(r,c) = pattern[r] where
input(r,c) is gray, else input(r,c) unchanged — i.e. each gray box is recoloured by its row's
column-0 colour, broadcast across the box.
**Current:** 16.35 pts, structural over-model (mem 5700), separable-per-row escape diagnosed.
**Target tier:** S/A — pure copy/recolour, output routes into the FREE Where (no carrier plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 (prior) | Where(slice_ch5_full[3600 f32], col0[1200], input) | S | 5700 | ~6 | 16.35 | — | baseline |
| 1 | Where(crop+uint8-pad gray bool, col0, input) | S | 3720 | 22 | 16.77 | 500/500 | ADOPTED-candidate |

## Best achieved
16.77 @ mem 3720 params 22 — adopted? N (build-only). Beats prior 16.35? Y (+0.42, > +0.3).

## Irreducible-floor analysis
Three planes remain, all required by the `Where(cond[1,1,30,30], col0[1,10,30,1], input)` structure:
- col0 = column-0 slice [1,10,30,1] f32 = 1200B. Must be fp32 (Where requires X/Y dtype == fp32
  input) and 10-channel (carries the per-row colour one-hot) and 30 rows (broadcasts over cols),
  so irreducible.
- gray condition path: crop ch5+grid to [1,1,12,12] f32 (576) -> Cast uint8 (144) -> Pad uint8 to
  30x30 (900) -> Greater->bool (900) = 2520B. The two 900B uint8/bool 30x30 planes are the floor of
  the pad-back: ORT Pad REJECTS bool (verified), so the uint8 round-trip (Pad uint8 then ->bool) is
  mandatory to lift the cropped 12x12 mask to the 30x30 Where condition.
The 5700->3720 win is purely CROP-TO-ACTIVE on the dominant gray slice (fixed 12x12 grid): the naive
full ch5 slice is fp32 3600B; cropped it is 576B, and the pad-back costs only 2x900B uint8/bool.

## OPEN ANGLES (re-attack backlog)
- Eliminate one 900B plane: any way to lift a 12x12 bool mask to a 30x30 Where condition without a
  uint8 round-trip? (bool Pad rejected; uint8 Where condition rejected.) If ORT ever accepts a uint8
  Where condition, drop the final Greater (->3-?- saves 900B, ~+0.27). Currently blocked.
- col0 1200B: Where forces fp32 + 30 rows. No fp16/crop route survives the broadcast requirement.

## INSIGHT (transferable)
⭐ A `Where(full-channel-slice, col0, input)` recolour net is NOT at floor when the generator grid is
FIXED-SMALL: the dominant cost is the fp32 channel slice (3600B). Crop the slice to the active grid in
ONE Slice (channel+row+col axes together), then lift the mask back to 30x30 via uint8 Pad (bool Pad is
ORT-rejected) + Greater->bool. 5700->3720 (+0.42) with the Where output still free. The per-row colour
vector (col0 [1,10,30,1] f32) and the two 30x30 uint8/bool pad-back planes are the residual floor.
