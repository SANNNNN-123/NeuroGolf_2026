# task064 — 2c608aff

**Rule:** One solid `boxcolor` axis-aligned rectangle (>=3x3) plus sparse `dotcolor`
pixels (density 0.02, never inside the box) on a `b` background; grid is top-left
anchored and width,height in [8,24] (so active canvas <= 24x24). Each dot whose ROW
lies in the box's row-span shoots a horizontal ray toward the box, filling every bg
cell from itself up to (but not into) the box edge with dotcolor; each dot whose
COLUMN lies in the box's col-span does the same vertically. A dot in a corner region
(both row- and col-offset nonzero, i.e. diagonal to the box) stays a single pixel.
Equivalently: in a box-row the fill is `[leftmost-left-dot .. box_left-1] U
[box_right .. rightmost-right-dot]` (rays all terminate at the box, so per-side runs
union to one span); same per box-column. The box itself and the original dots remain.
**Current:** 15.49 pts, ext:kojimar7113, mem 13368, params 68 (crowd net; SUPERSEDED
our old 22104/14.99 custom via keep-best). Re-golf attempt 2026-06-19 (Opus): replicated
kojimar's formulation exactly (15.494/13368, fresh 200/200) and could not beat +0.3.
2026-06-28: source re-synced to the installed live graph via exact builder, so
`src/custom/task064.py` now scores the same 15.494307 / mem 13368 / params 68.
**Target tier:** detection/fill (4-direction box-blocked prefix/suffix). NOT S
(output color per cell is not a fixed linear/permutation of input — it is a
data-dependent directional fill). NOT A (row⊗col separable): horizontal fill in a
box-row depends on the DOTS in that row, which differ row-to-row, so the column
mask is not shared across rows — genuinely 2-D per direction.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (incumbent) 30x30 fp16 dyn-Conv run-signal + 4 box-blocked prefix/suffix Convs (-30 sentinel) + per-row/col span-threshold Greaters, Where->output | fill | 22104 | 66 | 14.99 | 200/200 | baseline, already optimized |
| 2 | same algo on 24x24 active-canvas slice (E0->fp16-30->Slice24), cum/Greater at 24, fp16 Pad fill back to 30 | fill | 22188 | 60 | 14.99 | (eval ok 267/267) | WASH — pad-back tax (fp16-30 pad 1800 + cast 1152 + bool30 900) exactly cancels the canvas savings on the 4 cum planes |

## Best achieved
14.99 @ mem 22104 params 66 — adopted? **N** (== current P, no gain). Beats prior
14.99? **NO** (MARGINAL/at-floor).

## Irreducible-floor analysis
Memory (22104) is a tight sum of 11 full-canvas planes that cannot be removed
without breaking correctness:
- 3600 E0f fp32 [1,1,30,30] — the entry run-signal from the dynamic-weight 1x1
  Conv (the documented 3600B fp32 colour/value-plane floor; the Conv on the fp32
  input must output fp32).
- 1800 E0 fp16 [1,1,30,30] — cast of E0f so the 4 prefix/suffix Convs run in fp16.
- 7200 = 4 x 1800 cumL/cumR/cumT/cumB fp16 [1,1,30,30] — the FOUR directional
  box-blocked sweeps are intrinsic (a dot can be left/right/above/below the box;
  all four are needed and none is a flip-reuse of another without an extra plane).
- 2400 = 2 x 1200 R,C fp32 [1,10,30,1]/[1,10,1,30] — ReduceMax keep-channel needed
  for the EXACT box discriminator (count == nrows*ncols AND count>=9). Verified
  empirically that max-count-non-bg is NOT the box in 125/5000 fresh instances
  (dots can outnumber the box), so the rectangle test (hence R,C) is mandatory.
- 6300 = 4 x 900 Greater bools + 3 x 900 OR bools — the threshold + 4-way union.
  Consolidating via Max(cumL,cumR) trades a saved bool for a new fp16 Max plane of
  equal cost (wash), so this layer is already minimal.

Reaching the +0.3 bar (15.29) needs mem+params <= ~16450 — i.e. dropping ~5700 B,
roughly three full fp16 planes. There is no reformulation that removes a direction,
the entry plane, or the exact rect detector, so the floor for a CORRECT net sits at
~21-22k => ~14.99. This is the public/incumbent floor and it is already reached.

## 2026-06-19 re-golf vs kojimar7113 (15.49, mem 13368) — AT FLOOR, no +0.3
kojimar uses a DIFFERENT, leaner formulation than our old 4-conv prefix net:
TopK(3) on per-ch counts -> rect candidate = count==row_count*col_count (Equal);
marker channel = the other candidate. Box bounds = ArgMax(first/last) of rect
row/col presence. Per box-row horizontal span = [min(first_dot_col,box_left),
max(last_dot_col,box_right)] with box COLUMNS punched out (coord->255 via
Where(rect_col_bool,255,colcoord)); symmetric per box-col. All band tests on the
24x24 active region, OR'd, Pad back to 30x30, one final Where. Replicated exactly
(src form) = 15.494 / mem 13368 / params 69, fresh 200/200.
Per-plane floor (13368, every plane structurally required):
- 3600 marker_grid fp32 [1,1,30,30] = Gather(input,marker_idx). The 4 ArgMaxes
  (first/last dot per row AND per col) NEED this 2D plane; Gather inherits input
  fp32; a uint8/fp16 copy only ADDS a plane (3600+900). IRREDUCIBLE.
- 2400 rowp+colp fp32 [1,10,30,1]+[1,10,1,30] = ReduceMax(input). Serve double duty
  (rect row/col mask + marker row/col mask + box bounds). Cropping to 24 is
  NET-NEGATIVE (reduce gives 30 then Slice adds the 24 plane; both count).
- 4032 = 7 bool 24x24: per-direction band = (coord>=low) AND (coord_punched<=high)
  = 3 planes (ge,le,and) x2 directions + 1 OR. Tried Clip+Equal to fuse the AND
  into 2 planes: ORT Clip REQUIRES SCALAR min/max (per-row [1,1,24,1] bounds crash
  "min should be a scalar") -> cannot fuse. Band block is minimal.
- 960 = 4 ArgMax int64 [1,1,30,1]/[1,1,1,30] (first/last dot positions). int64 is
  forced by ArgMax; cropping marker_grid first to shrink these adds 2304 > saved.
- 900 fill_mask bool [1,1,30,30] (Pad accepts bool at opset 13) — needed for output.
+0.3 bar (15.79) needs mem+par <= ~10010 = cut 3358B = delete one whole large block
(marker_grid / presence / band). None is removable without breaking the rule. AT FLOOR.

## 2026-06-28 high-score frontier check

Not a 20+ frontier candidate.  20+ requires `mem+params <= ~148`, but this task
requires per-row/per-column marker extrema and a 2-D ray mask.  The installed
graph already keeps the final one-hot expansion at the output; the wall is the
intermediate fill-mask construction, not a premature 10-channel output.

## OPEN ANGLES (re-attack backlog)
- Drop R,C (2400) only if a single small Conv could prove "solid rect" exactly —
  it cannot (a 2x2/3x3-solid Conv response is not an exact discriminator vs. rare
  adjacent dots; the bbox-area==count test is the only exact one and needs R,C).
  Even if removed, mem ~19700 => ~15.11, still < +0.3 (MARGINAL).
- 24x24 active canvas: tried (attempt 2) — pad-back to the mandatory 30x30 Where
  mask cancels the gain. The final mask must be [1,1,30,30] (broadcast vs input),
  and ORT Pad rejects bool, so the 24-route always pays a fp16-30 pad (1800) +
  bool30 (900) that exceeds the per-plane 30->24 savings (1800->1152).
- No Tier-A: per-row horizontal fill is not a shared col-mask (dots differ per row).

## INSIGHT (transferable)
⭐ **A box-directed ray-FILL ("dots aligned with a box shoot rays to it") is a
4-direction box-blocked prefix/suffix sum, NOT a connectivity wall** — build a run
signal `E0 = +1 at dots, -30 at box cells` (one dynamic-weight 1x1 Conv: weight
vector = dot_onehot - 30*box_onehot), then four all-ones Convs with asymmetric SAME
pads give cumL/R/T/B; `cum>0` means "a dot precedes with no box in the prefix" (the
-30 sentinel poisons any prefix crossing the box). Span-gate via
`thr = Where(rowspan, 0.5, 1e4)` from `ReduceMin(E0)<-1` so only box-rows/cols fill.
⭐ **But it sits at a ~21-22k floor** because four full directional planes + the
fp32 entry + the exact rect detector are all irreducible. **Negative result for the
24x24-active-canvas lever:** when the final op is a `Where` whose MASK must be the
full 30x30 output shape, cropping the intermediate sweeps to the active region does
NOT help — the mandatory pad-back of the fill mask to 30x30 (and ORT's bool-Pad
rejection forcing an fp16 pad + re-threshold) costs as much as the cropping saved.
The active-canvas lever only pays off when the FINAL output plane is itself the
small canvas, not when it must be padded back for a 30x30 broadcast.

## S8 (2026-07-02) — pow2-log extremes + u8 wraparound range (+0.297) ADOPTED, div 0
CAUTION LOGGED: count-profile ArgMax was WRONG here (multiple dots per row possible) — landed
instead: per-row first/last via pow2-weight einsums 'bchw,cxy,dw->bdhy' + trunc(log2) (grid≤24
⇒ sums <2^24 exact fp32; Cast fp32→u8 truncates). u8 wraparound range test
(coord−low mod 256 ≤ high−low) fuses GE/LE/And pairs. 9852+134 vs 13302+134 → 15.494→15.791.

## S11 (2026-07-03) — mech-15 finder scout: KILL — per-row fill extent depends on that row's dots (2-D per direction, tasklog-confirmed non-separable); cost = 4-direction prefix/suffix planes + exact-rect detector. No fat carrier.
