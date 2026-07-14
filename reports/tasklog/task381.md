# task381 — ef135b50

**Rule:** size=10 grid, red(2) boxes on black(0).  For each row, the maximal run
of non-red cells BETWEEN two red cells is painted maroon(9) UNLESS any cell in
that run has a red directly above/below (then the whole run stays black). Red
copies through; outside the 10x10 active region is black.  Generator validation
forbids a red-black-red pattern on a row and forbids maroon in the top/bottom
rows.

**Current (prompt P):** 16.75 pts.  Adopted net in manifest: 16.84
(`ext:ghiotto_conv4`, mem 3350, params 146 — Slice/MaxPool/per-row ReduceMax).
**Target tier:** B (label-map + Equal; output is a whole-row reduce, not a
per-cell neighbourhood function, so no single-Conv Tier-S).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | iterative radius-1 run-spread of "danger" within non-red runs (8 then 7 iters) + label/Equal | B | 6200 | 433 | 16.20 | 500/500 | correct but 14 spread planes dominate; below P |
| 2 | doubling spread (offsets 1,2,4,8) | B | 6200 | 833 | 16.14 | 126/200 | WRONG — offset-o jump leaks across a single red separator |
| 3 | PER-ROW block test + "red above" only (no spread) + label/Equal | B | 3250 | 44 | 16.90 | 500/500 | best (prior); beats P, edges adopted 16.84, short of +0.3 |
| 4 | STATIC ROW-MASK (rows 1..8 safe) + leftOR/rightOR maxpools + Concat-10ch carrier + Pad | B | 3200 | 129 | 16.89 | 200/200 | beats adopted +0.05; danger-detection chain fully removed |

## ⭐ NEW INSIGHT (2026-06-19 re-probe) — the danger chain is ELIMINABLE
The generator's two reject clauses pin the danger result to a STATIC row mask:
  - "Avoid maroons in top/bottom row" => row 0 and row 9 NEVER contain maroon.
  - The interior-row red-black-red validator => rows 1..8 NEVER contain a black
    between-reds gap.
Verified on 4000 fresh instances: 0 maroons in row 0/9, 0 interior black gaps.
=> a between-reds run is maroon IFF 1<=row<=8.  The ENTIRE danger machinery
(red-above/below maxpool + gap&danger AND + per-row ReduceMax + Not) collapses to
ONE constant [1,1,10,1] bool init `rowsafe` (rows 1..8 = True) AND'd with gap.
This removed ~620B (vmax fp16 200 + danger/gd/gd_u8/rowdng bools) vs attempt #3.
Also: uint8 ReduceMax is VALID at opset>=12 (fails at opset 11) — but became moot
once the reduce itself was deleted.

## Best achieved
16.89 @ mem 3200 params 129 (src/custom/task381.py).  Beats adopted 16.84 by
+0.05.  Reaches +0.3 (target 17.14, mem+params <= 2588)?  NO -> MARGINAL.

## 2026-06-29 live-frontier refresh

Current live/source is ahead of this older semantic note: **17.304697 pts @
mem 2128 params 70** (`ext:franksunp7166_65` structure captured in
`src/custom/task381.py`).  Mem profile:
`color30` Pad 900, `red8_f` Slice 320, `lower_bits` Mod 160, `color10` Concat
100, then eight 80B bool/uint8 inner tensors and small top/bottom slices.

Rechecked the apparent 900B lever.  Replacing one-channel `color10 -> Pad
color30 -> Equal(output)` with an inner bool one-hot and `Pad(output)` is not a
gain: the inner 10x10 bool one-hot is 1000B, while the current one-channel
carrier is 100B + 900B.  The 30x30 label plane is already tied with the cheapest
bool-output carrier, and the remaining 320/160/80B tensors are the compact
bit-code between-red computation.  No +0.3 mechanism remains here without a new
sub-byte/fused final-output representation.

## Irreducible-floor analysis (updated for attempt #4 @ 3200)
mem 3200 = inner_u8 Concat carrier [1,10,10,10] 1000 (10-channel output carrier;
Pad-rejects-bool so it must be uint8 before the final Pad->output; can't place the
3 nonzero channels {0,2,9} non-contiguously so all 10 channels materialize) +
red_f32 Slice [1,1,10,10] 400 (Slice preserves input fp32; red plane needed float
for MaxPool) + R/leftOR/rightOR fp16 600 (run detection: red-on-both-sides) + ~11
bool/uint8 10x10 planes 1100 (is_red, notred, lb, rb, between_b, gap, maroon,
nonblack, black + 2 carrier casts).
Floor pieces inner(1000)+red(400)=1400 are hard.  To reach +0.3 (17.14, mem+params
<= 2588) leaves only ~1059B for ALL run detection, but that needs ~600 fp16 +
~800 bool = ~1400.  Structurally short by ~800B => MARGINAL CONFIRMED.
Verified the Where-into-FREE-output alternative is WORSE: it drops the carrier
(red copies from input) but the 30x30 maroon condition costs pad-uint8(900)+
cast-bool(900)=1800 > Concat(1000), netting ~3600.  Concat carrier wins.

## OPEN ANGLES (re-attack backlog)
- Single fused op for the "between two reds" span replacing the two MaxPools
  (−~300B) — e.g. one cumulative-red signal whose sign distinguishes left/right
  simultaneously; not found exact yet.
- A 30x30 label below 900B (output shape is fixed 30x30 so likely impossible
  without a sub-uint8 carrier, which ORT upcasts).
- Confirm whether the official sanitize_model path makes bool Concat→Pad legal
  (the adopted ghiotto net uses it) — if so a Where(maroon30,onehot9,input)
  output could drop the Lm/L10 label-build, but still needs a 30x30 plane.

## 2026-06-29 adopted: free maroon overlay via bool Pad

The open angle above was real under the current opset18/ORT path: bool `Pad` is
accepted.  Replaced the final label carrier:
`fill_color/top_color/inner_color/bottom_color -> Concat color10 -> Pad color30 -> Equal`
with a free-output overlay:

`not_red8 = Not(red8); maroon8 = And(span, not_red8); maroon30 = Pad(maroon8); output = Where(maroon30, onehot9, input)`.

Why exact: the incumbent only changes non-red inner cells in the span to maroon
9.  Red cells, black cells, and the off-grid harness area are already present in
the free one-hot `input`, so they do not need to be re-materialized through a
one-channel label map.  The bool maroon mask pads from `[1,1,8,10]` to
`[1,1,30,30]` with pads `[1,0,21,20]` on axes `[2,3]`.

Stored eval: **17.41777080572354 @ mem 1908 params 55**, exact `265/265`,
improving from **17.304696865036433 @ mem 2128 params 70**.  Fresh generator:
**1000/1000**.  Adopted as `custom:task381+free-maroon-overlay`.

Transfer note: when the output differs from input only on a single semantic
overlay class, prefer a one-class bool mask plus `Where(mask, onehot_color,
input)` over rebuilding a one-channel label plane and final `Equal`.  This is
strongest when the unchanged background/red/other classes are already present
in the free input and the overlay mask can be padded as bool.

## INSIGHT (transferable)
⭐ "Fill the run BETWEEN two markers, whole-run gated by a per-cell predicate"
is NOT necessarily an iterative-flood / per-run-reduce wall: if the GENERATOR
VALIDATION forbids re-entrant patterns on the scan axis (here: red-black-red is
rejected) then a PER-ROW (per-line) ReduceMax of the block predicate is exactly
equivalent to the per-run reduce, collapsing an O(width) radius-1 spread (14
planes, ~2800B) into ONE reduce. Also: checking only "red ABOVE" (not above OR
below) sufficed because the top-row exclusion + row symmetry make the two
directions redundant under the validator. DISCRIMINATOR vs a true flood: read
the generator's reject/validate clauses, not just the draw loop — they often
constrain the input distribution enough to make a global reduce exact.
Anti-lever: gap-DOUBLING (offset 2^i) LEAKS across a single barrier cell (jumps
over a red onto a non-red in the next run); barrier-bounded spread must step by
radius 1.
