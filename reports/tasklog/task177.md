# task177 — 7468f01a

**Rule:** A solid `colors[0]` rectangle (tall×wide, both 4..8) sits at (rowoffset,coloffset)
on a 0 background; inside it a small connected creature is drawn in `colors[1]`. Outside the
rectangle is all background 0, and the rectangle fully covers its own bbox, so the bbox of all
non-background pixels == the rectangle. Output = that rectangle cropped to the top-left of a
fresh HxW grid and MIRRORED LEFT-RIGHT: `output[r][c] = input[min_row+r][min_col+(W-1-c)]`.
Colours are random per instance, so the per-cell colour value must be carried.

**Current (prior):** ~14.18 pts, tier A label (base net), mem high.
**Target tier:** B (crop-from-data-dependent-window + horizontal flip). Tier A/S blocked: the
output colour per cell is an arbitrary per-instance value read from a data-dependent mirrored
window — not a row⊗col-separable rectangle and not a fixed linear/permutation function of the
local one-hot. The flip is a column permutation coupling all columns.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf fp32 plane + bbox + flipped Gather window + label map + Equal; occupancy over ALL channels | B | 8188 | 119 | 0.0 | 0/265 | BUG: ch0 (bg) =1 everywhere → bbox = whole grid |
| 2 | same but occupancy = colf>0 (non-bg) | B | 8188 | 119 | 15.97 | 265 stored | correct |
| 3 | drop fp16 colplane, gather on fp32 colf directly, cast tiny window to uint8 | B | 6996 | 119 | 16.13 | 200/200 | KEEP |

## Best achieved
16.13 @ mem 6996 params 119 — adopted? N (orchestrator gates). Beats prior ~14.18? YES (+1.95).

## Irreducible-floor analysis
Dominant intermediates: `colf` [1,1,30,30] fp32 = 3600B (the per-cell colour index plane;
Conv output must match the fp32 input dtype — casting input to fp16 would materialize an 18000B
fp16 copy of the 10-channel input, far worse), `L` [1,1,30,30] uint8 = 900B (needed full-size for
the final 30×30 Equal; Pad rejects bool so cannot Equal-then-Pad), `Vr` [1,1,8,30] fp32 = 960B
(row-gather of an 8-row window still spans all 30 columns; gathering cols first gives [1,1,30,8],
same cost). These three are structural to "carry an arbitrary colour through a data-dependent
crop+flip window". Casting colf→uint8 to shrink Vr adds a 900B plane that exceeds the 720B saved.

## OPEN ANGLES (re-attack backlog)
- Per-channel direct Gather of input (skip colour plane + Equal): output one-hot = Gather window
  of input. But the 10-channel gather intermediate [1,10,8,30] fp32 = 9600B > colf 3600B. Only
  wins if input can be carried as bool/uint8 cheaply — but casting the free 10-ch input costs 9000B.
- Double-MatMul flip (task112/250 idiom) as a reflection matrix instead of a flipped col-Gather:
  still needs colf materialized and a data-dependent reflection matrix; no memory win, more params.
- Trim small occupancy planes (~1400B of 120B tensors) via max-min instead of ReduceSum: ~+0.1 pt,
  not worth bug risk.

## INSIGHT (transferable)
⭐ Occupancy/bbox over the 10-channel input MUST exclude channel 0: every background cell sets
channel-0=1, so ReduceMax over ALL channels marks every in-grid cell occupied (bbox = whole grid).
Use the colour-index plane `colf = sum_k k·input_k` (>0 ⇔ non-background) as the occupancy signal —
it doubles as the value plane you already need, for free. A horizontal mirror of a cropped window
is just a flipped column-index ramp `min_col + (W-1) - arange(WORK)` fed to the col Gather — no
reflection matrix needed when you're already gathering a fixed window (task036 crop idiom + flip).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.054)
**Mechanism (real diff):** the incumbent ran **3 Convs**: after cropping the
one-hot window it collapsed 10-ch → a colour-index plane `crop_f` [1,1,8,8] (via
`label_weights` [1,10,1,1]) and then re-expanded through `class_ids` [1,9,1,1] before
padding (`output_pad`). The teacher drops that collapse↔re-expand round-trip: it
keeps the cropped window as a **bool one-hot** `crop_bool` [1,10,8,8] (640B) masked
by a single `onehot_keep_mask` [1,10,1,1] **1×1 keep-conv** (2 Convs total), removing
`crop_f` (256B) + `one_hot_crop` (576B) + the `label_weights`/`class_ids` tables.
Ops: Conv 3→2; init `output_pad`→`spatial_pad`.
**Old→new:** mem 3692→3500 (−192), params 130→121 (−9). LB 3822→3621.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK.
Backup `reports/retired_networks/task177_pre_s10.onnx`; source `public_candidates/bobmyers7186/task177.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
⭐ **TRANSFERABLE:** to route a cropped/flipped one-hot window to output, carry it as
a **bool one-hot masked by a 1×1 keep-conv** rather than collapsing to a colour-index
plane and re-expanding — that kills one Conv plus the label/class const tables. This
is the cheap realization of this tasklog's "carry input as bool" open angle (no
9600B per-channel gather needed).
