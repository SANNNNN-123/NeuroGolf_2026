# task387 — ARC-AGI f35d900a

**Rule:** Input is a width×height grid (14..18) with EXACTLY 4 coloured pixels at the corners
of a (wide×tall, each 5..11) rectangle: (row,col)=c0, (row,col+wide)=c1, (row+tall,col)=c1,
(row+tall,col+wide)=c0 (two colours, never gray, diagonally matched). Output, fully determined
by those 4 pixels: at each corner a 3×3 block of the OTHER colour with the centre cell = own
colour; gray (5) marks along the 4 frame edges at EVEN distances from each corner (top/bottom
rows at cols col+dc & col+wide−dc for dc∈{2,4,…,wide//2}; left/right cols symmetric in tall).
Off-grid stays all-zero. NOT a detection wall — exactly 4 fixed pixels, no flood/argmax/variable
components; a deterministic per-cell function of 6 recovered scalars.

**Current (stored):** 14.55 pts (ext:thbdh6285 import, mem+params≈34437) — labelled
"confirmed-infeasible" with a BLANK note; was a FALSE POSITIVE.
**Target tier:** B (data-dependent colour-index plane routed into the free bool output; a true
Tier-S pure copy is impossible because output content is synthesised, not copied).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 30×30 colf-index + 9-Where chain + And(onehot,ingrid) | B | 88248 | 86 | 13.61 | — | correct, bloated |
| 2 | sentinel off-grid (drop And + onehot plane) | B | 60288 | 87 | 13.99 | — | |
| 3 | combined masks, occ-as-centre (centre=input pixel) | B | 38024 | 92 | 14.45 | — | |
| 4 | Gather c0/c1 (drop 2 Where planes) | B | 38024 | 92 | 14.45 | — | |
| 5 | fold in-grid into colf via Wcol ch0=0.5 | B | 34424 | 94 | 14.55 | — | tie |
| 6 | crop to WORK=18 active region (1296B planes) | B | 20552 | 85 | 15.07 | 200/200 | |
| 7 | uint8 Cast+Pad of Lout (drop fp32 Pad) | B | 18176 | 85 | **15.19** | 500/500 | **adopted** |

## Best achieved
15.19 @ mem 18176 params 85 — beats prior 14.55 by **+0.64** (≫ +0.30). Fresh 500/500 isolated.

## Irreducible-floor analysis
Two forced 30×30 full planes dominate: the 1×1 Conv colf [1,1,30,30] fp32 (3600B, needed to
read colours+in-grid off the free input) and the uint8 Pad output [1,1,30,30] (ORT traces it at
3600B via upcast). Everything else is on the WORK=18 crop (1296B fp32 / 324B bool). The 5-Where
priority chain (5×1296) is the assembly floor for a 3-value (bg/gray/c0/c1/centre) index plane.
Cannot crop below 18 (grid is up to 18×18); cannot avoid the Conv (need colours from the input);
cannot avoid one 30×30 expansion (Pad to canvas). Not at a hard wall but near the practical floor.

## OPEN ANGLES (re-attack backlog)
- The traced uint8-Pad upcast to 3600B: if a Pad path that stays uint8 in the ORT trace exists,
  drops ~2700B → ~15.5. (Equal-then-Pad-bool is rejected; Cast-then-Pad costs more.)
- Could the colf Conv and the Pad share one 30×30 plane? (compute index at 30×30 once, skip crop)
  — tested worse because the 5 Wheres then run at 3600B each.

## INSIGHT (transferable)
- ⭐ A "confirmed-infeasible / BLANK-note" ledger label on a closed-form synthesis task (fixed
  pixel count, deterministic per-cell rule) is a FALSE POSITIVE — re-triage on structure, not the
  inflated stored bloat. 34437→18176 = +0.64.
- ⭐ CROP-TO-ACTIVE-REGION is the biggest single lever when the generator bounds the grid size
  (here ≤18): Conv full 30×30 once, Slice colf to WORK×WORK, run ALL Where/mask planes at
  (WORK/30)²≈0.36× cost, then Pad the small index plane back to 30×30. Cut mem 34k→18k alone.
- ⭐ Fold the in-grid mask INTO the colour Conv: weight ch0(bg)=0.5 so colf encodes
  off-grid=0 / bg=0.5 / pixel=k in ONE plane — kills a separate ReduceSum(input) in-grid plane;
  recover pixel-occupancy with a >0.75 threshold, in-grid with >0.25.
- A 3×3-block "centre = own colour, surround = other colour" pattern: fill the whole block with
  the surround colour, then restore centres for FREE by Where(occ, colf, …) — the centres ARE the
  original input pixels (occ = colf>0.75, colf = their colour). Avoids 4 separate centre masks.
- uint8 Cast+Pad of a small index plane beats an fp32 Pad even though ORT upcasts the Pad output
  in the trace, because the pre-Pad working tensors stay 1-byte.

## S11 (2026-07-03) — mech-15/pointer scout: KILL — already 084-shaped (ScatterElements-into-free-input); the +3.1KB over 084 is real added semantics (2 data-dependent corner colours + 6 geometry scalars + even-distance gray-edge indices), not bloat. Composition constraint blocks hybrids.

## S16 (2026-07-06) — time-for-cost einsum fold: eliminated fp32 crop plane (+0.239) ADOPTED
Independent-lever (einsum-vs-free-input, runtime headroom). Incumbent Sliced input[:,0:1,0:18,0:18]→bg0_f
fp32 [1,1,18,18] 1296B (largest counted plane), feeding Cast→ScatterElements base + 2 ReduceSum (row/col
bg-cell counts). FOLD: row_sum/col_sum = Einsum('bchw,c->bh'/'->bw', input, e0=[1,0..]) off FREE input
(off-grid one-hot = 0 so full-sum == 18-crop sum); base rebuilt as in-grid mask (r<H)&(c<W) via
Less(arange, H/W scalars)+Mul — the 4 pixel cells differ but are overwritten by corner 3x3 blocks →
identical. mem 4954→3842, params 105→142, cost 5059→3984, pts 16.471→16.710. fresh_verify 5000/5000 =
0/0/0 bit-identical. private-LB safe.
⭐ TRANSFERABLE: (1) any ReduceSum/ReduceMax of a Slice/crop of a single channel folds into a free-input
Einsum 'bchw,c->b{h|w}' (off-grid one-hot zero-pads for free — no crop plane). (2) a ScatterElements/base
that is just an in-grid rectangle mask is reconstructible from W/H scalars via arange-compare, never a
spatial fp32 plane. Finder = reports/scripts/fold_finder.py.
