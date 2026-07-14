# task091 — 3f7978a0 ("glowsticks" zoom-crop)

**Rule:** Input is an H×W grid (H,W ∈ 9..15, so ≤15×15) with scattered cyan(8)
pixels plus a marked "zoom box": its four corners are cyan and its left/right
vertical edges (interior rows) are grey(5). The OUTPUT is *exactly* the sub-grid
inside the box: `output = input[row:row+zoom_h, col:col+zoom_w]` (verified
byte-exact, 2000 fresh). Only colours {0,5,8} ever appear. The box is recovered
purely from the grey edges (grey marks ONLY the two vertical edges, interior
rows; zoom_h≥3 guaranteed ⇒ grey always present): `col=min_grey_col`,
`col_hi=max_grey_col`, `row=min_grey_row-1`, `row_hi=max_grey_row+1`,
`zoom_w=col_hi-col+1`, `zoom_h=row_hi-row+1` (0 mismatches / 5000 fresh).
**Current (this session):** 16.23 pts, custom:task091 (label-map crop on a 15×15
canvas), mem 6344, params 90. Prior 14.15 (public gen:biohack_new).
**Target tier:** B (label-map). The output is a *crop* = per-output-cell gather
of the input. Not S (not a single linear/conv/permute over a fixed window — the
offset is input-derived) and not separable A (a 2-D crop is row-gather ⊗
col-gather of *indices*, not an AND of row/col conditions). B is the highest
admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-index Conv[0..9]→[1,1,30,30], slice 15×15, 2 Gathers, Where-rect+sentinel, Pad, Equal | B | 11969→9044 | 92 | 15.88 | 200/200 | working, not minimal |
| 2 | drop grey Conv (grey = colour==5); colour-index from ch5,ch8 slices = 5·c5+8·c8 (no full Conv) | B | 7469 | 88 | 16.07 | — | uses only colours {0,5,8} |
| 3 | gather grey & cyan bit-planes (uint8) separately, build label post-gather at 15×15 | B | **6344** | 90 | **16.23** | 300/300 + 5 edge cases | best; candidate |

## Best achieved
**16.23 @ mem 6344 params 90 — fresh 300/300 + edge cases (3×3 min zoom, 14×14
max, box at (0,0) and (12,12), zoom_h=3 wide).** Beats prior 14.15 by **+2.08**.
Adopted? N (main adopts via `python -m src.adopt 91`).

## Irreducible-floor analysis
Dominant: **2 × 900 B fp32 channel slices** (`Slice(input, ch5)` and
`Slice(input, ch8)` → [1,1,15,15]) + **900 B uint8 Pad** of the 30×30 label.
- The two fp32 slices are the **entry gateway**: Slice preserves the input's
  float dtype, so the per-cell colour info cannot be obtained more cheaply —
  casting the whole [1,10,30,30] input to uint8 first is 9000 B. 15×15 is the
  true active region (H,W≤15) and the box can sit anywhere within it, so the
  slice can't shrink further.
- The 900 B Pad is irreducible: the output region is 30×30 and the final Equal
  must span it (cells outside the zoom grid must be all-zero ⇒ sentinel 10).
- Everything else is ≤225 B (uint8/bool 15×15) or scalar.

## OPEN ANGLES (re-attack backlog)
- **Drop one 900 B float slice.** cyan_crop could be derived as
  `in-region & occupied & ¬grey`, but "occupied" = `¬ch0` is itself a float
  channel slice (900 B) — net neutral. No cheaper non-contiguous {5,8} or {0,5}
  channel grab found (Slice of [5:9] = 4 ch = 3600 B).
- **Tier S long-shot:** a crop is a row-permutation MatMul ⊗ col-permutation
  MatMul with input-derived selection matrices; building Pr,Pc [15,15] uint8
  (225 each) and two MatMuls might route straight into a small intermediate, but
  MatMul needs float and the 10-channel content still has to be carried — likely
  ≥ current. Not obviously below 6344.
- Equal at 15×15 then Pad the bool [1,10,15,15] output: 2250 B > 900 Pad. Worse.

## INSIGHT (transferable)
⭐⭐ **A variable-offset CROP is a Tier-B gather, and the cheapest encoding is:
collapse to the minimal set of single-channel bit-planes, GATHER (row idx then
col idx) to shift the window to the top-left, then label-map + Equal into the
free output.** ri=row+i / ci=col+j built from input-derived scalar offsets +
arange, clipped to valid range; cells past the true extent are zeroed by an
`(i<zoom_h)&(j<zoom_w)` rect → sentinel.
⭐ **Enumerate the colours the generator actually uses** (`set()` over 20k fresh
grids). Here only {0,5,8} occur ⇒ no [0..9] colour Conv needed; two channel
slices + post-gather Where-label beat a full colour-index plane (9044→6344).
⭐ ORT **ReduceMax/ReduceSum reject uint8 and bool** (need float) and **Clip
rejects int64** — clip indices in float *before* casting to int64 for Gather.
⭐ Grey edges encode the box: 1-D `ReduceMax(channel5)` over each axis gives
row/col presence; min/max indices via `Where(present, arange, ±sentinel)` +
ReduceMin/Max give the bounding box as scalars — no 2-D bbox plane needed.
