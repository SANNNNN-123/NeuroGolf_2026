# task024 — 178fcbfb

**Rule:** A width×height grid (each 6..15) at the canvas top-left holds scattered seed pixels of
colours {2 red, 1 blue, 3 green}, each on a distinct ROW (rows sampled w/o replacement). Red seeds
draw a full VERTICAL line down their column; blue/green seeds draw a full HORIZONTAL line across
their row. At a horiz/red crossing the HORIZONTAL colour wins (verified 12944/12944). Off-grid
(r≥height or c≥width) is all-zero. Closed form per cell: `out[r,c] = horiz_colour[r]` if row r has a
blue/green seed; else `2` if col c has a red seed; else `0` (in-grid bg). Verified 0/6000 fresh.

**Current:** 15.89 pts (prior), public label-map net.
**Target tier:** A (separable) — every output channel = (row-condition[r]) AND (col-condition[c]);
the horiz>red priority is enforced by gating red/bg on `nothoriz[r]` (no subtraction needed).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map L + Equal, no in-grid mask | B | 15300 | 34 | 0 | 0/200 | off-grid leaked ch0=1 (benchmark off-grid is all-zero, not bg) |
| 2 | label-map + off-grid sentinel (−99·offgrid) | B | 15060 | 23 | 15.38 | 200/200 | correct but 6 full fp16 planes upcast to fp32 → too heavy |
| 3 | separable RC[1,10,30,1]⊗CC[1,10,1,30] → And→output | A | 4140 | 70 | 16.65 | 200/200 | win; rowpres/colpres [1,10,30,1] bool redundant |
| 4 | drop per-channel bool planes; slice fp32 reduction per channel | A | 3900 | 70 | **16.71** | 500/500 | **best** |

## Best achieved
16.71 @ mem 3900 params 70 — adopted? N (build-agent does not adopt). Beats prior 15.89 by **+0.82** (Y).

## Irreducible-floor analysis
Dominant intermediates: the two per-channel column-collapse reductions `rowmax_f=ReduceMax(input,[3])`
and `colmax_f=ReduceMax(input,[2])`, each [1,10,30,1] fp32 = 1200B (2400B together). They are the
cheapest way to get per-colour row/col presence: any single-channel route (slice input ch → [1,1,30,30]
= 3600B) is more expensive than the all-channel [1,10,30,1] reduction. ORT keeps ReduceMax output fp32
(rejects bool/uint8 reduce), so these can't be narrowed. Everything else is tiny [1,1,30,1]/[1,1,1,30]
vectors + two [1,10,30,1]/[1,10,1,30] bool concats (300B each); the only [1,10,30,30] tensor is the FREE
bool output.

## OPEN ANGLES (re-attack backlog)
- Tier S (single Conv) is blocked: output colour at a cell depends on a NON-local OR along the whole
  row/col (line fill), not a fixed neighbourhood — a 1×1 or small Conv can't see the seed anywhere on
  the line. So separable Tier A is the admissible top here.
- Could a single MatMul contract column-axis AND channel-axis to emit <10 channels and shave the 2×1200B
  reductions? MatMul only contracts the last axis; channel reduction would need a transpose copy — net
  neutral. Not pursued.

## INSIGHT (transferable)
⭐ "Scatter seeds → full row/col line fill with a fixed crossing-priority" is a clean Tier-A separable
task, NOT a flood/detection wall: each output channel factors into rowcond[r] AND colcond[c], and the
priority (horizontal beats vertical) is enforced for FREE by gating the lower-priority channels on
`nothoriz[r]` — no ¬-mask subtraction. Build RC[1,10,30,1]/CC[1,10,1,30] by Concat of tiny per-channel
bool vectors and `And` them into the FREE bool output: zero full-canvas intermediates. Also: off-grid
cells in this benchmark are left ALL-ZERO (convert_to_numpy only iterates the grid extent), so a
label-map approach MUST sentinel off-grid — but the separable form handles it automatically via the
`colany`/`rowany` in-grid factors.
