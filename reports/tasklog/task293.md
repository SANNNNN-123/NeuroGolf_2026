# task293 — ba97ae07

**Rule:** A H×W grid (5..15, top-left of canvas; off-grid cells are ALL-zero, not bg-channel)
has one vertical line (cols [off0,off0+thick0), all H rows, colour colors[0]) crossing one
horizontal line (rows [off1,off1+thick1), all W cols, colour colors[1]); thick∈{1,2}. INPUT draws
`first` then `second`; OUTPUT swaps the draw order. The ONLY difference is the intersection
rectangle (horiz_rows × vert_cols): its colour flips from the input's top (second-drawn) colour to
the OTHER line's colour. output = input everywhere except the intersection.

**Current:** 16.79 pts, separable-mask Where + 2-MatMul intersection-colour read, mem 3058, params 611
**Target tier:** A (separable) — change is intersection = vert_col[1,1,1,30] ⊗ horiz_row[1,1,30,1]
routed into the FREE Where output; the per-cell new colour is a single [1,10,1,1] one-hot.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior draft: ReduceMax row/col-presence (4×1200B planes) | A | 6660 | 902 | 16.07 | 200/200 | == P |
| 1 | counts via no-pad Conv; H/W=#nonzero rows/cols; inter colour via 2-MatMul (1×1200B) | A | 3358 | 611 | 16.71 | 200/200 | win |
| 2 | vert/horiz = argmax-count col/row (drop H,W nonzero-count chain) | A | 3058 | 611 | 16.79 | 500/500 | ADOPT-candidate |

## Best achieved
16.79 @ mem 3058 params 611 — beats prior 16.07 by +0.72 (≥+0.3 ✓), fresh 500/500.

## Irreducible-floor analysis
Two dominant intermediates: (a) inter_mask [1,1,30,30] bool = 900B — the Where cond, irreducible
(Where needs a real cond tensor; a separable mask cannot be deferred into the free output). (b) byrow
[1,10,30,1] fp32 = 1200B — reading a per-channel value at the data-dependent intersection requires
materialising one per-channel spatial strip (MatMul OR Gather both cost 1200B; ORT MatMul needs
matching dtype so it stays fp32, and casting the input to fp16 costs 18000B). Everything else is
≤120B vectors. Floor ≈ 900+1200 ≈ 2.1KB.

## OPEN ANGLES (re-attack backlog)
- Kill byrow (→ ~17.2) by identifying the TOP colour from counts alone: top line is COMPLETE
  (cnt == its full area H·thick0 or W·thick1), bottom line is missing thick0·thick1 cells. Blocker
  tried: still needs to ASSOCIATE each present colour with vert vs horiz, which itself needs a
  per-channel presence-in-vert-col strip (1200B) — no cheaper association found. Raw count is NOT a
  discriminator (top is not always the larger-count colour — verified on samples).
- Read the intersection colour via ArgMax(colcount)/ArgMax(rowcount)→scalar (r*,c*) + chained Gather:
  same 1200B (Gather one row keeps all 30 cols) — no win over MatMul.

## INSIGHT (transferable)
"Swap which of two crossing lines is on top" is a SEPARABLE Where, NOT a detection task: the change
is confined to the rank-1 intersection (vert_col ⊗ horiz_row) and the new colour is the present colour
NOT at the intersection. vert_col/horiz_row fall out as argmax-of-coloured-count col/row (a full line
is strictly the max), dodging any H/W nonzero-count chain. ⭐ Reading a per-channel value at a
data-dependent (row,col) costs ONE 1200B [1,10,30,1] strip whether via 2-MatMul (contract one axis
then the other) or chained Gather — that is the irreducible price of a data-dependent per-channel pixel
read; raw per-channel counts do NOT identify which crossing line was drawn on top.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k3(cost 1288): 3.6k pos viols; k5 gain negative. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.197)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 1043 (mem 994 + params 49), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: Add/Expand x3 -> Concat fusion; fewer counted planes.
