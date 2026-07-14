# task070 — 32597951

**Rule:** A 17×17 grid is a PERIODIC tiling of black(0)/blue(1) with period (height,width):
cell[r][c]=colors[(r%height)*width + c%width]. A (tall×wide) rectangle is placed at (row,col);
in the INPUT every non-blue cell of that rectangle is painted cyan(8) (blue cells of the
rectangle stay blue). The OUTPUT equals the input except blue(1) cells inside the rectangle
become green(3). The rectangle is exactly the bounding box of the cyan(8) cells. Verified
0/800 fails: `box=bbox(cyan); out=input with (input==1 & box)->3`.

**Current:** 13.90 pts, gen:thbdh6332, mem 66060, params 78
**Target tier:** A — closed-form bbox-fill, no flood-fill; output routed into the FREE Where output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | cyan bbox via boolean prefix/suffix-OR (triangular MatMul), green=box∧blue, Where(green_oh,input) | A | 6152 | 1191 | 16.10 | 200/200 | pass 266/266 |
| 2 | drop transposed triangulars (tril.T==triu, reuse on col side) | A | 6152 | 613 | 16.18 | — | params halved |
| 3 | ReduceMax on fp32 cyan slice directly (kill full-plane fp16 cast) | A | 5710 | 613 | 16.25 | 200/200 | adopted-as-best |

## Best achieved
16.25 @ mem 5710 params 613 — adopted? N (per instructions, not self-adopted). Beats prior 13.90? Y (+2.35).

## Irreducible-floor analysis
Dominant intermediates: the two fp32 channel slices cyan_f32 and blue_f32 ([1,1,17,17]=1156B each)
plus the padded mask green30(u8 900) + cond(bool 900). The fp32 slices are forced because Slice
preserves the input fp32 dtype and the rectangle can sit anywhere in the 17×17 canvas (cannot crop
tighter). The 30×30 padded cond is required so Where can broadcast against the [1,10,30,30] input.
The u8-pad→bool-cast path (900+900) already beats the fp16-pad→Greater path (1800+900).

## OPEN ANGLES (re-attack backlog)
- Recover cyan row/col occupancy via a per-channel batched MatMul that contracts the channel-8
  slice out of the FREE input (guide's MatMul(input,vec) lever), eliminating the cyan fp32 slice
  (~1156B) → ~+0.2.
- Same trick for the blue plane; would need blue occupancy AND the per-cell blue mask, so only the
  occupancy half is collapsible — the per-cell green=box∧blue still needs a 17×17 blue plane.

## INSIGHT (transferable)
⭐ Bounding-box-as-MASK with NO scalar argmin/argmax: a row r is in the cyan bbox iff
(prefix-OR of rowhas)(r) AND (suffix-OR of rowhas)(r). Boolean prefix/suffix-OR = MatMul with a
lower/upper triangular {0,1} matrix then Greater>0. Only TWO triangulars are ever needed because
tril.T==triu, so the column side (right-multiply) reuses them with swapped roles (UpTri for prefix,
LowTri for suffix). This turns an apparent "find the rectangle component" task into a pure
closed-form bbox-fill, beating the public CumSum-scan floor — distinct from the variable-component
global-argmax wall that genuinely floors near ~13.4.

## 2026-07-03 S12 — train-to-golf(단일 Conv SGD 컴파일) KILL
k5(cost 3008): 1.39M 패치, 19k viols 고착. 상세: reports/train_to_golf_report.md. 재탐사 금지 (mem-0 단일노드 경로는 이 태스크에서 선형분리 불가).
