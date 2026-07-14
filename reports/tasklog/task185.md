# task185 (ARC-AGI 7837ac64) — line-grid stamp -> 3x3 colour key

**Status: WIN. pts 15.75 (was 14.52, +1.23). mem 9981, params 383. eval 267/267, fresh 200/200 isolated.**
Blank-note "confirmed-infeasible" was a FALSE POSITIVE.

## Rule (verified 800/800 against generator)
Input: `size x size` cell line-grid, `linecolor` lines at spacing `sp in {2,3,4}`
(size in {6,7,10}); period p=sp+1, line rows/cols at full-grid index `i*p+(p-1)`.
A 3x3 colour key (1..9, plus 0 blanks) is stamped onto line INTERSECTIONS: cell
(row,col) colour c paints c at the 2x2 intersection block (brow+row+1..+2,
bcol+col+1..+2). OUTPUT = the 3x3 colours grid.

Decode on the intersection subgrid S (S[i,j]=colour at intersection (i,j),
linecolor/off-grid -> 0): `m2[a,b]` = the 2x2 block S[a:a+2,b:b+2] is uniformly one
NONZERO colour; `v2` = that colour. Output = v2 cropped to the bounding box of m2
(always exactly 3x3, stride 1).

## Encoding (task080/task159 lattice lever)
- colf = Conv(input, k-ramp) -> the ONE fp32 30x30 entry plane (3600B, irreducible).
- p = first full-line-row index + 1 (rowsum no-pad conv + ArgMax).
- Downsample colf at line indices i*p+(p-1) (clamped, off-grid masked via `li<A`)
  -> 10x10 fp16 subgrid S.
- **linecolor = colf[p-1, 0]** — a line cell at COLUMN 0, which is never an
  intersection, so always plain linecolor. (CRITICAL: S[0,0] is WRONG because
  brow=bcol=0 stamps the corner intersection (0,0); that was the only bug.)
- 2x2 corners via ONE pad-to-12x12 + 4 fixed Slices; m2 = all-4-equal & top-left>0.
- r0/c0 = ArgMax of m2 row-any/col-any; Gather a 3x3 stride-1 window of v2.
- Pad 3x3 uint8 (sentinel 99 outside) to 30x30; Equal(arange) -> FREE BOOL output.

## Dominant intermediate
colf32 3600B (the fp32 10->1 colour reduction, irreducible) + Sg gather [1,1,10,30]
fp32 1200B (inherits colf dtype) + the tiny 10x10 working planes. Output is FREE.

## INSIGHT
A "line-grid decode" task is the task080 lattice family: recover period p from the
first full-line row, gather line/block cells to a tiny bitmap, solve on it, route the
small result to the free output. Watch the linecolor probe — pick a line cell that is
provably NOT an intersection (column 0 of a line row), since the stamp region can
reach the corner when brow=bcol=0.

## 2026-07-03 S12 — mech-16 (runtime-parameterized stamp) KILL (측정 반증)
스카우트가 "per-sp 뱅크+mux ~660B"로 지목했으나 실측: 3개 grid conv는 free 입력의
static-stride 단일-탭 컬러 읽기 (sample+reduce 융합, 각 ~200B 출력) = 이미 검출
플로어. p 선-검출 후 runtime Gather로 통합하면 소스 평면이 필요 → full-res 컬러
평면 3600B가 강제 (인컴번트 전체 mem 1651보다 큼). 후보 실측 6468 vs 1961 (−1.19pt),
bit-identical(2500 fresh 발산 0)이지만 비용 실격. 산물: reports/candidates/task185_rts.py.
⭐경계 조건: mech-16은 "기존 평면 위 same-shape 커널 뱅크"에만 성립 — 뱅크가
free-input strided 탭 읽기면 통합이 오히려 평면을 구체화시킴. 재탐사 금지.
