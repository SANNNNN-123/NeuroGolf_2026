# task126 — drop yellow floor-marker under each comb shape

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** input holds one or more open-top comb/U-shapes (3-wide) of a single colour; output copies the input and drops a yellow (4) marker in the BOTTOM row of the grid at the centre column of each shape's opening (a marker "falls to the floor" under each shape).

**Cost (grader mem 590, params 15):** ops Einsum/ReduceMax/ScatterND/Xor/Equal/Cast×3/Concat×3. Counted intermediates: `updates` [2,30] fp32 240B, `column_sums` [1,30] fp32 120B, `updates_bool` [2,30] 60B, `indices` [2,3] int64 48B, three [1,30] masks 30B. Params: `channel_selector` [10] fp32 40B, `prefix_idx` [2,2] int64 32B. Output [1,10,30,30] fp32 36000B is FREE (ScatterND target).

**Blocker class:** already-at-floor (column-reduce + marker scatter). No full-canvas working plane; cost is 1-D column tensors (≤30 wide) plus a tiny scatter index. This is a compact column-detect→floor-marker mechanism, not a carrier or per-cell read.

**Lever:** fp16 recast candidate — `updates` (240B) carries a channel/colour selector and `column_sums` (120B) are integer column counts ≤30 (fp16-exact); recasting both to fp16 → ~180B saved if Einsum/ScatterND tolerate fp16 operands. Worth a bit-identical gate probe.

- S12 추가: 위 fp16 recast 레버는 측정 반증(KILL) — 대상 평면이 fp32 input 직생산(Slice/Einsum)이라 producer-측 fp16 불가, Cast 경계비용이 절감을 초과 (384: +17804B, 126: +56B, 156: +44B). dtype 레버 재탐사 금지.
