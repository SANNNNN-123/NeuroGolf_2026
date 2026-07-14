# task384 — 2x2 mirror-tiling (kaleidoscope) of the foreground shape

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** the foreground shape is mirror-tiled into a 2×2 point-symmetric (kaleidoscope) block via ConvTranspose, then cropped to content — output size = ~2× the shape's bounding box, varying per instance.

**Cost (grader mem 540, params 53):** ops ReduceMax×4/ArgMax×2/Slice×2/Cast×3/ConvTranspose/Concat/Mul/Add/Sub. Counted intermediates: `crop7_f32` [1,1,7,7] fp32 196B, `crop7` [1,1,7,7] fp16 98B, `fg_small`/`extent`/`fg2`/`signed_small` [1,1,4,5] fp16 40B each. Params: `signed_to_output` [1,10,2,2] fp16 80B (the 2×2 mirror stamp kernel), int32 crop/fg index specs 8-12B. Output [1,10,30,30] fp16 18000B is FREE.

**Blocker class:** sprite-stamp. A ConvTranspose with a [1,10,2,2] kernel performs the 2× mirror-tiling (template/scale machinery); the bounding-box crop + ArgMax fg-extent read is the surrounding detection.

**Lever:** fp16 recast candidate — `crop7_f32` (196B) is the only fp32 plane, feeding ArgMax/ReduceMax fg-extent detection; if the values are the small colour indices it is fp16-exact → 196→98B. Gate for ArgMax tie-break equivalence.

- S12 추가: 위 fp16 recast 레버는 측정 반증(KILL) — 대상 평면이 fp32 input 직생산(Slice/Einsum)이라 producer-측 fp16 불가, Cast 경계비용이 절감을 초과 (384: +17804B, 126: +56B, 156: +44B). dtype 레버 재탐사 금지.
