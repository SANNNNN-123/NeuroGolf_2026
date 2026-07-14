# task377 — eb5a1d5d

## S5 win — TopK-width re-fit (LANDED +0.008)
**Before:** mem 8230, params 150, total 8380, pts 15.966.
`TopK` width K=13 (theoretical gen max). K-dependent: top_values/top_idx/inner_colors/active_inner
([1,1,13,1]) + colors4/colors (14) + colors41 (41=K+28).
**Measured:** instrumented `inner_top_rows` selection over 310k fresh instances → max true selected
rows = 6 (depth 6 seen 12×; 7 never). Bundled max 4.
**Change:** K 13→**9** (empirical 6 + margin 3); resized all K-dependent value_infos. TopK feed kept fp16.
**After: mem 8162, params 150, total 8312, pts 15.975.** evaluate fail 0; `fresh_verify 377 "" 3000`
fail 0. See [[neurogolf-topk-width-refit]].

## FLOOR siblings (same lever, no room): task46 (K16, empirical max 16/160k = tight),
task361 (K15, already BELOW safe 16 — under-provisioned), task285 (K33 = max+1, tight).

## S9 (2026-07-03) — TopK K=9→6 shrink (+0.006) ADOPTED; einsum angle FLOOR
200k fresh: max depth 7 (not 6 per old 310k note), active_rows=depth−1; but codeK core
11×11 caps correct geometry at depth 6 → K margin above 6 = pure waste, zero private-LB
downside. K=6 bit-identical on every realizable input (div 0/4000 vs live onnx; 6000+800
uncached fresh 0/0/0). mem 8162→8111. Angle (a) floor: grid_f 2916 = 27²×4 native
detection read (already cropped); alternatives ≥5832B. DO NOT re-probe reformulation.
